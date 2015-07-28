#!/usr/bin/env python
import numpy as np
from dedalus.public import *
import h5py

import os
import sys

import equations

from mpi4py import MPI
CW = MPI.COMM_WORLD

class FC_onset_solver:
    '''
    This class creates a polytropic atmosphere as defined by equations.py,
    then uses EVP solves on that atmosphere to determine the critical Ra of
    convection by searching for the first value of Ra at which an unstable
    (e^[positive omega]) mode arises.
    '''

    def __init__(self, ra_range, profiles=['u','w','T1'], nx=64, nz=64, Lx=30, Lz=10, epsilon=1e-4, gamma=5/3,
        n_rho_cz=3.5, constant_diffusivities=True, constant_kappa=False,
        grid_dtype=np.complex128, comm=MPI.COMM_SELF,
	    out_dir=''):
        '''
        Initializes the atmosphere, sets up basic variables.

        NOTE: To run in parallel, set comm=MPI.COMM_SELF.
        '''
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_rho_cz = n_rho_cz
        self.constant_diffusivities=constant_diffusivities
        self.constant_Kappa=constant_kappa

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                gamma=gamma, grid_dtype=grid_dtype, n_rho_cz=n_rho_cz,
                constant_diffusivities=constant_diffusivities,
                constant_kappa=constant_kappa,
                comm=comm, epsilon=epsilon)

        self.x = self.atmosphere.x
        self.z = self.atmosphere.z

        self.ra_range = ra_range
        self.profiles = profiles

        self.out_file_name = 'evals_eps_{0:.0e}_ras_{1:04g}-{2:04g}_nrho_{3:.1f}'.format(self.epsilon, self.ra_range[0], self.ra_range[-1], self.n_rho_cz)

        out_dir_base = sys.argv[0].split('/')[-1].split('.py')[0] + '/'
        self.out_dir = out_dir + out_dir_base
        if not os.path.exists(self.out_dir) and CW.rank == 0:
            os.makedirs(self.out_dir)
        if CW.size > 1:
            self.local_file_dir = self.out_dir + self.out_file_name + '/'
            if not os.path.exists(self.local_file_dir):
                os.makedirs(self.local_file_dir)
            self.local_file_name = self.local_file_dir + 'proc_{}.h5'.format(CW.rank)
            self.local_file = h5py.File(self.local_file_name, 'w')

    def build_solver(self, ra, pr=1):
        '''
        Sets up the EVP solver for a given value of ra.
        '''
        self.atmosphere.set_eigenvalue_problem(ra, pr,\
                    include_background_flux=False)
        self.atmosphere.set_BC()
        self.problem = self.atmosphere.get_problem()
        self.solver = self.problem.build_solver()
        return self.solver

    def solve(self, wave, ra, pr=1):
        '''
        Solves the system at the wavenumber index specified by wave and at
        the given ra.  Stores eigenvalues.
        '''
        self.build_solver(ra, pr=pr)
        self.solver.solve(self.solver.pencils[wave])
        self._eigenvalues_r = np.real(self.solver.eigenvalues)
        self._eigenvalues_i = np.imag(self.solver.eigenvalues)

    def get_positive_eigenvalues(self):
        '''
        Filters real and imaginary parts of eigenvalues and stores the
        finite, positive parts of these arrays plus their indices
        '''
        indices_i = np.where(self._eigenvalues_i > 1e-10)
        indices_i_new = np.where(self._eigenvalues_i[indices_i] != np.inf)
        indices_r = np.where(self._eigenvalues_r > 1e-10)
        indices_r_new = np.where(self._eigenvalues_r[indices_r] != np.inf)
        indices_i_new = np.asarray(indices_i)[0,indices_i_new]
        indices_r_new = np.asarray(indices_r)[0,indices_r_new]


        self._positive_eigenvalues_i = np.zeros([2, len(indices_i_new[0])])
        self._positive_eigenvalues_r = np.zeros([2, len(indices_r_new[0])]) 
      
        self._positive_eigenvalues_i[0,:] = indices_i_new
        self._positive_eigenvalues_i[1,:] = self._eigenvalues_i[indices_i_new]
        self._positive_eigenvalues_r[0,:] = indices_r_new
        self._positive_eigenvalues_r[1,:] = self._eigenvalues_r[indices_r_new]

    def set_profiles(self, eigenvalue_index):
        '''
        Sets the profiles of the solution (u, u_z, w, w_z....) to the
        ith eigenvector, where i is specified by eigenvalue_index
        '''
        self.eigenvalue_state = eigenvalue_index
        self.solver.set_state(eigenvalue_index)
        self.field_dict = self.solver.state.field_dict
        self.field_keys = list(self.field_dict.keys())

    def get_profile(self, eigenvalue_index, key='w'):
        '''
        Returns the 2D profile of the variable specified by key at the
        eigenvector specified by eigenvalue_index
        '''
        if not hasattr(self, 'field_keys') or \
                self.eigenvalue_state != eigenvalue_index:
            self.set_profiles(eigenvalue_index)
        if key in self.field_keys:
            return self.field_dict[key]
        else:
            print('Error: Unknown field key')
            return
        
    def get_unstable_modes(self, ra, wavenumber, profiles=['w', 'T1']):
        '''
        Solves the system at the specified ra and horizontal wavenumber,
        gets all positive eigenvalues, and returns the specified profile,
        index, and real eigenvalue component of all unstable modes.
        '''
        self.solve(wavenumber, ra)
        self.get_positive_eigenvalues()

        sample_profile = self.get_profile(0, key=profiles[0])
        shape = [self._positive_eigenvalues_r.shape[1], len(profiles), sample_profile['g'].shape[0], sample_profile['g'].shape[1]]
        unstables = np.zeros(shape, dtype=np.complex128)
        #Substitution is omega*f.
        for i in range(shape[0]):
            index = self._positive_eigenvalues_r[0,i]
            for j in range(len(profiles)):
                unstables[i,j,:] = self.get_profile(index, key=profiles[j])['g']

        return (unstables, self._positive_eigenvalues_r)
    
    def solve_unstable_modes_parallel(self, ra, **kwargs):
        '''
        Solves the EVP at all horizontal wavenumbers for the specified value of
        ra and collects all unstable modes at that ra.
        '''
        kxs = int(nx/2)
        kxs_global = np.arange(kxs)
        kxs_local = kxs_global[CW.rank::CW.size]

        returns_local = [self.get_unstable_modes(ra, kx, **kwargs) for kx in kxs_local]
        if CW.rank == 0:
            returns_global = [0]*kxs
            returns_global[CW.rank::CW.size] = returns_local
        
        CW.Barrier()
        for i in range(CW.size):
            if i == 0:
                continue
            if i == CW.rank:
                CW.send(returns_local, dest=0, tag=i)
            elif CW.rank == 0:
                vals = CW.recv(source=i, tag=i)
                returns_global[i::CW.size] = vals
        CW.Barrier()
        if CW.rank == 0:
            return returns_global
        else:
            return True

    def plot_onsets(self, ra_range, **kwargs):
        if CW.rank == 0:
            import matplotlib.pyplot as plt
        evals = dict()
        prof_returns = dict()
        if 'profiles' in kwargs.keys():
            profiles = kwargs['profiles']
        for ra in ra_range:
            returned = self.solve_unstable_modes_parallel(ra, **kwargs)
            if CW.rank == 0:
                for i in range(len(returned)):
                    item = returned[i]
                    if np.asarray(item[0]).shape[0] > 0:
                        eigval_array = np.asarray(item[1])
                        eigvals = []
                        for j in range(eigval_array.shape[1]):
                            eigval = eigval_array[:,j][1]
                            eigvals.append((ra, eigval))
                        if str(i) not in evals.keys():
                            evals[str(i)] = eigvals
                        else:
                            for pack in eigvals:
                                evals[str(i)].append(pack)
                        for j in range(np.asarray(item[0]).shape[0]):
                            profiles_returned = np.asarray(item[0])[j,:]
                            count = 0
                            for key in profiles:
                                new_key = "{0}_".format(i) + key
                                if new_key not in prof_returns.keys():
                                    prof_returns[new_key] = [profiles_returned[count,:]]
                                else:
                                    prof_returns[new_key].append(profiles_returned[count,:])
                                count += 1
                            
        if CW.rank == 0:
            print('saving output')
            import h5py

            outstring = self.out_dir+'evals_eps_{0:.0e}_ras_{1:04g}-{2:04g}_nrho_{3:.1f}'.format(self.epsilon, ra_range[0], ra_range[-1], self.n_rho_cz)
            f = h5py.File(outstring +'.h5', 'w')
            file_keys = []

            plt.figure(figsize=(15,10))
            for key in evals.keys():
                ras = []
                evals_local = []
                for pack in evals[key]:
                    ras.append(pack[0])
                    evals_local.append(pack[1])
                plt.plot(ras, evals_local, label='wavenum='+key)
#                print('wavenum {0} has {1} at {2}'.format(int(key), ras, evals_local))
                dict_key_ra = '{0}_ras'.format(int(key))
                dict_key_eval = '{0}_evals'.format(int(key))
                f[dict_key_ra] = ras
                f[dict_key_eval] = evals_local
                file_keys.append(dict_key_ra)
                file_keys.append(dict_key_eval)
                for p_key in profiles:
                    my_key = "{0}_".format(int(key))+p_key
                    f[my_key] = np.asarray(prof_returns[my_key])
                    file_keys.append(my_key)

            file_keys.append('x')
            file_keys.append('z')
            f['x'] = self.x
            f['z'] = self.z
            print(file_keys)
                    

            #Storing string list from http://stackoverflow.com/questions/23220513/storing-a-list-of-strings-to-a-hdf5-dataset-from-python
            asciiList = [n.encode('ascii', 'ignore') for n in file_keys]
            f.create_dataset('keys', (len(asciiList),1), 'S10', asciiList)
            #plt.axhline(self.epsilon, linestyle='dashed', color='black', label=r'$\epsilon$')
            plt.legend(loc='lower left')
            plt.xlabel('Ra')
            plt.ylabel(r'$Re(\omega)$')
            plt.yscale('log')
            plt.savefig(outstring + '.png', dpi=200)

            
    def find_onset_ra(self, start=1, end=4, tol=1e-2):
        '''
        Steps in log space (specified by start (10**start) and end (10**end))
        to find the highest value of Ra at which there are no unstable modes,
        returns the critical rayleigh number.  Tol determines the accuracy
        of the solve.
        '''
        N = 5
        first = True
        while(end/start > 1+tol):
            ras = np.logspace(start, end, N)
            if not first:
                ras = ras[1:]
            else:
                first = False
            for i in range(N):
                if CW.rank == 0:
                    print('Solving at Ra {0}...'.format(ras[i]))
                unstables = self.solve_unstable_modes_parallel(ras[i])
                if CW.rank == 0:
                    for j in range(len(unstables)):
                        if unstables[j][1].shape[1] > 0:
                            end = np.log10(ras[i])
                            break
                    if end != np.log10(ras[i]):
                        start = np.log10(ras[i])
                info = CW.bcast(obj=[start,end], root=0)
                start = info[0]
                end = info[1]
                if end == np.log10(ras[i]):
                    break
        crit = (end+start)/2
        if CW.rank == 0:
            print('Critical Ra of {0:.4g} found'.format(10**crit))
        return 10**crit

    def full_parallel_solve(self, stitch_files=True):
        kxs = int(nx/2)
        kxs_global = list(np.arange(kxs))
        tasks = kxs_global * len(self.ra_range)
        for i in range(len(self.ra_range)):
            for j in range(kxs):
                tasks[i*kxs + j] = (tasks[i*kxs + j], self.ra_range[i])
        my_tasks = tasks[CW.rank::CW.size]

        my_keys = []
        for task in my_tasks:
            wave = task[0]
            ra = task[1]
            returns = self.get_unstable_modes(ra, wave, profiles=self.profiles)
            profiles = returns[0]
            eigenvalues = returns[1]
            key = '{0:07.2f}_{1:04d}'.format(ra, wave)
            key_prof = key + '_prof'
            key_eigenvalues = key + '_eig'
            if eigenvalues.shape[0] > 0:
                self.local_file[key_prof] = profiles
                self.local_file[key_eigenvalues] = eigenvalues
                my_keys.append(key_prof)
                my_keys.append(key_eigenvalues)
        asciiList = [n.encode('ascii', 'ignore') for n in my_keys]
        self.local_file.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
        self.local_file.close()
        CW.Barrier()
        if stitch_files and CW.rank == 0:
            self.stitch_files()

    def stitch_files(self):
        filename = self.out_dir + self.out_file_name + '.h5'
        file_all = h5py.File(filename, 'w')
        keys = []
        for i in range(CW.size):
            partial_file = self.local_file_dir + 'proc_{}.h5'.format(i)
            file_part = h5py.File(partial_file, 'r')
            keys_part = []
            for raw_key in file_part['keys'][:]:
                key = str(raw_key[0])[2:-1]
                keys.append(key)
                keys_part.append(key)
            for key in keys_part:
                file_all[key] = np.asarray(file_part[key][:])
        asciiList = [n.encode('ascii', 'ignore') for n in np.sort(keys)]
        file_all.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
            

if __name__ == '__main__':
    eqs = 7
    nx = 32
    nz = 32
    Lx = 100
    epsilon=1e-4
    out_dir = './'#'/regulus/exoweather/evan/'
    n_rho_cz = 20


    start_ra = 65
    stop_ra  = 66
    res = 0.5
    steps = (stop_ra - start_ra)/res + 1


    solver = FC_onset_solver(np.linspace(start_ra, stop_ra, steps), profiles=['u','w','T1'],nx=nx, nz=nz, Lx=Lx, n_rho_cz=n_rho_cz, epsilon=epsilon, comm=MPI.COMM_SELF, out_dir=out_dir, constant_kappa=True)
    solver.full_parallel_solve()
#    solver.plot_onsets(np.linspace(start_ra, stop_ra, steps), profiles=['w', 'T1'])
if False:
    returned = solver.solve_unstable_modes_parallel(ra)#find_onset_ra(start=1, end=3)
    if CW.rank == 0:
        for i in range(len(returned)):
            item = returned[i]
            zs, xs = np.meshgrid(solver.z, solver.x)
            if np.asarray(item[0]).shape[0] > 0:
                eigenvalue = np.asarray(item[1])
                string = 'w field; '
                for j in range(eigenvalue.shape[1]):
                    array = eigenvalue[:,j]
                    string += 'Eval: {0:.4g} at index {1}'.format(array[1], array[0])
                    string += '; wavenumber {0}'.format(i)
                print(string)
