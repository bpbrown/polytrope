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

    def __init__(self, nx=64, nz=64, Lx=30, Lz=10, epsilon=1e-4, gamma=5/3,
        constant_diffusivities=True, constant_kappa=True,
        grid_dtype=np.complex128, comm=MPI.COMM_SELF):
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
        self.constant_diffusivities=constant_diffusivities
        self.constant_Kappa=constant_kappa

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                gamma=gamma, grid_dtype=grid_dtype,
                constant_diffusivities=constant_diffusivities,
                constant_kappa=constant_kappa,
                comm=comm, epsilon=epsilon)

        self.x = self.atmosphere.x
        self.z = self.atmosphere.z
        self.crit_ras = []

        self.out_dir = './FC_onset_solve/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

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
        
    def get_unstable_modes(self, ra, wavenumber, profile='w'):
        '''
        Solves the system at the specified ra and horizontal wavenumber,
        gets all positive eigenvalues, and returns the specified profile,
        index, and real eigenvalue component of all unstable modes.
        '''
        self.solve(wavenumber, ra)
        self.get_positive_eigenvalues()

        sample_profile = self.get_profile(0, key=profile)
        shape = [self._positive_eigenvalues_r.shape[1], sample_profile['g'].shape[0], sample_profile['g'].shape[1]]
        unstables = np.zeros(shape, dtype=np.complex128)
        #Substitution is omega*f.
        for i in range(shape[0]):
            index = self._positive_eigenvalues_r[0,i]
            unstables[i,:] = self.get_profile(index, key=profile)['g']

        return (unstables, self._positive_eigenvalues_r)
    
    def solve_unstable_modes_parallel(self, ra):
        '''
        Solves the EVP at all horizontal wavenumbers for the specified value of
        ra and collects all unstable modes at that ra.
        '''
        kxs = int(nx/2)
        kxs_global = np.arange(kxs)
        kxs_local = kxs_global[CW.rank::CW.size]

        returns_local = [self.get_unstable_modes(ra, kx, profile='w') for kx in kxs_local]
        if CW.rank == 0:
            returns_global = [0]*kxs
            returns_global[CW.rank::CW.size] = returns_local

        for i in range(CW.size):
            if i == CW.rank and i != 0:
                CW.send(returns_local, dest=0, tag=i)
            else:
                if i == 0:
                    continue
                vals = CW.recv(source=i, tag=i)
                returns_global[i::CW.size] = vals
        CW.Barrier()
        if CW.rank == 0:
            return returns_global
        else:
            return True

    def plot_onsets(self, ra_range):
        if CW.rank == 0:
            import matplotlib.pyplot as plt
        evals = dict()
        for ra in ra_range:
            returned = self.solve_unstable_modes_parallel(ra)
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
        if CW.rank == 0:
            import h5py
            f = h5py.File(self.out_dir+'evals_output_text_{0}-{1}.h5'.format(ra_range[0], ra_range[-1]), 'w')

            plt.figure(figsize=(15,10))
            for key in evals.keys():
                ras = []
                evals_local = []
                for pack in evals[key]:
                    ras.append(pack[0])
                    evals_local.append(pack[1])
                plt.plot(ras, evals_local, label='wavenum='+key)
                print('wavenum {0} has {1} at {2}'.format(int(key), ras, evals_local))
                dict_key_ra = '{0}_ras'.format(int(key))
                dict_key_eval = '{0}_evals'.format(int(key))
                f[dict_key_ra] = ras
                f[dict_key_eval] = evals_local
            plt.axhline(self.epsilon, linestyle='dashed', color='black', label=r'$\epsilon$')
            plt.legend(loc='upper left')
            plt.xlabel('Ra')
            plt.ylabel(r'$Re(\omega)$')
            plt.yscale('log')
            plt.savefig(self.out_dir+'evals_onset_fig_{0}-{1}.png'.format(ra_range[0], ra_range[-1]), dpi=200)

            
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


        

    
            

if __name__ == '__main__':
    eqs = 7
    nx = 32
    nz = 32
    Lx = 100


    start_ra = 60
    stop_ra = 70
    steps = 101
    solver = FC_onset_solver(nx=nx, nz=nz, Lx=Lx, comm=MPI.COMM_SELF)
    solver.plot_onsets(np.linspace(start_ra, stop_ra, steps))
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
