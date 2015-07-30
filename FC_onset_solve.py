#!/usr/bin/env python
import numpy as np
from dedalus.public import *
import h5py

import os
import sys

import equations

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from mpi4py import MPI
CW = MPI.COMM_WORLD
import matplotlib
matplotlib.use('Agg')

class FC_onset_solver:
    '''
    This class creates a polytropic atmosphere as defined by equations.py,
    then uses EVP solves on that atmosphere to determine the critical Ra of
    convection by searching for the first value of Ra at which an unstable
    (e^[positive omega]) mode arises.
    '''

    def __init__(self, ra_range, profiles=['u','w','T1'], nx=64, nz=64, aspect_ratio=10, epsilon=1e-4, gamma=5/3,
        n_rho_cz=3.5, constant_diffusivities=True, constant_kappa=False,
        grid_dtype=np.complex128, comm=MPI.COMM_SELF,
	    out_dir=''):
        '''
        Initializes the atmosphere, sets up basic variables.

        NOTE: To run in parallel, set comm=MPI.COMM_SELF.
        '''
        self.nx = nx
        self.nz = nz
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_rho_cz = n_rho_cz
        self.constant_diffusivities=constant_diffusivities
        self.constant_Kappa=constant_kappa

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, aspect_ratio=10,
                gamma=gamma, grid_dtype=grid_dtype, n_rho_cz=n_rho_cz,
                constant_diffusivities=constant_diffusivities,
                constant_kappa=constant_kappa,
                comm=comm, epsilon=epsilon)
        self.Lx = self.atmosphere.Lx
        self.Lz = self.atmosphere.Lz

        self.n_rho_true = np.log(np.max(self.atmosphere.rho0['g'])/np.min(self.atmosphere.rho0['g']))

        logger.info('# of density scale heights: {}'.format(self.n_rho_true))
        self.x = self.atmosphere.x
        self.z = self.atmosphere.z

        self.ra_range = ra_range
        self.profiles = profiles

        self.out_file_name = 'evals_eps_{0:.0e}_ras_{1:04g}-{2:04g}_nrho_{3:.1f}_{4:04g}x{5:04g}'.format(self.epsilon, self.ra_range[0], self.ra_range[-1], self.n_rho_cz, self.nx, self.nz)

        out_dir_base = sys.argv[0].split('/')[-1].split('.py')[0] + '/'
        self.out_dir = out_dir + out_dir_base
        if not os.path.exists(self.out_dir) and CW.rank == 0:
            os.makedirs(self.out_dir)
        if CW.size > 1:
            self.local_file_dir = self.out_dir + self.out_file_name + '/'
            if not os.path.exists(self.local_file_dir) and CW.rank == 0:
                os.makedirs(self.local_file_dir)
            CW.Barrier()
            self.local_file_name = self.local_file_dir + 'proc_{}.h5'.format(CW.rank)
            self.local_file = h5py.File(self.local_file_name, 'w')
        else:
            self.local_file_name = self.out_dir + self.out_file_name + '.h5'
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
            key = '{0:09.2f}_{1:04d}'.format(ra, wave)
            key_prof = key + '_prof'
            key_eigenvalues = key + '_eig'
            if eigenvalues.shape[1] > 0:
                self.local_file[key_prof] = profiles
                self.local_file[key_eigenvalues] = eigenvalues
                my_keys.append(key_prof)
                my_keys.append(key_eigenvalues)
        logger.info('Setting up file keys...')
        if my_keys == []:
            my_keys.append('none')
        asciiList = [n.encode('ascii', 'ignore') for n in my_keys]
        self.local_file.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
        self.local_file.close()
        CW.Barrier()
        if stitch_files and CW.rank == 0:
            self.stitch_files()

    def stitch_files(self):
        logger.info('Merging files...')
        filename = self.out_dir + self.out_file_name + '.h5'
        file_all = h5py.File(filename, 'w')
        keys = []
        for i in range(CW.size):
            partial_file = self.local_file_dir + 'proc_{}.h5'.format(i)
            file_part = h5py.File(partial_file, 'r')
            keys_part = []
            for raw_key in file_part['keys'][:]:
                key = str(raw_key[0])[2:-1]
                keys_part.append(key)
            for key in keys_part:
                if key == 'none':
                    break
                keys.append(key)
                file_all[key] = np.asarray(file_part[key][:])
        if keys == []:
            keys.append('none')
        asciiList = [n.encode('ascii', 'ignore') for n in np.sort(keys)]
        file_all.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
            
    def read_file(self, start_ra=None, stop_ra=None, eps=None, n_rho=None, nx=None, nz=None, process=0):
        filename = self.out_dir
        if start_ra == None and stop_ra == None and eps==None and n_rho==None:
            filename += self.out_file_name + '.h5'
        if start_ra == None:
            start_ra = self.ra_range[0]
        if stop_ra == None:
            stop_ra = self.ra_range[-1]
        if eps == None:
            eps = self.epsilon
        if n_rho == None:
            n_rho = self.n_rho_true
        if nx == None:
            nx = self.nx
        if nz == None:
            nz = self.nz
        if filename[-3:] != '.h5':
            filename += 'evals_eps_{0:.0e}_ras_{1:04g}-{2:04g}_nrho_{3:.1f}_{4:04g}x{5:04g}.h5'.format(eps, start_ra, stop_ra, n_rho, nx, nz)

        logger.info('reading file {}'.format(filename))

        if CW.rank == process:
            f = h5py.File(filename, 'r')
            keys = []
            for raw_key in f['keys'][:]:
                key = str(raw_key[0])[2:-1]
                if key == 'none':
                    return None, None, None, None
                split = key.split('_')
                keys.append([key, float(split[0]), int(split[1])])
            wavenumbers = []
            profiles = []
            ra_counter = 0
            for key_pack in keys:
                if wavenumbers == []:
                    wavenumbers.append([key_pack[1]])
                    profiles.append([key_pack[1]])
                elif wavenumbers[ra_counter][0] != key_pack[1]:
                    wavenumbers.append([key_pack[1]])
                    profiles.append([key_pack[1]])
                    ra_counter += 1
                if 'eig' in key_pack[0]:
                    wavenumbers[ra_counter].append((key_pack[2], np.asarray(f[key_pack[0]][1])))
                if 'prof' in key_pack[0]:
                    profiles[ra_counter].append((key_pack[2], f[key_pack[0]][:]))
            return wavenumbers, profiles, filename.split('/')[-1].split('.h5')[0], (eps, n_rho)
        return None, None, None, None

    def plot_growth_modes(self, wavenumbers, filename, process=0):
        logger.info('plotting growth modes on process {}'.format(process))
        if CW.rank == process:
            import matplotlib.pyplot as plt
            wavenums = []
            wavenums_ind = []
            plot_pairs = []
            plt.figure(figsize=(15,10))
            for ra_pack in wavenumbers:
                my_ra = ra_pack[0]
                for j in range(len(ra_pack)-1):
                    if ra_pack[j+1][0] not in wavenums:
                        wavenums.append(ra_pack[j+1][0])
                        wavenums_ind.append(len(wavenums)-1)
                    if len(plot_pairs) != len(wavenums):
                        plot_pairs.append([[my_ra],[ra_pack[j+1][1][0]]])
                    else:
                        index = wavenums_ind[wavenums.index(ra_pack[j+1][0])]
                        plot_pairs[index][0].append(my_ra)
                        plot_pairs[index][1].append(ra_pack[j+1][1][0])
            for i in range(len(plot_pairs)):
                plt.plot(plot_pairs[i][0], plot_pairs[i][1], label='wavenum {0}'.format(wavenums[i]))
            plt.legend()
            plt.xlabel('Ra')
            plt.ylabel('growth node strength')
            plt.yscale('log')
            figname = self.out_dir + filename + '_growth_node_plot.png'
            plt.savefig(figname, dpi=100)

    def plot_onset_curve(self, wavenumbers, filename, atmosphere, process=0, clear=True, save=True, linestyle='-', figname='default', dpi=150):
        logger.info('plotting onset curve on process {}'.format(process))
        if CW.rank == process:
            import matplotlib.pyplot as plt
            wavenums = np.arange(self.nx/2)
            onsets = np.zeros(wavenums.shape[0])
            if clear:
                plt.figure(figsize=(15,10))
                plt.clf()
            eps = atmosphere[0]
            n_rho = atmosphere[1]
            if len(atmosphere) > 2:
                Lx = atmosphere[2]
            else:
                Lx = None
            for ra_pack in wavenumbers:
                my_ra = ra_pack[0]
                for j in range(len(ra_pack)-1):
                    wavenum = ra_pack[j+1][0]
                    if onsets[wavenum] == 0:
                        onsets[wavenum] = my_ra
            wavenums = wavenums[np.where(onsets > 0)]
            if Lx != None:
                wavenums = 2*np.pi*wavenums/Lx
            print(onsets)
            onsets = onsets[np.where(onsets > 0)]
            plt.plot(wavenums, onsets, label='n_rho: {} & eps: {:.1e}'.format(n_rho, eps), linestyle=linestyle)
            plt.legend(loc='lower right')
            if Lx != None:
                plt.xlabel('wavenum')
            else:
                plt.xlabel('wavenum index')
            plt.ylabel('Ra_top crit')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(0, self.nx/2)
            if figname == 'default':
                figname = self.out_dir + filename + '_onset_curve.png'
            else:
                figname = self.out_dir + figname
            if save:
                plt.savefig(figname, dpi=dpi)
                
if __name__ == '__main__':
    eqs = 7
    nx = 64
    nz = 128
    aspect_ratio=10
    epsilon=1e-4
    out_dir = '/regulus/exoweather/evan/'
    n_rho_cz = [3.5, 5, 7, 12]

    '''
    for n_rho in n_rho_cz:
        solver = FC_onset_solver(np.logspace(1, 4, 200), profiles=['u','w','T1'],nx=nx, nz=nz, aspect_ratio=aspect_ratio, n_rho_cz=n_rho, epsilon=epsilon, comm=MPI.COMM_SELF, out_dir=out_dir, constant_kappa=True)
        solver.full_parallel_solve()
        wavenumbers, profiles, filename, atmosphere = solver.read_file()
        if wavenumbers != None:
            solver.plot_growth_modes(wavenumbers, filename)
    '''
    for i in range(len(n_rho_cz)):
        solver = FC_onset_solver(np.logspace(1, 4, 200), profiles=['u','w','T1'],nx=nx, nz=nz, aspect_ratio=aspect_ratio, n_rho_cz=n_rho_cz[i], epsilon=epsilon, comm=MPI.COMM_SELF, out_dir=out_dir, constant_kappa=True)
        solver.full_parallel_solve()
        wavenumbers, profiles, filename, atmosphere = solver.read_file(n_rho=n_rho_cz[i])
        atmosphere = (atmosphere[0], atmosphere[1], solver.Lx)
        if wavenumbers == None:
            continue
        if i == 0 and len(n_rho_cz) > 1:
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=True, save=False)
        elif i == len(n_rho_cz)-1:
            if len(n_rho_cz) == 1:
                clear = True
            else:
                clear = False
            figname = 'onset_{:.04g}x{:.04g}_nrhos_{:.04g}-{:.04g}.png'.format(nx, nz, n_rho_cz[0], n_rho_cz[-1])
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=clear, save=True, figname=figname)
        else:
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=False, save=False)
