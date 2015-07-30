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

    def __init__(self, ra_range, profiles=['u','w','T1'], nx=64, nz=64, 
                aspect_ratio=10, epsilon=1e-4, gamma=5/3,
                n_rho_cz=3.5, constant_diffusivities=True, constant_kappa=False,
                grid_dtype=np.complex128, comm=MPI.COMM_SELF, out_dir=''):
        '''
        Initializes the atmosphere, sets up basic variables and output files
        NOTE: To run in parallel, set comm=MPI.COMM_SELF.

        ----
        PARAMETERS
        ----
        ra_range - a 1d numpy array of Ra values over which to solve EVPs
        profiles - The dedalus profiles to save to file for all unstable modes
        nx, nz   - The number of dedalus nodes to run in in the x and z, respectively
        aspect_ratio - the value of Lx/Lz for the atmosphere
        epsilon  - The level of superadiabaticity of the atmosphere (0 = adiabatic)
        gamma    - The adiabatic index of the atmosphere
        n_rho_cz - The number of density scale heights which the atmosphere spans
        constant_diffusivites - If true, diffusivities will be constant in the atmosphere
        constant_kappa - If true, kappa will be constant in the atmosphere
        grid_dtype - The data type of the dedalus grid
        comm - The MPI communicator.  If not set to MPI.COMM_SELF, the problem will
                    not run in parallel.
        out_dir  - The base output directory of all data.
        '''
        self.nx, self.nz, self.epsilon, self.gamma = nx, nz, epsilon, gamma
        self.n_rho_cz, self.constant_diffusivities, self.constant_kappa = \
                                n_rho_cz, constant_diffusivities, constant_kappa

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, aspect_ratio=10,
                        gamma=gamma, grid_dtype=grid_dtype, n_rho_cz=n_rho_cz,
                        constant_diffusivities=self.constant_diffusivities,
                        constant_kappa=self.constant_kappa,
                        comm=comm, epsilon=epsilon)
        self.x, self.z = self.atmosphere.x, self.atmosphere.z
        self.Lx, self.Lz = self.atmosphere.Lx, self.atmosphere.Lz
        self.n_rho_true = np.log(np.max(self.atmosphere.rho0['g'])/np.min(self.atmosphere.rho0['g']))
        logger.info('# of density scale heights: {}'.format(self.n_rho_true))

        self.ra_range = ra_range
        self.profiles = profiles

        self.out_file_name = 'evals_eps_{0:.0e}_ras_{1:04g}-{2:04g}_nrho_{3:.1f}_{4:04g}x{5:04g}'.\
                                format(self.epsilon, self.ra_range[0], self.ra_range[-1], self.n_rho_cz, self.nx, self.nz)

        #Set up output files and directory
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
        Sets up the EVP solver for a given value of ra and Pr.
        '''
        self.atmosphere.set_eigenvalue_problem(ra, pr,\
                    include_background_flux=False)
        self.atmosphere.set_BC()
        self.problem = self.atmosphere.get_problem()
        self.solver = self.problem.build_solver()
        return self.solver

    def solve(self, wave, ra, pr=1):
        '''
        Solves the system at the wavenumber INDEX specified and at
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
        '''
        Solves EVPs in parallel and keeps track of all unstable modes, saving them
        out to a local file.
        '''
        #Grabs wavenumbers and ras.  Splits up ras and wavenumbers across all processors.
        kxs = int(self.nx/2)
        kxs_global = list(np.arange(kxs))
        tasks = kxs_global * len(self.ra_range)
        for i in range(len(self.ra_range)):
            for j in range(kxs):
                tasks[i*kxs + j] = (tasks[i*kxs + j], self.ra_range[i])
        my_tasks = tasks[CW.rank::CW.size]

        #For each 'task' (ra & wavenumber), solve the EVP and save unstable modes.
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

        #Save all file keys for later reading of the file
        logger.info('Setting up file keys...')
        if my_keys == []:
            my_keys.append('none')
        asciiList = [n.encode('ascii', 'ignore') for n in my_keys]
        self.local_file.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
        self.local_file.close()
        CW.Barrier()
        if stitch_files and CW.rank == 0:
            self.stitch_files()

    def stitch_files(self, num_files=CW.size):
        '''
        Merges the output files from all processors involved in an EVP run.
        '''
        logger.info('Merging files...')
        filename = self.out_dir + self.out_file_name + '.h5'
        file_all = h5py.File(filename, 'w')
        keys = []

        #Pull all the data out of each file
        for i in range(num_files):
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
        #Store all keys properly for later reading
        if keys == []:
            keys.append('none')
        asciiList = [n.encode('ascii', 'ignore') for n in np.sort(keys)]
        file_all.create_dataset('keys', (len(asciiList),1), 'S20', asciiList)
            
    def read_file(self, start_ra=None, stop_ra=None, eps=None, n_rho=None, nx=None, nz=None, process=0):
        '''
        Read a specified file using kwargs OR read the natural file for the given solver.

        IF all kwargs=None (except for process), then this reads the file with the name 'self.out_dir'+'.h5'
        If kwargs are specified, this will try to read the file which is SIMILAR to self.out_dir, but which
        is modified by those kwargs.

        RETURNS: (on CW.rank == process):
            wavenumbers:    a list of Ra, where each Ra is paired with each wavenumber that has an unstable mode
                            and the values of those unstable modes
            profiles:       a list of Ra, where each Ra is paried with each wavenumber and the profiles of the
                            unstable modes at that wavenumber
            filename:       The name of the read file, minus the '.h5'
            atmosphere:     A tuple containing (epsilon, n_rho)

                (on CW.rank != process):
            returns None, None, None, None

        '''
        filename = self.out_dir
        if start_ra == None and stop_ra == None and eps==None and n_rho==None and nx == None and nz == None:
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

        #If we're on the specified process, open the file and read it.
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
        '''
        Plots the value of unstable growth nodes vs. Ra
        '''
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

    def plot_onset_curve(self, wavenumbers, filename, atmosphere, 
                         process=0, clear=True, save=True, 
                         linestyle='-', figname='default', dpi=150):
        '''
        Plots a critical Ra curve vs. wavenumber.  Has options to save/clear
            the figure (or not) depending on whether or not you want to add
            curves from other atmospheres.
        '''
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
                plt.xlim(0, self.nx/2)
            plt.ylabel('Ra_top crit')
            plt.yscale('log')
            plt.xscale('log')
            if figname == 'default':
                figname = self.out_dir + filename + '_onset_curve.png'
            else:
                figname = self.out_dir + figname
            if save:
                plt.savefig(figname, dpi=dpi)

