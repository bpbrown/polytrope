#!/usr/bin/env python
import numpy as np
from dedalus.public import *
import h5py

import os
import sys

import equations


class FC_onset_solver:
    ''' I'll describe it later '''

    def __init__(self, nx=64, nz=64, Lx=30, Lz=10, epsilon=1e-4, gamma=5/3,
        constant_diffusivities=True, constant_kappa=True,
        grid_dtype=np.complex128):

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                gamma=gamma, grid_dtype=grid_dtype)

        self.crit_ras = []

    def build_solver(self, ra, pr=1):
        self.atmosphere.set_eigenvalue_problem(ra, pr,\
                    include_background_flux=False)
        self.atmosphere.set_BC()
        self.problem = self.atmosphere.get_problem()
        self.solver = self.problem.build_solver()
        return self.solver

    def solve_parallel(self, params, pr=1):
        h_wavenumber_index = int(params[0])
        ra = params[1]
        self.build_solver(ra, pr=pr)
        self.solver.solve(self.solver.pencils[h_wavenumber_index])
        self._eigenvalues_r = np.real(self.solver.eigenvalues)
        self._eigenvalues_i = np.imag(self.solver.eigenvalues)

    def get_positive_eigenvalues(self):
        indices_i = np.where(self._eigenvalues_i > 0)
        indices_r = np.where(self._eigenvalues_r > 0)
        self._positive_eigenvalues_i = np.zeros([2, len(indices_i)])
        self._positive_eigenvalues_r = np.zeros([2, len(indices_r)]) 
        
        self._positive_eigenvalues_i[0,:] = indices_i
        self._positive_eigenvalues_i[1,:] = self._eigenvalues_i[indices_i]
        self._positive_eigenvalues_r[0,:] = indices_r
        self._positive_eigenvalues_r[1,:] = self._eigenvalues_r[indices_r]

    def get_profiles(self, eigenvalue_index):
        self.solver.set_state(eigenvalue_index)
        return self.solver.state.field_dict



import matplotlib.pyplot as plt
import os

write_dir='./figures/'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)


plot_rows = 2
plot_cols = 3
eqs = 7
nx = 64
nz = 32
Lx = 100


solver = FC_onset_solver(nx=nx, nz=nz, Lx=Lx)
for e in range(nz*eqs):
    plt.figure(figsize=(30,20))
    wavenum = 2
    real_wavenum = 2*np.pi*wavenum/solver.atmosphere.Lx/solver.atmosphere.nx

    string = 'wavenum {0:.4g}'.format(real_wavenum)

    for i in range(3):#solver.atmosphere.nx-2):
        ra = 1e2*(10**(i+1))

        print('solving wavenum {0} at ra = {1}'.format(real_wavenum, ra))
        solver.solve_parallel([wavenum, ra])

        ws = solver.get_profiles(e)['w']

        string += '; eval{0}: {1:.3g}'.format(i+1, solver.solver.eigenvalues[e])
        
        print(string)
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.plot(solver.atmosphere.z[0], np.real(ws['g'][0,:]), color='blue')
        plt.plot(solver.atmosphere.z[0], np.imag(ws['g'][0,:]), color='green')
        plt.title('Ra = {0:.2e}'.format(ra))
        plt.xlabel('z')
        plt.ylabel('w[0,:] (Bu=Real, Gr=Imag)')
        plt.xlim(0, solver.atmosphere.z[0][-1])
        
        
        zs, xs = np.meshgrid(solver.atmosphere.z, solver.atmosphere.x)
        
        plt.subplot(plot_rows, plot_cols, plot_cols+i+1)
        im = plt.pcolormesh(xs, zs, ws['g'], cmap='PuOr_r')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('w')
        cbar = plt.colorbar(im)


    plt.suptitle(string, fontsize=20) 
    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4)
    plt.savefig(write_dir+'res{}x{}_wavenum{:03d}_evalindex{:03d}.png'.format(nx, nz, wavenum,e), dpi=100)
    plt.clf()
