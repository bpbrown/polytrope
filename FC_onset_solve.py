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
        dtype=np.complex128):

        self.atmosphere = equations.FC_polytrope(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                epsilon=epsilon, gamma=gamma,
                constant_diffusivities=constant_diffusivities,
                constant_kappa=constant_kappa,
                dtype=dtype)

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

plot_rows = 2
plot_cols = 5


plt.figure(figsize=(30,20))
solver = FC_onset_solver(nz=32, Lx=100)
for i in range(solver.atmosphere.nx-2):

    ra = 1e2
    plt.clf()
    wavenum = i+1
    real_wavenum = 2*np.pi*wavenum/solver.atmosphere.Lx/solver.atmosphere.nx

    print('solving wavenum {0} at ra = {1}'.format(real_wavenum, ra))
    solver.solve_parallel([wavenum, ra])

    for e in range(len(solver.solver.eigenvalues)):#range(solver.atmosphere.nz*2, solver.atmosphere.nz*3):
        if np.abs(np.imag(solver.solver.eigenvalues[e])) == np.inf:
            continue
        plt.clf()
        ws = solver.get_profiles(e)['w']
        break
        string = 'wavenum {0:.4g}; eigval {1:.4g}'.format(real_wavenum, solver.solver.eigenvalues[e])
        if np.imag(solver.solver.eigenvalues[e]) > 0:
            string += '; UNSTABLE'
        else:
            continue
            string += '; STABLE'
        print(string)
        plt.suptitle(string, fontsize=16) 
        for j in range(plot_rows):
            for k in range(plot_cols):
                plt.subplot(plot_rows, plot_cols, j*plot_cols + k + 1)
                plt.plot(np.real(ws['g'][j*plot_cols + k,:]))
                plt.plot(np.imag(ws['g'][j*plot_cols + k,:]))
        plt.subplots_adjust(top=0.88, wspace=0.5, hspace=0.4)
        plt.savefig('./figures/unstable_first10_evalindex{:03d}_wavenum{:06d}_ra{:.1e}.png'.format(e,wavenum, ra), dpi=100)
