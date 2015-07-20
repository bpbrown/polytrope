"""
Plot energy fluxes from joint analysis files.

Usage:
    plot_fluxes.py join <base_path>
    plot_fluxes.py plot <files>... [--output=<output>]

Options:
    --output=<output>  Output directory [default: ./fluxes]

"""
import numpy as np
import h5py
import os

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class profile():
    def __init__(self, *args, **kwargs):
        self.read_data(*args, **kwargs)
        self.average_data()
        
    def read_data(self, files, verbose=False, data_keys=None):
        data_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_s')[1]))

        if data_keys is None:
            f = h5py.File(data_files[0], flag='r')
            self.keys = np.copy(f['tasks'])
            f.close()
            logger.debug("tasks = {}".format(data_keys))
        else:
            self.keys = data_keys
            
        if verbose:
            f = h5py.File(data_files[0], flag='r')
            logger.info(10*'-'+' tasks '+10*'-')
            for task in f['tasks']:
                logger.info(task)
        
            logger.info(10*'-'+' scales '+10*'-')
            for key in f['scales']:
                logger.info(key)
            f.close()

        self.data = OrderedDict()
        for key in self.keys:
            self.data[key] = np.array([])

        self.times = np.array([])

        N = 1
        for filename in data_files:
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in data_keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            # same z for all files
            self.z = f['scales']['z']['1.0'][:]
            self.times = np.append(self.times, f['scales']['sim_time'][:])
            f.close()

        for key in self.keys:
            logger.debug("{} shape {}".format(key, self.data[key].shape))

    def average_data(self):
        self.average = OrderedDict()
        self.std_dev = OrderedDict()
        for key in self.keys:
            self.average[key] = np.mean(self.data[key], axis=0)[0]
            self.std_dev[key] = np.std( self.data[key], axis=0)[0]

        for key in self.keys:
            logger.debug("{} shape {} and {}".format(key, time_avg[key].shape, std_dev[key].shape))

def plot_flows(averages, z, output_path='./'):
    figs = {}

    fig_flow = plt.figure(figsize=(16,8))
    ax1 = fig_flow.add_subplot(1,1,1)
    ax1.plot(z, averages['Re_rms'], label="Re")
    ax1.plot(z, averages['Pe_rms'], label="Pe")
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("Re and Pe")
    figs["Re_Pe"]=fig_flow

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.semilogy(z, averages['Rayleigh_global'], label="global Ra")
    ax1.semilogy(z, averages['Rayleigh_local'],  label=" local Ra")
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("Ra")
    figs["Ra"]=fig

    for key in figs.keys():
        figs[key].savefig(output_path+'flows_{}.png'.format(key))


def diagnose_overshoot(averages, z, boundary=None, output_path='./'):
    import scipy.optimize as scpop
    figs = {}
    apjfig = APJSingleColumnFigure()
    norm_diag = OrderedDict()
    norm_diag['enstrophy'] = ('enstrophy', averages['enstrophy']/np.max(averages['enstrophy']))
    norm_diag['KE'] = ('KE', averages['KE']/np.max(averages['KE']))
    norm_diag['KE_flux'] = ('KE_flux', averages['KE_flux_z']/np.max(np.abs(averages['KE_flux_z'])))
    min_plot = 1
    max_plot = np.log(5) # half a log-space unit above 1
    for key in norm_diag:
        if key=='KE_flux':
            semilogy_posneg(apjfig.ax, z, norm_diag[key][1], label=norm_diag[key][0])
        else:
            apjfig.ax.semilogy(z, norm_diag[key][1], label=norm_diag[key][0])
            print(np.max(norm_diag[key][1]))
        min_plot = min(min_plot, np.min(np.abs(norm_diag[key][1])))
        
    apjfig.ax.axhline(y=1e-2, color='black', linestyle='dashed')
    apjfig.ax.axhline(y=1e-4, color='black', linestyle='dashed')
    if boundary is not None:
        apjfig.ax.axvline(x=boundary, color='black', linestyle='dotted')
    apjfig.legend(loc="lower right", title="normalized by max", fontsize=6)
    apjfig.ax.set_xlabel("z")
    apjfig.ax.set_ylabel("overshoot diags")#, \n normalized by max")
    apjfig.ax.set_ylim(min_plot, max_plot)
    padding = 0.1*np.max(z)
    xmin = np.min(z) - padding
    xmax = np.max(z) + padding
    apjfig.ax.set_xlim(xmin, xmax)
    figs["overshoot"]=apjfig

    # estimate penetration depths
    penetration_depths = OrderedDict()
    if boundary is None:
        # start in the middle
        start_search = 0.5*(np.max(z)+np.min(z))
    else:
        # otherwise, start at the interface between CZ and RZ
        start_search = boundary

    logger.info("searching for roots starting from z={}".format(start_search))
    Lz = np.max(z)-np.min(z) 
    x = 2/Lz*z-1 # convert back to [-1,1]
    x_start_search = 2/Lz*start_search-1

    for key in norm_diag:
        degree = 512

        cheb_coeffs = np.polynomial.chebyshev.chebfit(x, norm_diag[key][1], degree)
        cheb_interp = np.polynomial.chebyshev.Chebyshev(cheb_coeffs)
        if key=='KE_flux':
            #print(cheb_interp.roots())
            def newton_func(x_newton):
                return np.polynomial.chebyshev.chebval(x_newton, cheb_coeffs)

            try:
                x_root = scpop.newton(newton_func, x_start_search)
            except:
                logger.info("root find failed to converge")
                x_root = x_start_search
            z_root = (x_root+1)*Lz/2
            logger.info("{} : found root z={} (x:{} -> {})".format(key, z_root, x_start_search, x_root))  
            #print(np.polynomial.chebyshev.chebroots(cheb_coeffs))
            apjfig.ax.semilogy(z,np.polynomial.chebyshev.chebval(x, cheb_coeffs), label='fit')
    for key in figs.keys():
        figs[key].savefig(output_path+'diag_{}.png'.format(key))


def plot_fluxes(fluxes, z, output_path='./'):

    #flux_keys = ['enthalpy_flux_z', 'kappa_flux_z', 'kappa_flux_fluc_z', 'KE_flux_z']

    figs = {}

    fig_fluxes = plt.figure(figsize=(16,8))
    ax1 = fig_fluxes.add_subplot(1,1,1)
    ax1.plot(z, fluxes['enthalpy_flux_z'], label="h flux")
    ax1.plot(z, fluxes['kappa_flux_z'], label=r"$\kappa\nabla T$")
    ax1.plot(z, fluxes['KE_flux_z'], label="KE flux")
    ax1.plot(z, fluxes['enthalpy_flux_z']+fluxes['kappa_flux_z']+fluxes['KE_flux_z'], color='black', linestyle='dashed', label='total')
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("energy fluxes")
    figs["fluxes"]=fig_fluxes

    fig_fluxes = plt.figure(figsize=(16,8))
    ax1 = fig_fluxes.add_subplot(1,1,1)
    ax1.plot(z, fluxes['enthalpy_flux_z'], label="h flux")
    ax1.plot(z, fluxes['kappa_flux_fluc_z'], label=r"$\kappa\nabla T_1$")
    ax1.plot(z, fluxes['KE_flux_z'], label="KE flux")
    ax1.plot(z, fluxes['enthalpy_flux_z']+fluxes['kappa_flux_fluc_z']+fluxes['KE_flux_z'], color='black', linestyle='dashed', label='total')
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("energy fluxes")
    figs["relative_fluxes"]=fig_fluxes

    for key in figs.keys():
        figs[key].savefig(output_path+'energy_{}.png'.format(key))
    

def main(files, output_path='./'):
    averages, std_devs, z, times = read_data(files)
    delta_t = times[-1]-times[0]
    logger.info("Averaged over interval t = {:g} -- {:g} for total delta_t = {:g}".format(times[0], times[-1], delta_t))
    plot_fluxes(averages, z, output_path=output_path)
    plot_flows(averages, z, output_path=output_path)
    diagnose_overshoot(averages, z, output_path=output_path)
    


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    if args['join']:
        post.merge_analysis(args['<base_path>'])
    elif args['plot']:
        output_path = pathlib.Path(args['--output']).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        logger.info(output_path)
        main(args['<files>'], output_path=str(output_path)+'/')


