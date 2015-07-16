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


def semilogy_posneg(ax, x, y, **kwargs):
    pos_mask = np.logical_not(y>0)
    neg_mask = np.logical_not(y<0)
    pos_line = np.ma.MaskedArray(y, pos_mask)
    neg_line = np.ma.MaskedArray(y, neg_mask)

    color = next(ax._get_lines.color_cycle)
    ax.semilogy(x, pos_line, color=color, **kwargs)
    ax.semilogy(x, np.abs(neg_line), color=color, linestyle='dashed')


def read_data(files, verbose=False, data_keys=None):
    data_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_s')[1]))

    if data_keys is None:
        f = h5py.File(data_files[0], flag='r')
        data_keys = np.copy(f['tasks'])
        f.close()
        logger.debug("tasks = {}".format(data_keys))

    if verbose:
        f = h5py.File(data_files[0], flag='r')
        logger.info(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            logger.info(task)
        
        logger.info(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            logger.info(key)
        f.close()

    data_set = OrderedDict()
    for key in data_keys:
        data_set[key] = np.array([])

    times = np.array([])

    N = 1
    for filename in data_files:
        f = h5py.File(filename, flag='r')
        # clumsy
        for key in data_keys:
            if N == 1:
                data_set[key] = f['tasks'][key][:]
                logger.debug("{} shape {}".format(key, data_set[key].shape))
            else:
                data_set[key] = np.append(data_set[key], f['tasks'][key][:], axis=0)

        N += 1
        # same z for all files
        z = f['scales']['z']['1.0'][:]
        times = np.append(times, f['scales']['sim_time'][:])
        f.close()

    for key in data_keys:
        logger.debug("{} shape {}".format(key, data_set[key].shape))

    time_avg = OrderedDict()
    std_dev  = OrderedDict()
    for key in data_keys:
        time_avg[key] = np.mean(data_set[key], axis=0)[0]
        std_dev[key]  = np.std(data_set[key], axis=0)[0]

    for key in data_keys:
        logger.debug("{} shape {} and {}".format(key, time_avg[key].shape, std_dev[key].shape))

    return time_avg, std_dev, z, times

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

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.semilogy(z, averages['enstrophy']/np.max(averages['enstrophy']), label="enstrophy")
    ax1.semilogy(z, averages['KE']/np.max(averages['KE']),  label="KE")
    semilogy_posneg(ax1, z, averages['KE_flux_z']/np.max(averages['KE_flux_z']),  label="KE_flux")
    ax1.axhline(y=1e-2, color='black', linestyle='dashed')
    ax1.axhline(y=1e-4, color='black', linestyle='dashed')
    ax1.legend(loc="upper left")
    ax1.set_xlabel("z")
    ax1.set_ylabel("penetration diags, normalized by max")
    figs["pen"]=fig

    for key in figs.keys():
        figs[key].savefig(output_path+'flows_{}.png'.format(key))


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


