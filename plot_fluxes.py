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


def main(files, output_path='./'):
    averages, std_devs, z = read_data(files)
    plot_fluxes(averages, z, output_path=output_path)
    
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
        f.close()

    for key in data_keys:
        logger.debug("{} shape {}".format(key, data_set[key].shape))

    time_avg = OrderedDict()
    std_dev = OrderedDict()
    for key in data_keys:
        time_avg[key] = np.mean(data_set[key], axis=0)[0]
        std_dev[key] = np.std(data_set[key], axis=0)[0]

    for key in data_keys:
        logger.debug("{} shape {} and {}".format(key, time_avg[key].shape, std_dev[key].shape))
    return time_avg, std_dev, z

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
        figs[key].savefig('./'+'energy_{}.png'.format(key))
    


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
        main(args['<files>'], output_path=output_path)


