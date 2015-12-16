"""
Plot energy fluxes from joint analysis files.

Usage:
    plot_fluxes.py join <base_path>
    plot_fluxes.py <files>... [--output=<output> --overshoot]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made
    --overshoot        Do overshoot diagnostics
"""
import numpy as np

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import analysis

def plot_flows(averages, z, output_path='./'):
    figs = {}

    fig_flow = plt.figure(figsize=(16,8))
    ax1 = fig_flow.add_subplot(1,1,1)
    ax1.semilogy(z, averages['Re_rms'], label="Re", color='darkblue')
    ax1.semilogy(z, averages['Pe_rms'], label="Pe", color='darkred')
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("Re and Pe")
    figs["Re_Pe"]=fig_flow

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,1,1)
    try:
        ax1.semilogy(z, averages['Rayleigh_global'], label="global Ra")
    except:
        pass
    try:
        ax1.semilogy(z, np.abs(averages['Rayleigh_local']),  label=" local Ra")
    except:
        pass
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("Ra")
    figs["Ra"]=fig

    for key in figs.keys():
        figs[key].savefig(output_path+'flows_{}.png'.format(key))

def plot_fluxes(fluxes, z, output_path='./'):

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

def plot_profiles(data, z, output_path='./'):
    figs = {}

    keys = ['IE_fluc', 'PE_fluc',
            's_mean', 's_fluc', 's_tot', 's_fluc_std',
            'T1', 'ln_rho1',
            'kappa_flux_fluc_z', 'kappa_flux_mean_z', 'kappa_flux_z',
            'T1_source_terms']

    if "ME" in data:
        keys.append("ME")
        
    for key in keys:
        try:
            fig_flow = plt.figure(figsize=(16,8))
            ax1 = fig_flow.add_subplot(1,1,1)
            logger.info(data[key].shape)
            for i in range(data[key].shape[0]):
                ax1.plot(z, data[key][i,0,:])

            ax1.set_xlabel("z")
            ax1.set_ylabel(key)
            figs[key]=fig_flow
        except:
            logger.info("Missing key {}".format(key))
            
    for key in figs.keys():
        figs[key].savefig(output_path+'profiles_{}.png'.format(key), dpi=600)

def main(files, output_path='./', overshoot=True):
    logger.info("opening {}".format(files))
    data = analysis.Profile(files)
    averages = data.average
    std_devs = data.std_dev
    times = data.times
    z = data.z
    delta_t = times[-1]-times[0]
    logger.info("Averaged over interval t = {:g} -- {:g} for total delta_t = {:g}".format(times[0], times[-1], delta_t))
    plot_fluxes(averages, z, output_path=output_path)
    plot_flows(averages, z, output_path=output_path)
    if overshoot:
        import plot_overshoot
        plot_overshoot.analyze_case(files, verbose=True)
    plot_profiles(data.data, z, output_path=output_path)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    if args['join']:
        post.merge_analysis(args['<base_path>'])
    else:
        if args['--output'] is not None:
            output_path = pathlib.Path(args['--output']).absolute()
        else:
            data_dir = args['<files>'][0].split('/')[0]
            data_dir += '/'
            output_path = pathlib.Path(data_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        logger.info("output to {}".format(output_path))
        main(args['<files>'], output_path=str(output_path)+'/', overshoot=args['--overshoot'])


