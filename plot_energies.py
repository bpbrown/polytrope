"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py join <base_path>
    plot_energies.py <base_path> [--output=<output> --unjoined]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made
    --unjoined         Use unjoined scalar files

"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from tools import analysis

def plot_energies(data, times, output_path='./'):
    t = times
        
    figs = {}
    two_size = (16,8)
    one_size = (8,8)
    
    fig_energies = plt.figure(figsize=two_size)
    ax1 = fig_energies.add_subplot(2,1,1)
    ax1.semilogy(t, data['KE'], label="KE")
    ax1.semilogy(t, data['PE'], label="PE")
    ax1.semilogy(t, data['IE'], label="IE")
    ax1.semilogy(t, data['TE'], label="TE")
    if 'ME' in data:
        ax1.semilogy(t, data['ME'], label="ME")
    ax1.legend()
    
    ax2 = fig_energies.add_subplot(2,1,2)
    ax2.plot(t, data['KE'], label="KE")
    ax2.plot(t, data['PE'], label="PE")
    ax2.plot(t, data['IE'], label="IE")
    ax2.plot(t, data['TE'], label="TE")
    if 'ME' in data:
        ax2.plot(t, data['ME'], label="ME")
    ax2.legend()
    figs["energies"]=fig_energies

    fig_KE = plt.figure(figsize=two_size)
    ax1 = fig_KE.add_subplot(1,1,1)
    ax1.plot(t, data['KE'], label="KE")
    ax1.plot(t, data['PE']-data['PE'][0], label="PE-PE$_0$")
    ax1.plot(t, data['IE']-data['IE'][0], label="IE-IE$_0$")
    #ax1.plot(t, data['IE_fluc'], label="IE$_1$", linestyle='dashed')
    #ax1.plot(t, data['PE_fluc'], label="PE$_1$", linestyle='dashed')
    ax1.plot(t, data['TE']-data['TE'][0], label="TE-TE$_0$", color='black')
    if 'ME' in data:
         ax1.plot(t, data['ME'], label="ME")
    ax1.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    figs["fluctuating_energies"] = fig_KE

    fig_KE_only = plt.figure(figsize=two_size)
    ax1 = fig_KE_only.add_subplot(2,1,1)
    ax1.plot(t, data['KE'], label="KE")
    ax1.legend()
    ax1.set_ylabel("energy")
    ax2 = fig_KE_only.add_subplot(2,1,2)
    ax2.semilogy(t, data['KE'], label="KE")
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")
    figs["KE"] = fig_KE_only
 
    fig_Nu = plt.figure(figsize=two_size)
    ax1 = fig_Nu.add_subplot(2,1,1)
    ax1.plot(t, data['Nusselt_G75'], label="Nu_G75")
    ax1.plot(t, data['Nusselt_AB17'], label="Nu_AB17")
    ax1.legend()
    ax1.set_ylabel("Nu")
    ax2 = fig_Nu.add_subplot(2,1,2)
    ax2.semilogy(t, np.abs(data['Nusselt_G75']), label="Nu_G75")
    ax2.semilogy(t, np.abs(data['Nusselt_AB17']), label="Nu_AB17")
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("Nu")
    figs["Nu"] = fig_Nu

    fig_Nu2 = plt.figure(figsize=two_size)
    ax1 = fig_Nu2.add_subplot(2,1,1)
    ax1.plot(t, data['Nusselt_AB17'], label="Nu_AB17")
    #ax1.legend()
    ax1.set_ylabel("Nu")
    ax2 = fig_Nu2.add_subplot(2,1,2)
    ax2.semilogy(t, np.abs(data['Nusselt_AB17']), label="Nu_AB17")
    #ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("Nu")
    figs["Nu_AB17"] = fig_Nu2

    fig_nrho = plt.figure(figsize=one_size)
    ax1 = fig_nrho.add_subplot(1,1,1)
    ax1.plot(t, data['n_rho'], label=r'$n_\rho$')
    #ax1.legend()
    ax1.set_ylabel(r'$n_\rho$')
    ax1.set_xlabel("time")
    figs["n_rho"] = fig_nrho

    fig_Re = plt.figure(figsize=one_size)
    ax1 = fig_Re.add_subplot(1,1,1)
    ax1.plot(t, data['Re_rms'], label=r'Re$_\mathrm{rms}$')
    #ax1.legend()
    ax1.set_ylabel(r'Re$_\mathrm{rms}$')
    ax1.set_xlabel("time")
    figs["Re_rms"] = fig_Re

    fig_equil = plt.figure(figsize=one_size)
    ax1 = fig_equil.add_subplot(111)
    ax1.plot(t, data['flux_equilibration'], label=r'instantanous')
    ax1.plot(t, analysis.cumulative_avg(data['flux_equilibration']), label=r'cumulative average')
    ax1.axhline(0,alpha=0.4,color='k')
    ax1.legend()
    ax1.set_ylabel(r'$F_{bottom} - F_{top}$')
    ax1.set_xlabel("time")
    figs["flux_equilibration"] = fig_equil
        
    for key in figs.keys():
        figs[key].savefig(output_path+'scalar_{}.png'.format(key))
    
def main(files, output_path='./'):
    data = analysis.Scalar(files)
    plot_energies(data.data, data.times, output_path=output_path)
    

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import glob
    import os

    args = docopt(__doc__)

    if args['join']:
        post.merge_process_files(args['<base_path>'])
    else:
        if args['--output'] is not None:
            output_path = pathlib.Path(args['--output']).absolute()
        else:
            data_dir = args['<base_path>'].split('/')[0]
            data_dir += '/'
            output_path = pathlib.Path(data_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        logger.info("output to {}".format(output_path))
        base_path = args['<base_path>']
        if args['--unjoined']:
            logger.info("Using unjoined output...")
            file_glob = os.path.join(base_path,"scalar_s*/*p0.h5")
        else:
            file_glob = os.path.join(base_path,"scalar_s*.h5")
        files = glob.glob(file_glob)
        main(files, output_path=str(output_path)+'/')


