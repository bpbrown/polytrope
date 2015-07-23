"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py join <base_path>
    plot_energies.py plot <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./scalar]

"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(files, output_path='./'):
    [KE, PE, IE, TE], t = read_data(files)
    plot_energies([KE, PE, IE, TE], t, output_path=output_path)
    
def read_data(files, verbose=False):
    data_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_s')[-1]))
    if verbose:
        f = h5py.File(data_files[0], flag='r')
        print(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            print(task)
        print(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            print(key)

    KE = np.array([])
    PE = np.array([])
    IE = np.array([])
    TE = np.array([])
    t = np.array([])

    for filename in data_files:
        f = h5py.File(filename, flag='r')
        KE = np.append(KE, f['tasks']['KE'][:])
        PE = np.append(PE, f['tasks']['PE'][:])
        IE = np.append(IE, f['tasks']['IE'][:])
        TE = np.append(TE, f['tasks']['TE'][:])

        t = np.append(t,f['scales']['sim_time'][:])
        f.close()

    return [KE, PE, IE, TE], t

def plot_energies(energies, t, output_path='./'):
    [KE, PE, IE, TE] = energies

    figs = {}
    
    fig_energies = plt.figure(figsize=(16,8))
    ax1 = fig_energies.add_subplot(2,1,1)
    ax1.semilogy(t, KE, label="KE")
    ax1.semilogy(t, PE, label="PE")
    ax1.semilogy(t, IE, label="IE")
    ax1.semilogy(t, TE, label="TE")
    ax1.legend()
    
    ax2 = fig_energies.add_subplot(2,1,2)
    ax2.plot(t, KE, label="KE")
    ax2.plot(t, PE, label="PE")
    ax2.plot(t, IE, label="IE")
    ax2.plot(t, TE, label="TE")
    ax2.legend()
    figs["energies"]=fig_energies

    fig_relative = plt.figure(figsize=(16,8))
    ax1 = fig_relative.add_subplot(1,1,1)
    ax1.plot(t, TE/TE[0]-1)
    ax1.plot(t, IE/IE[0]-1)
    ax1.plot(t, PE/PE[0]-1)
    figs["relative_energies"] = fig_relative

    fig_KE = plt.figure(figsize=(16,8))
    ax1 = fig_KE.add_subplot(1,1,1)
    ax1.plot(t, KE, label="KE")
    ax1.plot(t, PE-PE[0], label="PE-PE$_0$")
    ax1.plot(t, IE-IE[0], label="IE-IE$_0$")
    ax1.plot(t, TE-TE[0], label="TE-TE$_0$", color='black')
    ax1.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    figs["fluctating_energies"] = fig_KE

    fig_KE_only = plt.figure(figsize=(16,8))
    ax1 = fig_KE_only.add_subplot(2,1,1)
    ax1.plot(t, KE, label="KE")
    ax1.legend()
    ax1.set_ylabel("energy")
    ax2 = fig_KE_only.add_subplot(2,1,2)
    ax2.semilogy(t, KE, label="KE")
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")
    figs["KE"] = fig_KE_only
 
    fig_log = plt.figure(figsize=(16,8))
    ax1 = fig_log.add_subplot(1,1,1)
    ax1.semilogy(t, KE, label="KE")
    ax1.semilogy(t, np.abs(PE-PE[0]), label="|PE-PE$_0$|")
    ax1.semilogy(t, np.abs(IE-IE[0]), label="|IE-IE$_0$|")
    ax1.semilogy(t, np.abs(TE-TE[0]), label="|TE-TE$_0$|", color='black')
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    ax1.legend()
    figs["log_fluctuating_energies"] = fig_log

    output_path = str(output_path)
    if output_path[-1] != '/':
        output_path += '/'

    for key in figs.keys():
        figs[key].savefig(output_path+'scalar_{}.png'.format(key))
    


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
        print(output_path)
        main(args['<files>'], output_path=output_path)


