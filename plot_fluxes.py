"""
Plot energy fluxes from joint analysis files.

Usage:
    plot_fluxes.py join <base_path>
    plot_fluxes.py plot <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./fluxes]

"""
import numpy as np
import h5py
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(files, output_path='./'):
    fluxes, z = read_data(files)
    plot_fluxes(fluxes, z, output_path=output_path)
    
def read_data(files, verbose=False):
    data_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_s')[1]))
    if verbose:
        f = h5py.File(data_files[0], flag='r')
        print(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            print(task)
        print(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            print(key)

    enthalpy = np.array([])
    kappa = np.array([])
    KE = np.array([])
    z = np.array([])
    N = 1
    for filename in data_files:
        f = h5py.File(filename, flag='r')
        # clumsy
        if N == 1:
            enthalpy = f['tasks']['enthalpy_flux_z'][:]
            kappa = f['tasks']['kappa_flux_fluc_z'][:]
            KE = f['tasks']['KE_flux_z'][:]
            print("KE shape {}".format(KE.shape))
        else:
            
            enthalpy = np.append(enthalpy, f['tasks']['enthalpy_flux_z'][:], axis=0)
            kappa = np.append(kappa, f['tasks']['kappa_flux_fluc_z'][:], axis=0)
            KE = np.append(KE, f['tasks']['KE_flux_z'][:], axis=0)

        N += 1
        # same z for all files
        z = f['scales']['z']['1.0'][:]
        f.close()

    print("KE shape {}".format(KE.shape))
    enthalpy = np.mean(enthalpy, axis=0)[0]
    kappa = np.mean(kappa, axis=0)[0]
    KE = np.mean(KE, axis=0)[0]
    print("KE shape {}".format(KE.shape))
    return [enthalpy, kappa, KE], z

def plot_fluxes(fluxes, z, output_path='./'):
    [enthalpy, kappa, KE] = fluxes

    figs = {}
    for flux in fluxes:
        print(flux.shape)
    fig_fluxes = plt.figure(figsize=(16,8))
    ax1 = fig_fluxes.add_subplot(1,1,1)
    ax1.plot(z, enthalpy, label="h flux")
    ax1.plot(z, kappa, label=r"$\kappa\nabla T$")
    ax1.plot(z, KE, label="KE flux")
    ax1.plot(z, enthalpy+kappa+KE, color='black', linestyle='dashed', label='total')
    ax1.legend()
    ax1.set_xlabel("z")
    ax1.set_ylabel("energy fluxes")
    figs["fluxes"]=fig_fluxes

    fig_fluxes = plt.figure(figsize=(16,8))
    ax1 = fig_fluxes.add_subplot(1,1,1)
    ax1.plot(z, enthalpy, label="h flux")
    ax1.plot(z, kappa-kappa[0], label=r"$\kappa\nabla T$")
    ax1.plot(z, KE, label="KE flux")
    ax1.plot(z, enthalpy+kappa+KE-enthalpy[0]-kappa[0]-KE[0], color='black', linestyle='dashed', label='total')
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
        print(output_path)
        main(args['<files>'], output_path=output_path)


