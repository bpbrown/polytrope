"""
Plot energy fluxes from joint analysis files.

Usage:
    plot_fluxes.py join <base_path>
    plot_fluxes.py plot <files>... [--output=<output>]

Options:
    --output=<output>  Output directory [default: ./fluxes]

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
    apjfig = analysis.APJSingleColumnFigure()

    def norm(f):
        return f/np.max(np.abs(f))

    norm_diag = OrderedDict()
    norm_diag['enstrophy'] = ('enstrophy', norm(averages['enstrophy']))
    norm_diag['KE'] = ('KE', norm(averages['KE']))
    norm_diag['KE_flux'] = ('KE_flux', norm(averages['KE_flux_z']))
    dz = np.gradient(z)
    try:
        norm_diag['grad_s_post'] = (r'$\nabla (s_0+s_1)^*$', np.gradient(norm(averages['s_tot']), dz))
    except:
        logger.info("Missing s_tot from outputs")

    try:
        norm_diag['grad_s'] = (r'$\nabla (s_0+s_1)$', norm(averages['grad_s_tot']))
        norm_diag['grad_s_mean'] = (r'$\nabla (s_0)$', norm(averages['grad_s_mean']))
    except:
        logger.info("Missing grad_s from outputs")

    min_plot = 1
    max_plot = np.log(5) # half a log-space unit above 1
    for key in norm_diag:
        if key=='KE_flux' or key=="grad_s" or key=="grad_s_mean" or key=="grad_s_post":
            analysis.semilogy_posneg(apjfig.ax, z, norm_diag[key][1], label=norm_diag[key][0])
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

    def poor_mans_root(f, z):
        i_near = (np.abs(f)).argmin()
        return z[i_near], f[i_near], i_near



    for key in norm_diag:
        if key=='grad_s' or key=='grad_s_mean' or key=='grad_s_tot':
            threshold = 0
            root_color = 'blue'
        else:
            threshold = 1e-2
            root_color = 'red'

        if key=='enstrophy':
            degree = 512    
            cheb_coeffs = np.polynomial.chebyshev.chebfit(x, norm_diag[key][1]-threshold, degree)
            cheb_interp = np.polynomial.chebyshev.Chebyshev(cheb_coeffs)

            #print(cheb_interp.roots())
            def newton_func(x_newton):
                return np.polynomial.chebyshev.chebval(x_newton, cheb_coeffs)

            try:
                x_root = scpop.newton(newton_func, x_start_search)
            except:
                logger.info("root find failed to converge")
                x_root = x_start_search
            z_root = (x_root+1)*Lz/2

            logger.info("newton: {} : found root z={} (x:{} -> {})".format(key, z_root, x_start_search, x_root))  
            #print(np.polynomial.chebyshev.chebroots(cheb_coeffs))
            #apjfig.ax.semilogy(z,np.polynomial.chebyshev.chebval(x, cheb_coeffs), label='fit')

        z_root_poor, f, i = poor_mans_root(norm_diag[key][1]-threshold, z)
        logger.info("poor man: {} found root z={}".format(key, z_root_poor))
        apjfig.ax.axvline(x=z_root_poor, linestyle='dotted', color=root_color)

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
        logger.info("output to {}".format(output_path))
        main(args['<files>'], output_path=str(output_path)+'/')


