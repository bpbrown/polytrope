"""
Plot overshoot from joint analysis files.

Usage:
    plot_fluxes.py <files> ... [--output=<output>]

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

def make_plots(norm_diag):
    figs = {}
    apjfig = analysis.APJSingleColumnFigure()

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

    #apjfig.ax.axvline(x=z_root_poor, linestyle='dotted', color=root_color)

    for key in figs.keys():
        figs[key].savefig(output_path+'diag_{}.png'.format(key))

def diagnose_overshoot(averages, z, boundary=None, output_path='./'):
    import scipy.optimize as scpop

    def norm(f):
        return f/np.max(np.abs(f))

    norm_diag = OrderedDict()
    norm_diag['enstrophy'] = ('enstrophy', norm(averages['enstrophy']))
    norm_diag['KE'] = ('KE', norm(averages['KE']))
    norm_diag['KE_flux'] = ('KE_flux', norm(averages['KE_flux_z']))
    dz = np.gradient(z)
    try:
        norm_diag['grad_s'] = (r'$\nabla (s_0+s_1)$', norm(averages['grad_s_tot']))
        norm_diag['grad_s_mean'] = (r'$\nabla (s_0)$', norm(averages['grad_s_mean']))
    except:
        logger.info("Missing grad_s from outputs; trying numeric gradient option")
        try:
            norm_diag['grad_s*'] = (r'$\nabla (s_0+s_1)^*$', np.gradient(norm(averages['s_tot']), dz))
            norm_diag['grad_s_mean*'] = (r'$\nabla (s_0)^*$', np.gradient(norm(averages['s_mean']), dz))
        except:
            logger.info("Missing s_tot from outputs")

    # estimate penetration depths
   overshoot_depths = OrderedDict()

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

        z_root_poor, f, i = poor_mans_root(norm_diag[key][1]-threshold, z)
        overshoot_depths[key] = z_root_poor
        logger.info("poor man: {} found root z={}".format(key, z_root_poor))
        
    return overshoot_depths
    
def main(files, output_path='./'):
    data = analysis.Profile(files)
    averages = data.average
    std_devs = data.std_dev
    times = data.times
    z = data.z
    delta_t = times[-1]-times[0]
    logger.info("Averaged over interval t = {:g} -- {:g} for total delta_t = {:g}".format(times[0], times[-1], delta_t))
    overshoot_depths = diagnose_overshoot(averages, z, output_path=output_path)
    for key in overshoot_depths:
        print("{} --> z={}".format(key, overshoot_depths[key]))


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    logger.info("output to {}".format(output_path))
    main(args['<files>'], output_path=str(output_path)+'/')


