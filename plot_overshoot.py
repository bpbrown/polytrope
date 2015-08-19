"""
Plot overshoot from joint analysis files.

Usage:
    plot_overshoot.py [options]

Options:
    --output=<output>  Output directory [default: ./]
    --verbose          Make diagnostic plots of each sim
"""
import numpy as np
from analysis import cheby_newton_root, interp_newton_root

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import analysis

def plot_diagnostics(z, norm_diag, roots, output_path='/.', boundary=None):
    figs = {}
    apjfig = analysis.APJSingleColumnFigure()

    min_plot = 1
    max_plot = np.log(5) # half a log-space unit above 1
    plot_floor = 1e-7

    for key in norm_diag:
        color = next(apjfig.ax._get_lines.color_cycle)
        
        if key=='KE_flux' or key=="grad_s" or key=="grad_s_mean" or key=="grad_s_post" or key=="s_mean" or key=="s_tot":
            analysis.semilogy_posneg(apjfig.ax, z, norm_diag[key][1], label=norm_diag[key][0], color=color)
        else:
            apjfig.ax.semilogy(z, norm_diag[key][1], label=norm_diag[key][0], color=color)

        apjfig.ax.axvline(x=roots[key][0], linestyle='dotted', color=color)
        min_plot = min(min_plot, np.min(np.abs(norm_diag[key][1])))

    min_plot = max(plot_floor, min_plot)

    apjfig.ax.axhline(y=1e-2, color='black', linestyle='dashed')
    apjfig.ax.axhline(y=1e-4, color='black', linestyle='dashed')
    if boundary is not None:
        apjfig.ax.axvline(x=boundary, color='black', linestyle='dotted')
    apjfig.legend(loc="upper left", title="normalized by max", fontsize=6)
    apjfig.ax.set_xlabel("z")
    apjfig.ax.set_ylabel("overshoot diags")#, \n normalized by max")
    apjfig.ax.set_ylim(min_plot, max_plot)
    padding = 0.1*np.max(z)
    xmin = np.min(z) - padding
    xmax = np.max(z) + padding
    apjfig.ax.set_xlim(xmin, xmax)
    figs["overshoot"]=apjfig

    for key in figs.keys():
        figs[key].savefig(output_path+'diag_{}.png'.format(key), dpi=600)

def plot_overshoot_times(times, overshoot, average_overshoot=None, output_path='./'):
    apjfig = analysis.APJSingleColumnFigure()
    ref = 'grad_s_mean'
    ref_depth = overshoot[ref]
    min_z = 100
    max_z = 0
    logger.info("reference depth:")
    logger.info("{} -- {}".format(ref, ref_depth))
    for key in overshoot:
        if key!=ref and key!='grad_s':
            color = next(apjfig.ax._get_lines.color_cycle)
            logger.info("{:10s} -- OV: {}".format(key, ref_depth - overshoot[key]))
            q = overshoot[key] #np.abs(overshoot[key]-ref_depth)
            apjfig.ax.plot(times, q, label=key, color=color)
            if average_overshoot is not None:
                apjfig.ax.axhline(average_overshoot[key], linestyle='dashed', color=color)
        min_z = min(min_z, np.min(q))
        max_z = max(max_z, np.max(q))

    z_pad = 0.01*max_z
    apjfig.ax.set_ylim(min_z-z_pad, max_z+z_pad)
    apjfig.legend(loc="upper right", title="diagnostics", fontsize=6)
    apjfig.ax.set_xlabel("time")
    apjfig.ax.set_ylabel("$z_o$ of overshoot")
    apjfig.savefig(output_path+"overshoot_timetrace.png", dpi=600)
        
def diagnose_overshoot(averages, z, boundary=None, output_path='./', verbose=False):

    def norm(f):
        return f/np.max(np.abs(f))

    norm_diag = OrderedDict()
    norm_diag['enstrophy'] = ('enstrophy', norm(averages['enstrophy']))
    norm_diag['KE'] = ('KE', norm(averages['KE']))
    #norm_diag['KE_flux'] = ('KE_flux', norm(averages['KE_flux_z']))

    try:
        #norm_diag['grad_s'] = (r'$\nabla (s_0+s_1)$', norm(averages['grad_s_tot']))
        norm_diag['grad_s_mean'] = (r'$\nabla (s_0)$', norm(averages['grad_s_mean']))
    except:
        logger.info("Missing grad_s from outputs; trying numeric gradient option")
        dz = np.gradient(z)
        try:
            #norm_diag['grad_s*'] = (r'$\nabla (s_0+s_1)^*$', np.gradient(norm(averages['s_tot']), dz))
            norm_diag['grad_s_mean*'] = (r'$\nabla (s_0)^*$', np.gradient(norm(averages['s_mean']), dz))
        except:
            logger.info("Missing s_tot from outputs")

    norm_diag['s_mean'] = (r'$s_0$', norm(averages['s_mean']))
    norm_diag['s_tot'] = (r'$s_0+s_1$', norm(averages['s_tot']))

    # estimate penetration depths
    overshoot_depths = OrderedDict()
    roots = OrderedDict()
    
    def poor_mans_root(z, f):
        i_near = (np.abs(f)).argmin()
        return z[i_near], f[i_near], i_near
    
    for key in norm_diag:
        if key=="KE_flux" or key=='grad_s' or key=='grad_s_mean' or key=='grad_s_tot' or key=="s_mean" or key=="s_tot":
            threshold = 0
            if key=="s_tot" or key=="s_mean":
                # grab the top of the atmosphere value
                threshold = norm_diag[key][1][-1]
                logger.info("key: {} has threshold {}".format(key, threshold))
                
            root_color = 'blue'
            criteria = norm_diag[key][1] - threshold
            if key=="KE_flux":
                root_color = 'darkgreen'
        else:
            threshold = 1e-2
            root_color = 'red'
            criteria = np.log(norm_diag[key][1])-np.log(threshold)
        logger.info("key {}".format(key))
        
        z_root_poor, f, i = poor_mans_root(z, criteria)
        #i_sort = np.argsort(z)
        #z_search = np.copy(z[i_sort])
        #criteria_search = np.copy(criteria[i_sort])
        z_search = np.copy(z)
        criteria_search = np.copy(criteria)
        z_root = interp_newton_root(z_search, criteria_search, z0=None)
        #z_root = cheby_newton_root(z, criteria, z0=None)
 
        overshoot_depths[key] = z_root
        roots[key] = (z_root, root_color)
        
        logger.info("poor man: {:>10s} : found root z={}".format(key, z_root_poor))
        logger.info("  newton: {:>10s} : found root z={}".format(key, z_root))

    if verbose:
        logger.info("Plotting diagnostics in {}".format(output_path))
        plot_diagnostics(z, norm_diag, roots, output_path=output_path)
        
    return overshoot_depths

def overshoot_time_trace(averages, z, times, average_overshoot=None, output_path='./'):
    single_time = OrderedDict()
    overshoot_depths = OrderedDict()
    for i, time in enumerate(times):
        for key in averages:
            single_time[key] = averages[key][i,0,:]
        single_overshoot_depths = diagnose_overshoot(single_time, z)
        if i == 0:
            for key in single_overshoot_depths:
                overshoot_depths[key] = single_overshoot_depths[key]
        else:
            for key in single_overshoot_depths:
                overshoot_depths[key] = np.append(overshoot_depths[key], single_overshoot_depths[key])
                
    plot_overshoot_times(times, overshoot_depths, average_overshoot=average_overshoot, output_path=output_path)

    
def analyze_case(files, verbose=False, output_path=None):
    data = analysis.Profile(files)
    logger.info("read in data from {}".format(data.files))

    averages = data.average
    std_devs = data.std_dev
    times = data.times
    z = data.z
    delta_t = times[-1]-times[0]
    logger.info("Averaged over interval t = {:g} -- {:g} for total delta_t = {:g}".format(times[0], times[-1], delta_t))

    if output_path is None:
        import pathlib
        data_dir = files[0].split('/')[0]
        data_dir += '/'
        output_path = pathlib.Path(data_dir).absolute()

    overshoot_depths = diagnose_overshoot(averages, z, output_path=str(output_path)+'/', verbose=verbose)
    for key in overshoot_depths:
        print("{} --> z={}".format(key, overshoot_depths[key]))

    if verbose:
        overshoot_time_trace(data.data, z, times, average_overshoot=overshoot_depths, output_path=str(output_path)+'/')

    return overshoot_depths

def analyze_all_cases(stiffness_file_list, **kwargs):
    overshoot = OrderedDict()
    first_run = True
    
    for stiffness, files in stiffness_file_list:
        overshoot_one_case = analyze_case(files, **kwargs)
        for key in overshoot_one_case:
            if first_run:
                overshoot[key] = np.array(overshoot_one_case[key])
            else:
                overshoot[key] = np.append(overshoot[key], overshoot_one_case[key])
               
        if first_run:
            stiffness_array = np.array(stiffness)
            first_run = False
        else:
            stiffness_array = np.append(stiffness_array, stiffness)
            
    return stiffness_array, overshoot
    

def plot_overshoot(stiffness, overshoot, output_path='./'):
    apjfig = analysis.APJSingleColumnFigure()
    ref = 'grad_s_mean'
    ref_depth = overshoot[ref]
    min_z = 100
    max_z = 0
    logger.info("reference depth:")
    logger.info("{} -- {}".format(ref, ref_depth))
    for key in overshoot:
        if key!=ref and key!='grad_s':
            logger.info("{:10s} -- OV: {}".format(key, ref_depth - overshoot[key]))
            q = np.abs(overshoot[key]-ref_depth)
            apjfig.ax.loglog(stiffness, q, label=key, marker='o')
        min_z = min(min_z, np.min(q))
        max_z = max(max_z, np.max(q))
        
    apjfig.ax.set_ylim(0.9*min_z, 1.1*max_z)
    apjfig.legend(loc="upper right", title="diagnostics", fontsize=6)
    apjfig.ax.set_xlabel("Stiffness S")
    apjfig.ax.set_ylabel("$\Delta z$ of overshoot")
    apjfig.savefig(output_path+"overshoot.png", dpi=600)
    
def main(output_path='./', **kwargs):
    import glob

    file_list = [(1e1, glob.glob('FC_multi_nrhocz1_Ra1e7_S1e1/profiles/profiles_s[8,9].h5')),
                 (1e2, glob.glob('FC_multi_nrhocz1_Ra1e7_S1e2/profiles/profiles_s[8,9].h5')),
                 (1e3, glob.glob('FC_multi_nrhocz1_Ra1e7_S1e3/profiles/profiles_s[8,9].h5')),
                 (1e4, glob.glob('FC_multi_nrhocz1_Ra1e7_S1e4/profiles/profiles_s[8,9].h5')),
                 (1e5, glob.glob('FC_multi_nrhocz1_Ra1e7_S1e5/profiles/profiles_s[8,9].h5'))]

    file_list = [(1e2, glob.glob('FC_multi_nrhocz1_Ra1e6_S1e2/profiles/profiles_s12?.h5')),
                 (1e3, glob.glob('FC_multi_nrhocz1_Ra1e6_S1e3/profiles/profiles_s12?.h5')),
                 (3e3, glob.glob('FC_multi_nrhocz1_Ra1e6_S3e3/profiles/profiles_s12?.h5')),
                 (1e4, glob.glob('FC_multi_nrhocz1_Ra1e6_S1e4/profiles/profiles_s12?.h5')),
                 (3e4, glob.glob('FC_multi_nrhocz1_Ra1e6_S3e4/profiles/profiles_s12?.h5')),
                 (1e5, glob.glob('FC_multi_nrhocz1_Ra1e6_S1e5/profiles/profiles_s12?.h5'))]

#    file_list = [(1e3, glob.glob('FC_multi_nrhocz3.5_Ra1e6_S1e3/profiles/profiles_s8?.h5')),
#                 (1e4, glob.glob('FC_multi_nrhocz3.5_Ra1e6_S1e4/profiles/profiles_s8?.h5')),
#                 (1e5, glob.glob('FC_multi_nrhocz3.5_Ra1e6_S1e5/profiles/profiles_s8?.h5'))]


    stiffness, overshoot = analyze_all_cases(file_list, **kwargs)
    plot_overshoot(stiffness, overshoot, output_path=output_path)
     
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
    main(output_path=str(output_path)+'/', verbose=args['--verbose'])


