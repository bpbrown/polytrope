#!/usr/bin/env python3
"""
Perform scaling runs on special scaling scripts.

Usage:
    scaling.py run <scaling_script> [<z_resolution> --label=<label> --verbose]
    scaling.py plot <files>... [--output=<dir> --rescale=<rescale>]

Options:
     <z_resolution>  set Z resolution in chebyshev direction; X resolution is 2x larger.
    --output=<dir>   Output directory [default: ./scaling]
    --label=<label>  Label for output file
    --verbose        Print verbose output at end of each run (stdout and stderr)
    --rescale=<rescale>  rescale plots to particular Z resolution comparison case     
"""
import os
import numpy as np
import itertools
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shelve
import pathlib

from dedalus.tools.parallel import Sync

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def do_scaling_run(scaling_script, resolution, CPU_set, test_type='minimal', mpirun='mpirun', verbose=None, label=None):

    print('testing {}, from {:d} to {:d} cores'.format(scaling_script, np.min(CPU_set),np.max(CPU_set)))
    start_time = time.time()
    
    scaling_test_set = CPU_set
    sim_nx = resolution[0] 
    sim_nz = resolution[1]
    
    N_total_cpu = []
    N_x = []
    N_z = []
    startup_time = []
    wall_time = []
    wall_time_per_iter = []
    work = []
    work_per_core = []
    
    for ENV_N_TOTAL_CPU in scaling_test_set:

        print("scaling test of {}".format(scaling_script),
              " at {:d}x{:d}".format(sim_nx, sim_nz),
              " on {:d} cores".format(ENV_N_TOTAL_CPU))

        test_env = dict(os.environ, 
                        N_X='{:d}'.format(sim_nx),
                        N_Z='{:d}'.format(sim_nz),
                        N_TOTAL_CPU='{:d}'.format(ENV_N_TOTAL_CPU))

        proc = subprocess.Popen([mpirun, "-np","{:d}".format(ENV_N_TOTAL_CPU), 
                                 "--bind-to", "core", "--map-by", "core", 
                                 "python3", scaling_script], 
                                env=test_env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = proc.communicate()

        if verbose:
            for line in stdout.splitlines():
                print("out: {}".format(line))
            
            for line in stderr.splitlines():
                print("err: {}".format(line))
        
        for line in stdout.splitlines():
            if line.startswith('scaling:'):
                split_line = line.split()
                print(split_line)
                N_total_cpu.append(num(split_line[1]))

                N_x.append(num(split_line[2]))
                N_z.append(num(split_line[3]))

                startup_time.append(num(split_line[4]))
                wall_time.append(num(split_line[5]))
                wall_time_per_iter.append(num(split_line[6]))

                work.append(num(split_line[7]))
                work_per_core.append(num(split_line[8]))

    # change data storage to numpy arrays
    N_total_cpu = np.array(N_total_cpu)
    N_x = np.array(N_x)
    N_z = np.array(N_z)
    startup_time = np.array(startup_time)
    wall_time = np.array(wall_time)
    wall_time_per_iter = np.array(wall_time_per_iter)
    work = np.array(work)
    work_per_core = np.array(work_per_core)
    
    print(40*'-')
    print("scaling results")
    for i, temp in enumerate(N_total_cpu):
        print(N_total_cpu[i], N_z[i], startup_time[i], wall_time[i], wall_time_per_iter[i])

    data_set = dict()
    data_set['script'] = scaling_script
    data_set['sim_nx'] = sim_nx
    data_set['sim_nz'] = sim_nz
    data_set['N_total_cpu'] = N_total_cpu
    data_set['N_x'] = N_x
    data_set['N_z'] = N_z
    data_set['startup_time'] = startup_time
    data_set['wall_time'] = wall_time
    data_set['wall_time_per_iter'] = wall_time_per_iter
    data_set['work'] = work
    data_set['work_per_core'] = work_per_core
    data_set['file_label'] = '{:d}x{:d}'.format(sim_nx, sim_nz)
    
    data_set['plot_label'] = r'${:d}\times{:d}$'.format(sim_nx, sim_nz)        
    data_set['plot_label_short'] = r'${:d}^2$'.format(sim_nz)

    if not label is None:
        data_set['plot_label'] = data_set['plot_label'] + "-" + label
    write_scaling_run(data_set, label=label)

    end_time = time.time()
    print(40*'*')
    print('time to test {:d}x{:d}: {:8.3g}'.format(sim_nx,sim_nz, end_time-start_time))
    print(40*'*')

    return data_set

def write_scaling_run(data_set, label=None):
    file_name = 'scaling_data_'+data_set['file_label']
    if not label is None:
        file_name = file_name+'_'+label
    file_name = file_name+'.db'
    
    print("writing file {}".format(file_name))
    scaling_file = shelve.open(file_name, flag='n')
    data_set['file_name'] = file_name
    scaling_file['data'] = data_set
    scaling_file.close()

def read_scaling_run(file):
    print("opening file {}".format(file))
    scaling_file = shelve.open(file, flag='r')
    data_set = scaling_file['data']
    scaling_file.close()
    return data_set

# Plotting routines
def plot_scaling_run(data_set, ax_set, 
                     ideal_curves = True, scale_to = False, scale_to_resolution=None, 
                     linestyle='solid', marker='o', color='None', explicit_label = True, dim=2):
    
    sim_nx = data_set['sim_nx']
    sim_nz = data_set['sim_nz']
    N_total_cpu = data_set['N_total_cpu']
    N_x = data_set['N_x'] 
    N_z = data_set['N_z']
    if dim==3:
        sim_ny = data_set['sim_ny']
        N_y = data_set['N_y'] 
        N_x_cpu = data_set['N_x_cpu']
        N_y_cpu = data_set['N_y_cpu']
        
    startup_time = data_set['startup_time'] 
    wall_time = data_set['wall_time']
    wall_time_per_iter = data_set['wall_time_per_iter']
    work = data_set['work']
    work_per_core = data_set['work_per_core']
        
    if dim == 2:
        resolution = [sim_nx, sim_nz]
        if scale_to_resolution is None:
            scale_to_resolution = [128,128]
    elif dim == 3 :
        resolution = [sim_nx, sim_ny, sim_nz]
        if scale_to_resolution is None:
            scale_to_resolution = [128,128,128]
    
    if color is 'None':
        color=next(ax_set[0]._get_lines.color_cycle)

    scale_to_factor = np.prod(np.array(scale_to_resolution))/np.prod(np.array(resolution))
    scale_factor_inverse = np.int(np.rint((1./scale_to_factor)**(1/dim)))

    if explicit_label:
        label_string = data_set['plot_label']
        scaled_label_string = data_set['plot_label'] + r'$/{:d}^{:d}$'.format(scale_factor_inverse, dim)
    else:
        label_string = data_set['plot_label_short']
        scaled_label_string = data_set['plot_label_short'] + r'$/{:d}^{:d}$'.format(scale_factor_inverse, dim)
    
    ax_set[0].loglog(N_total_cpu, wall_time, label=label_string, 
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[1].loglog(N_total_cpu, wall_time_per_iter, label=label_string,
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[2].loglog(N_total_cpu, work_per_core/1e-6, label=label_string,
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[3].loglog(N_total_cpu, startup_time, label=label_string,
                     marker=marker,  linestyle=linestyle, color=color)

    i_max = N_total_cpu.argmax()
    ax_set[4].plot(N_total_cpu[i_max], work_per_core[i_max]/1e-6, label=label_string,
                     marker=marker,  linestyle=linestyle, color=color)

    if scale_to and scale_to_factor != 1:
        print("scaling by {:f} or (1/{:d})^{:d}".format(scale_to_factor, scale_factor_inverse, dim))
        ax_set[0].loglog(N_total_cpu, wall_time*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--', color=color)
        
        ax_set[1].loglog(N_total_cpu, wall_time_per_iter*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--',color=color)

    if ideal_curves:
        ideal_cores = N_total_cpu
        i_min = np.argmin(N_total_cpu)
        ideal_time = wall_time[i_min]*(N_total_cpu[i_min]/N_total_cpu)
        ideal_time_per_iter = wall_time_per_iter[i_min]*(N_total_cpu[i_min]/N_total_cpu)

        ax_set[0].loglog(ideal_cores, ideal_time, linestyle='--', color='black')
        
        ax_set[1].loglog(ideal_cores, ideal_time_per_iter, linestyle='--', color='black')

def initialize_plots(num_figs):
    fig_set = []
    ax_set = []

    for i in range(num_figs):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig_set.append(fig)
        ax_set.append(ax)
    
    return fig_set, ax_set


def legend_with_ideal(ax, loc='lower left'):
    handles, labels = ax.get_legend_handles_labels()
    idealArtist = plt.Line2D((0,1),(0,0), color='black', linestyle='--')
    ax.legend([handle for i,handle in enumerate(handles)]+[idealArtist],
              [label for i,label in enumerate(labels)]+['ideal'],
              loc=loc, fontsize='small')    

def finalize_plots(fig_set, ax_set, script):
    
    ax_set[0].set_title('Wall time {}'.format(script))
    ax_set[0].set_xlabel('N-core')
    ax_set[0].set_ylabel('total time [s]')
    legend_with_ideal(ax_set[0], loc='lower left')
    fig_set[0].savefig('scaling_time.png')

    
    ax_set[1].set_title('Wall time per iteration {}'.format(script))
    ax_set[1].set_xlabel('N-core')
    ax_set[1].set_ylabel('time/iter [s]')
    legend_with_ideal(ax_set[1], loc='upper right')
    fig_set[1].savefig('scaling_time_per_iter.png')

    
    ax_set[2].set_title('Normalized work {}'.format(script))
    ax_set[2].set_xlabel('N-core')
    ax_set[2].set_ylabel('N-cores * (time/iter/grid) [$\mu$s]')
    ax_set[2].legend(loc='upper left')
    fig_set[2].savefig('scaling_work.png')

    ax_set[3].set_title('startup time {}'.format(script))
    ax_set[3].set_xlabel('N-core')
    ax_set[3].set_ylabel('startup time [s]')
    ax_set[3].legend(loc='lower right')
    fig_set[3].savefig('scaling_startup.png')

    ax_set[4].set_title('Normalized work {}'.format(script))
    ax_set[4].set_xlabel('N-core')
    ax_set[4].set_ylabel('N-cores * (time/iter/grid) [$\mu$s]')
    ax_set[4].legend(loc='upper left')
    fig_set[4].savefig('scaling_work_strong.png')



if __name__ == "__main__":
    
    from docopt import docopt
    
    fig_set, ax_set = initialize_plots(5)
    args = docopt(__doc__)
    if args['run']:
        if not args['<z_resolution>'] is None:
            n_z = num(args['<z_resolution>'])
            
            resolution = [2*n_z, n_z]
            n_z_2 = np.log(n_z)/np.log(2) # 2 pencils per core min (arange goes to -1 of top)
            n_z_2_min = n_z_2-4
            
            CPU_set = (2**np.arange(n_z_2_min, n_z_2)).astype(int)[::-1] # flip order so large numbers of cores are done first
            print("scaling run with {} on {} cores".format(resolution, CPU_set))
        else:
            CPU_set = [512, 256, 128, 64]
            resolution = [2048, 1024]
        
        start_time = time.time()
        data_set = do_scaling_run(args['<scaling_script>'], resolution, CPU_set, mpirun='mpirun', verbose=args['--verbose'], label=args['--label'])
        end_time = time.time()
        
        plot_scaling_run(data_set, ax_set)
        script = args['<scaling_script>']

        print(40*'=')
        print('time to do all tests: {:f}'.format(end_time-start_time))
        print(40*'=')

    elif args['plot']:
        output_path = pathlib.Path(args['--output']).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        if not args['--rescale'] is None:
            n_z_rescale = num(args['--rescale'])
            
            scale_to_resolution = [2*n_z_rescale, n_z_rescale]
            scale_to = True
        else:
            scale_to_resolution = [1, 1]
            scale_to = False
            
        for file in args['<files>']:
            data_set = read_scaling_run(file)
            plot_scaling_run(data_set, ax_set, scale_to=scale_to, scale_to_resolution=scale_to_resolution)
        script = data_set['script']
        
    finalize_plots(fig_set, ax_set, script)


