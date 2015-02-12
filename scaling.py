#!/usr/bin/env python3
"""
Perform scaling runs on special scaling scripts.

Usage:
    scaling.py run <scaling_script> [--verbose]
    scaling.py plot <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

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
import scaling_plot
import pathlib

from dedalus2.tools.parallel import Sync

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def do_scaling_run(scaling_script, resolution, CPU_set, test_type='minimal', mpirun='mpirun', verbose=None):

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

        proc = subprocess.Popen([mpirun, "-np","{:d}".format(ENV_N_TOTAL_CPU), "python3", scaling_script], 
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
    data_set['label'] = '{:d}x{:d}'.format(sim_nx, sim_nz)
    write_scaling_run(data_set)

    end_time = time.time()
    print(40*'*')
    print('time to test {:d}x{:d}: {:8.3g}'.format(sim_nx,sim_nz, end_time-start_time))
    print(40*'*')

    return data_set

def write_scaling_run(data_set):
        
    scaling_file = shelve.open('scaling_data_'+data_set['label']+'.db', flag='n')
    scaling_file['data'] = data_set
    scaling_file.close()

def read_scaling_run(file):
    print("opening file {}".format(file))
    scaling_file = shelve.open(file, flag='r')
    data_set = scaling_file['data']
    scaling_file.close()
    return data_set
    
if __name__ == "__main__":
    
    from docopt import docopt

    start_time = time.time()

    args = docopt(__doc__)
    if args['run']:
        #CPU_set = [1, 2, 4, 8, 16, 32, 64, 128]
        CPU_set = [512, 256, 128, 64]
        resolution = [2048, 1024]
        data_set = do_scaling_run(args['<scaling_script>'], resolution, CPU_set, mpirun='mpirun', verbose=args['--verbose'])
    elif args['plot']:
        output_path = pathlib.Path(args['--output']).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        data_set = read_scaling_run(args['<files>'][0])

    fig_set, ax_set = scaling_plot.initialize_plots(4)
    scaling_plot.plot_scaling_run(data_set, ax_set)
    
    scaling_plot.finalize_plots(fig_set, ax_set)

    end_time = time.time()
    print(40*'=')
    print('time to do all tests: {:f}'.format(end_time-start_time))
    print(40*'=')

