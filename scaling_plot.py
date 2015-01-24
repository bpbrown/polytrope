import os
import numpy as np
import itertools
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shelve

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def read_ASH_scaling_run(file):

    N_x_cpu = []
    N_y_cpu = []
    N_total_cpu = []
    N_x = []
    N_y = []
    N_z = []
    wall_time = []
    wall_time_per_iter = []
    text_file = open(file, 'r')
    
    #open file and read it
    for line in text_file:
            if line.startswith('scaling:'):
                split_line = line.split()
                print(split_line)
                N_x_cpu.append(num(split_line[1]))
                N_y_cpu.append(num(split_line[2]))
                N_total_cpu.append(num(split_line[3]))

                # here re-ordered, so that r lines up with z
                N_x.append(num(split_line[6]))
                N_y.append(num(split_line[5]))
                N_z.append(num(split_line[4]))

                wall_time.append(num(split_line[7]))
                wall_time_per_iter.append(num(split_line[8]))
                
    sim_nx = N_x[0]
    sim_ny = N_y[0]
    sim_nz = N_z[0]

    N_x_cpu = np.array(N_x_cpu)
    N_y_cpu = np.array(N_y_cpu)
    N_total_cpu = np.array(N_total_cpu)
    N_x = np.array(N_x)
    N_y = np.array(N_y)
    N_z = np.array(N_z)

    # stub for startup time, as I didn't measure that in pseudo yet
    startup_time = np.ones(len(wall_time)) 
    
    wall_time = np.array(wall_time)
    wall_time_per_iter = np.array(wall_time_per_iter)

    work = wall_time_per_iter/(N_x*N_y*N_z)
    work_per_core = N_total_cpu*wall_time_per_iter/(N_x*N_y*N_z)

    
    data_set = [sim_nx, sim_ny, sim_nz, 
                N_x_cpu, N_y_cpu, N_total_cpu,
                N_x, N_y, N_z, 
                startup_time,
                wall_time, wall_time_per_iter,
                work, work_per_core]

    return data_set

def read_scaling_run(resolution):
    sim_nx = resolution[0]
    sim_ny = resolution[1]
    sim_nz = resolution[2]
    
    resolution_string = '{:d}x{:d}x{:d}'.format(sim_nx, sim_ny, sim_nz)
    scaling_file = shelve.open('scaling_data_'+resolution_string+'.db', flag='r')
    sim_nx = scaling_file['nx']
    sim_ny = scaling_file['ny']
    sim_nz = scaling_file['nz']
    N_x_cpu = scaling_file['N_x_cpu']
    N_y_cpu = scaling_file['N_y_cpu']
    N_total_cpu = scaling_file['N_total_cpu']
    N_x = scaling_file['N_x']
    N_y = scaling_file['N_y']
    N_z = scaling_file['N_z']
    startup_time = scaling_file['startup_time']
    wall_time = scaling_file['wall_time']
    wall_time_per_iter = scaling_file['wall_time_per_iter']
    work = scaling_file['work']
    work_per_core = scaling_file['work_per_core']

    data_set = [sim_nx, sim_ny, sim_nz, 
                N_x_cpu, N_y_cpu, N_total_cpu,
                N_x, N_y, N_z, 
                startup_time, wall_time, wall_time_per_iter,
                work, work_per_core]
    #print(data_set)
    return data_set

def plot_scaling_run(data_set, ax_set, 
                     ideal_curves = True, scale_to = False, scale_to_resolution=[128,128,128], 
                     linestyle='solid', marker='o', color='None', explicit_label = True):
    
    sim_nx = data_set[0]
    sim_ny = data_set[1]
    sim_nz = data_set[2]
    N_x_cpu = data_set[3]
    N_y_cpu = data_set[4]
    N_total_cpu = data_set[5]
    N_x = data_set[6]
    N_y = data_set[7]
    N_z = data_set[8]
    startup_time = data_set[9]
    wall_time = data_set[10]
    wall_time_per_iter = data_set[11]
    work = data_set[12]
    work_per_core = data_set[13]

    resolution = [sim_nx, sim_ny, sim_nz]
    
    if color is 'None':
        color=next(ax_set[0]._get_lines.color_cycle)


    scale_to_factor = np.prod(np.array(scale_to_resolution))/np.prod(np.array(resolution))
    scale_factor_inverse = np.int(np.rint((1./scale_to_factor)**(1./3.)))

    if explicit_label:
        label_string = r'${:d}\times{:d}\times{:d}$'.format(sim_nx, sim_ny, sim_nz)
        scaled_label_string = r'${:d}\times{:d}\times{:d}/{:d}^3$'.format(sim_nx, sim_ny, sim_nz, scale_factor_inverse)
    else:
        label_string = r'${:d}^3$'.format(sim_nz)
        scaled_label_string = r'${:d}^3/{:d}^3$'.format(sim_nz, scale_factor_inverse)
    
    ax_set[0].loglog(N_total_cpu, wall_time, label=label_string, 
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[1].loglog(N_total_cpu, wall_time_per_iter, label=label_string,
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[2].loglog(N_total_cpu, work_per_core/1e-6, label=label_string,
                     marker=marker, linestyle=linestyle, color=color)

    ax_set[3].loglog(N_total_cpu, startup_time, label=label_string,
                     marker=marker,  linestyle=linestyle, color=color)


    if scale_to:
        print("scaling by {:f} or (1/{:d})^3".format(scale_to_factor, scale_factor_inverse))
        ax_set[0].loglog(N_total_cpu, wall_time*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--', color=color)
        
        ax_set[1].loglog(N_total_cpu, wall_time_per_iter*scale_to_factor, marker=marker,
                         label=scaled_label_string, linestyle='--',color=color)

    if ideal_curves:
        ideal_cores = N_total_cpu
        ideal_time = wall_time[0]*(N_total_cpu[0]/N_total_cpu)
        ideal_time_per_iter = wall_time_per_iter[0]*(N_total_cpu[0]/N_total_cpu)

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

def finalize_plots(fig_set, ax_set):
    
    ax_set[0].set_title('Wall time (100 iterations) rb3d_scaling.py')
    ax_set[0].set_xlabel('N-core')
    ax_set[0].set_ylabel('total time [s]')
    legend_with_ideal(ax_set[0], loc='lower left')
    fig_set[0].savefig('scaling_time.png')

    
    ax_set[1].set_title('Wall time per iteration rb3d_scaling.py')
    ax_set[1].set_xlabel('N-core')
    ax_set[1].set_ylabel('time/iter [s]')
    legend_with_ideal(ax_set[1], loc='lower left')
    fig_set[1].savefig('scaling_time_per_iter.png')

    
    ax_set[2].set_title('Normalized work rb3d_scaling.py')
    ax_set[2].set_xlabel('N-core')
    ax_set[2].set_ylabel('N-cores * (time/iter/grid) [$\mu$s]')
    ax_set[2].legend(loc='upper left')
    fig_set[2].savefig('scaling_work.png')

    ax_set[3].set_title('startup time rb3d_scaling.py')
    ax_set[3].set_xlabel('N-core')
    ax_set[3].set_ylabel('startup time [s]')
    ax_set[3].legend(loc='upper left')
    fig_set[3].savefig('scaling_startup.png')

    
if __name__ == "__main__":
    
    fig_set, ax_set = initialize_plots(4)

    include_low_res = True

    if include_low_res:
        resolution = [128,128,128]
        data_set = read_scaling_run(resolution)
        plot_scaling_run(data_set, ax_set)

        scale_to_resolution = resolution
        scale_to = True
    
        resolution = [256,256,256]
        data_set = read_scaling_run(resolution)
        plot_scaling_run(data_set, ax_set, 
                     scale_to=scale_to, scale_to_resolution=scale_to_resolution)

    
    resolution = [512,512,512]
    #scale_to_resolution = resolution
    #scale_to = True
    data_set = read_scaling_run(resolution)
    plot_scaling_run(data_set, ax_set,
                     scale_to=scale_to, scale_to_resolution=scale_to_resolution)

    resolution = [1024,512,256]
    data_set = read_scaling_run(resolution)
    plot_scaling_run(data_set, ax_set,
                     scale_to=scale_to, scale_to_resolution=scale_to_resolution)

    resolution = [1024,1024,1024]
    data_set = read_scaling_run(resolution)
    plot_scaling_run(data_set, ax_set,
                     scale_to=scale_to, scale_to_resolution=scale_to_resolution, ideal_curves=False)

    scale_to = False
    data_set = read_ASH_scaling_run('pseudo_scaling_test.o2675079')
    plot_scaling_run(data_set, ax_set, linestyle='None', marker='*', color='red', explicit_label=True,
                     scale_to=scale_to, scale_to_resolution=scale_to_resolution)
    
    finalize_plots(fig_set, ax_set)

