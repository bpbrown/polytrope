import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

eps = 5e-02
ra_start = 60
ra_stop = 90
directory = './'

total_dir = directory + 'FC_onset_solve/'
file_name = '{0}evals_eps_{1:.0e}_ras_{2:04d}-{3:04d}.h5'.format(total_dir, eps, ra_start, ra_stop)

def plot_flow_profiles(file_name, profiles=['w', 'T1']):
    import os
    out_dir = file_name[:-3]+'_diagnostics/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f = h5py.File(file_name, 'r')
    keys = []
    for raw_key in f['keys'][:]:
        keys.append(str(raw_key[0])[2:-1])
    

    prof_keys = []
    for i in range(len(profiles)):
        prof = profiles[i]
        if len(prof_keys) != i+1:
            prof_keys.append([])
        for key in keys:
            if prof in key:
                prof_keys[i].append(key)
        prof_keys[i] = sorted(prof_keys[i])

    x = f['x'][:]
    z = f['z'][0]
    zs, xs = np.meshgrid(z, x)

    print(x)
    print(xs, zs)

    #loop through each wavenumber
    fig = plt.figure(figsize=(20,10))
    for i in range(len(prof_keys[0])):
        count = 1
        for j in range(f[prof_keys[0][i]][:].shape[0]):
            w_key = prof_keys[0][i]
            t_key = prof_keys[1][i]
            
            fig.clear()
            ax = fig.add_subplot(1,2,1)
            bx = fig.add_subplot(1,2,2)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.pcolormesh(xs, zs, f[w_key][j], cmap='PuOr_r')
            ax.set_ylim(z[0], z[-1])
            ax.set_xlim(x[0], x[-1])
            ax.set_title(profiles[0])
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(r'Vertical Velocity', rotation=270)

            divider = make_axes_locatable(bx)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = bx.pcolormesh(xs, zs, f[t_key][j], cmap='RdBu_r')
            bx.set_ylim(z[0], z[-1])
            bx.set_xlim(x[0], x[-1])
            bx.set_title(profiles[1])
            bx.set_xlabel('x')
            bx.set_ylabel('z')
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(r'Temp Fluctuations', rotation=270)


            plt.subplots_adjust(top=0.88, wspace=0.5, hspace=0.4)
            fig.suptitle('Wavenumber {0}; Ra'.format(w_key.split('_')[0]), fontsize=16)
            plt.savefig(out_dir+'wavenum{0}_{1:04d}'.format(w_key.split('_')[0], count), dpi=100)
            count +=1

print(file_name[:-3])
plot_flow_profiles(file_name)
