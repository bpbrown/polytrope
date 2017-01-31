'''
Unit test of FC_poly.py
After short runs of old and new FC_poly.py, this file outputs relevant data for comparison.

Usage:
    FC_poly_test.py --outdir1=<outdir1> --outdir2=<outdir2> --plot_dir=<plot_dir>

Options:
    --outdir1=<outdir1>       The output directory of the new FC_poly.py run
    --outdir2=<outdir2>       The output directory of the old FC_poly.py run
    --plot_dir=<plot_dir>     Output directory for comparison plots between the two runs
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def pcolormesh_subplot(ax, xx, yy, field, cmap='RdYlBu_r', title=None):
    cm = ax.pcolormesh(xx, yy, field, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=1)
    cb  = plt.colorbar(cm, cax=cax, orientation='vertical', format="%1.2e")
    ax.set_xlim(np.min(xx), np.max(xx))
    ax.set_ylim(np.min(yy), np.max(yy))
    if title != None:
        cax.set_xlabel(title)

def profile_comparison(profiles, key, output_dir):
    data = [i.data[key] for i in profiles]
    N    = np.min(np.array([i.shape[0] for i in data]))
    data = [i[:N,0,:] for i in data]
    zs   = [i.z for i in profiles]
    ts   = [i.times[:N] for i in profiles]
    fig, ax = plt.subplots(2,2, figsize=(12,10), dpi=150)
    
    #Timeplots
    for i in range(len(profiles)):
        yy_t, xx_t = np.meshgrid(zs[i], ts[i])
        pcolormesh_subplot(ax[0,i], xx_t, yy_t, data[i], title='run_{:d}'.format(i+1))
    
    #2D error space
    abs_diff = np.abs((data[1]-data[0])/data[1])
    abs_diff = np.ma.masked_where(np.isnan(abs_diff), abs_diff)
    pcolormesh_subplot(ax[1,0], xx_t, yy_t, np.log10(abs_diff), cmap='Reds', title='log(rel err)')

    #Trace of rel err vs time.
    ax[1,1].plot(ts[0], np.mean(abs_diff, axis=1))
    ax[1,1].set_yscale('log')

    for axis in (ax[0,0], ax[0,1], ax[1,0]):
        axis.set_xlabel('time')
        axis.set_ylabel('z')
    ax[1,1].set_xlabel('time')
    ax[1,1].set_ylabel('mean relative error')

    fig.savefig(str(output_dir)+'/profile_comp_{:s}.png'.format(key), bbox_inches='tight')
    plt.close()
 
def scalar_comparison(scalars, key, output_dir):
    data = [i.data[key] for i in scalars]
    N    = np.min(np.array([i.shape[0] for i in data]))
    data = [i[:N] for i in data]
    ts   = [i.times[:N] for i in scalars]
    fig, ax = plt.subplots(2,1,figsize=(10,10), dpi=150)

    #Plot lines vs each other and relative error of the two lines
    ax[0].plot(ts[0],data[0],label='run1')
    ax[0].plot(ts[1],data[1],label='run2')
    ax[1].plot(ts[0],np.abs((data[0]-data[1])/data[1]))
    ax[1].set_yscale('log')
    ax[0].set_xlabel('time')
    ax[1].set_xlabel('time')
    ax[0].set_ylabel('Magnitude')
    ax[1].set_ylabel('Relative Error')

    fig.savefig(str(output_dir)+'/scalar_comp_{:s}.png'.format(key), bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    from docopt import docopt
    import pathlib
    from mpi4py import MPI
    import analysis
    comm = MPI.COMM_WORLD
    rank = comm.rank
    args = docopt(__doc__)

    #Get data and setup output directory
    plot_dir = pathlib.Path(args['--plot_dir']).absolute()
    if rank == 0 and not plot_dir.exists():
        plot_dir.mkdir()

    profile_paths = [   pathlib.Path('{:s}/profiles/'.format(args['--outdir1'])),\
                        pathlib.Path('{:s}/profiles/'.format(args['--outdir2'])) ] 
    scalar_paths  = [   pathlib.Path('{:s}/scalar/'.format(args['--outdir1'])),\
                        pathlib.Path('{:s}/scalar/'.format(args['--outdir2']))   ]

    profiles = [    analysis.Profile([str(i) for i in list(profile_paths[0].glob('*.h5'))]),\
                    analysis.Profile([str(i) for i in list(profile_paths[1].glob('*.h5'))]) ]
    scalar   = [    analysis.Scalar([str(i) for i in list(scalar_paths[0].glob('*.h5'))]),\
                    analysis.Scalar([str(i) for i in list(scalar_paths[1].glob('*.h5'))])   ]

    profiles_keys_comp = ['T1', 'ln_rho1', 'w_rms', 'grad_s_fluc', 'kappa_flux_fluc_z', \
                            'enthalpy_flux_z', 'KE_flux_z', 'enstrophy']
    scalar_keys_comp   = ['PE', 'IE', 'TE', 'PE_fluc', 'IE_fluc', 'TE_fluc', \
                            'KE', 'u_rms', 'w_rms', 'Re_rms', 'Pe_rms', 'enstrophy']

    for key in profiles_keys_comp:
        print('plotting comparison profile {:s}'.format(key))
        profile_comparison(profiles, key, plot_dir)
        
    for key in scalar_keys_comp:
        print('plotting comparison scalar {:s}'.format(key))
        scalar_comparison(scalar, key, plot_dir)
