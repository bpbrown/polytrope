"""
Dedalus script for finding the onset of compressible convection in a multitrope

Usage:
    FC_multi_onset_curve.py [options] 

Options:
    --rayleigh_start=<Rayleigh>         Rayleigh number start [default: 1e-2]
    --rayleigh_stop=<Rayleigh>          Rayleigh number stop [default: 1e3]
    --rayleigh_steps=<steps>            Integer number of steps between start 
                                         and stop Ra   [default: 20]
    --kx_start=<kx_start>               kx to start at [default: 0.1]
    --kx_stop=<kx_stop>                 kx to stop at  [default: 10]
    --kx_steps=<kx_steps>               Num steps in kx space [default: 20]

    --nz=<nz>                           z resolution (rz, cz) [default: 64,128]

    --n_rho_cz_start=<n_rho_cz_start>   Density scale heights (smallest run)
    --n_rho_cz_stop=<n_rho_cz_stop>     Density scale heights (largest run) 
    --n_rho_cz_steps=<n_rho_cz_steps>   Num steps in n_rho space
    --n_rho_rz_start=<n_rho_cz_start>   Density scale heights (smallest run)
    --n_rho_rz_stop=<n_rho_cz_stop>     Density scale heights (largest run) 
    --n_rho_rz_steps=<n_rho_cz_steps>   Num steps in n_rho space
    --stiffness_start=<n_rho_cz_start>  stiffness (smallest run) 
    --stiffness_stop=<n_rho_cz_stop>    stiffness (largest run)  
    --stiffness_steps=<n_rho_cz_steps>  Number of stiffness steps to take 

    --bcs=<bcs>                         Boundary conditions ('fixed', 'mixed', 
                                            or 'flux') [default: fixed]
    --n_rho_cz_default=<n_rho>          Default nrho_cz [default: 3.5]
    --n_rho_rz_default=<n_rho>          Default nrho_cz [default: 1]
    --stiffness_default=<stiffness>     Default stiffness  [default: 1e4]
    --non_constant_Prandtl              If true, don't use const Pr
    --width=<width>                     Width of tanh

    --out_dir=<out_dir>                 Base output dir [default: ./]
"""
from docopt import docopt
from onset_solver import OnsetSolver

args = docopt(__doc__)
file_name = 'FC_multi_onsets'

##########################################
#Set up ra / kx spans
ra_start = float(args['--rayleigh_start'])
ra_stop = float(args['--rayleigh_stop'])
ra_steps = float(args['--rayleigh_steps'])

kx_start = float(args['--kx_start'])
kx_stop = float(args['--kx_stop'])
kx_steps = float(args['--kx_steps'])

file_name += '_ra{:s}-{:s}-{:s}_kx{:s}-{:s}-{:s}'.format(
                args['--rayleigh_start'],
                args['--rayleigh_stop'],
                args['--rayleigh_steps'],
                args['--kx_start'],
                args['--kx_stop'],
                args['--kx_steps'])

############################################
#Set up defaults for the atmosphere
const_pr = True
if args['--non_constant_Prandtl']:
    const_pr = False

nz = args['--nz'].split(',')
nz = [int(n) for n in nz]
n_rho_cz_default = float(args['--n_rho_cz_default'])
n_rho_rz_default = float(args['--n_rho_rz_default'])
stiffness_default = float(args['--stiffness_default'])
width = args['--width']
if width != None:
    width = float(width)
    file_name += '_w{:s}'.format(args['--width'])


defaults = {'n_rho_cz': n_rho_cz_default,
            'n_rho_rz': n_rho_rz_default,
            'stiffness':  stiffness_default,
            'constant_Prandtl':    const_pr,
            'nz':             nz,
            'width':          width}


###################################################
#Setup default arguments for equation building
eqn_defaults_args = [1] #prandtl number
eqn_defaults_kwargs = {}

###################################################
#Set up BCs
bcs = args['--bcs']
fixed_flux = False
fixed_T    = False
mixed_T    = False
if bcs == 'mixed':
    mixed_T = True
elif bcs == 'fixed':
    fixed_T = True
else:
    fixed_flux = True

bc_defaults = {'fixed_temperature': fixed_T,
               'mixed_flux_temperature': mixed_T,
               'fixed_flux': fixed_flux,
               'stress_free': True}
file_name += '_bc_{:s}'.format(bcs)

#####################################################
#Initialize onset solver
solver = OnsetSolver(
            eqn_set=0, atmosphere=1,
            ra_steps=(ra_start, ra_stop, ra_steps, True),
            kx_steps=(kx_start, kx_stop, kx_steps, True),
            atmo_kwargs=defaults,
            eqn_args=eqn_defaults_args,
            eqn_kwargs=eqn_defaults_kwargs,
            bc_kwargs=bc_defaults)

#################################################
#Add tasks (nrho, stiffness) in case we need to solve over
#those parameter spaces, too
if args['--n_rho_cz_start'] != None\
        and args['--n_rho_cz_stop'] != None \
        and args['--n_rho_cz_steps'] != None:
    start = float(args['--n_rho_cz_start'])
    stop  = float(args['--n_rho_cz_stop'])
    steps = int(args['--n_rho_cz_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('n_rho_cz', start, stop, n_steps=steps, log=log)
    file_name += '_nrhocz{:s}-{:s}-{:s}'.format(
                    args['--n_rho_cz_start'],
                    args['--n_rho_cz_stop'],
                    args['--n_rho_cz_steps'])

if args['--n_rho_rz_start'] != None\
        and args['--n_rho_rz_stop'] != None \
        and args['--n_rho_rz_steps'] != None:
    start = float(args['--n_rho_rz_start'])
    stop  = float(args['--n_rho_rz_stop'])
    steps = int(args['--n_rho_rz_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('n_rho_rz', start, stop, n_steps=steps, log=log)
    file_name += '_nrhorz{:s}-{:s}-{:s}'.format(
                    args['--n_rho_rz_start'],
                    args['--n_rho_rz_stop'],
                    args['--n_rho_rz_steps'])


if args['--stiffness_start'] != None\
        and args['--stiffness_stop'] != None \
        and args['--stiffness_steps'] != None:
    start = float(args['--stiffness_start'])
    stop  = float(args['--stiffness_stop'])
    steps = int(args['--stiffness_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('stiffness', start, stop, n_steps=steps, log=log)
    file_name += '_stiff{:s}-{:s}-{:s}'.format(
                    args['--stiffness_start'],
                    args['--stiffness_stop'],
                    args['--stiffness_steps'])



#####################################################
#Crit find!
out_dir = args['--out_dir']
solver.find_crits(out_dir=out_dir, out_file='{:s}.h5'.format(file_name))

#plot
solver.plot_onset_curves(out_dir=out_dir, fig_name='{:s}.png'.format(file_name))
