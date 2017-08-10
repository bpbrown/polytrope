"""
Dedalus script for finding the onset of compressible convection in a 
polytropic atmosphere

Usage:
    FC_poly_onset_curve.py [options] 

Options:
    --rayleigh_start=<Rayleigh>         Rayleigh number start [default: 1e-2]
    --rayleigh_stop=<Rayleigh>          Rayleigh number stop [default: 1e3]
    --rayleigh_steps=<steps>            Integer number of steps between start 
                                         and stop Ra   [default: 20]
    --kx_start=<kx_start>               kx to start at [default: 0.1]
    --kx_stop=<kx_stop>                 kx to stop at  [default: 1]
    --kx_steps=<kx_steps>               Num steps in kx space [default: 20]

    --ky_start=<ky_start>               kx to start at [default: 0.1]
    --ky_stop=<ky_stop>                 kx to stop at  [default: 1]
    --ky_steps=<ky_steps>               Num steps in kx space [default: 20]

    --nz=<nz>                           z (chebyshev) resolution [default: 48]

    --n_rho_cz_start=<n_rho_cz_start>   Density scale heights (smallest run)
    --n_rho_cz_stop=<n_rho_cz_stop>     Density scale heights (largest run) 
    --n_rho_cz_steps=<n_rho_cz_steps>   Num steps in n_rho space
    --epsilon_start=<n_rho_cz_start>    Epsilon (smallest run) 
    --epsilon_stop=<n_rho_cz_stop>      Epsilon (largest run)  
    --epsilon_steps=<n_rho_cz_steps>    Number of epsilon steps to take 
    --epsilons=<epsilon>                A comma-delimited list of epsilons

    --bcs=<bcs>                         Boundary conditions ('fixed', 'mixed', 
                                            or 'flux') [default: fixed]
    --n_rho_cz_default=<n_rho>          Default nrho [default: 3]
    --epsilon_default=<epsilon>         Default eps  [default: 0.5]
    --gamma_default=<gamma>             Default gamma [default: 5/3]
    --constant_chi                      If true, use const chi
    --constant_nu                       If true, use const nu
    --Taylor_default=<Ta>               If not None, solve for rotating convection

    --3D                                If flagged, use 3D eqns and search kx & ky
    --2.5D                              If flagged, use 3D eqns with ky = 0

    --load                              If flagged, attempt to load data from output file
    --out_dir=<out_dir>                 Base output dir [default: ./]
"""
from docopt import docopt
from onset_solver import OnsetSolver
import numpy as np

args = docopt(__doc__)

file_name = 'FC_poly_onsets'

##########################################
#Set up ra / kx / ky spans
ra_log, kx_log, ky_log = False, False, False

ra_start = float(args['--rayleigh_start'])
ra_stop = float(args['--rayleigh_stop'])
ra_steps = float(args['--rayleigh_steps'])
if np.abs(ra_stop/ra_start) > 10:
    ra_log = True

kx_start = float(args['--kx_start'])
kx_stop = float(args['--kx_stop'])
kx_steps = float(args['--kx_steps'])
if np.abs(kx_stop/kx_start) > 10:
    kx_log = True

ky_start = float(args['--ky_start'])
ky_stop = float(args['--ky_stop'])
ky_steps = float(args['--ky_steps'])
if np.abs(ky_stop/ky_start) > 10:
    ky_log = True

##########################################
#Set up defaults for the atmosphere
const_kap = True
if args['--constant_chi']:
    const_kap = False
const_mu = True
if args['--constant_nu']:
    const_mu = False

nz = int(args['--nz'])
n_rho_default = float(args['--n_rho_cz_default'])
epsilon_default = float(args['--epsilon_default'])
file_name += '_eps{}'.format(args['--epsilon_default'])
file_name += '_nrho{}'.format(args['--n_rho_cz_default'])
try:
    gamma_default = float(args['--gamma_default'])
except:
    from fractions import Fraction
    gamma_default = float(Fraction(args['--gamma_default']))


defaults = {'n_rho_cz': n_rho_default,
            'epsilon':  epsilon_default,
            'constant_kappa': const_kap,
            'constant_mu':    const_mu,
            'nz':             nz,
            'gamma':          gamma_default}

##############################################
#Setup default arguments for equation building
taylor = args['--Taylor_default']
if taylor != None:
    file_name += '_Ta{}'.format(taylor)
    taylor = float(taylor)
eqn_defaults_args = [1] #prandtl number
eqn_defaults_kwargs = {'Taylor': taylor}

############################################
#Set up BCs
bcs = args['--bcs']
fixed_flux = False
fixed_T    = False
mixed_T    = False
if bcs == 'mixed':
    mixed_T = True
    file_name += '_mixedBC'
elif bcs == 'fixed':
    fixed_T = True
    file_name += '_fixedTBC'
else:
    fixed_flux = True
    file_name += '_fluxBC'

bc_defaults = {'fixed_temperature': fixed_T,
               'mixed_flux_temperature': mixed_T,
               'fixed_flux': fixed_flux,
               'stress_free': True}
file_name += '_bc_{:s}'.format(bcs)

#####################################################
#Initialize onset solver
if args['--3D']:
    solver = OnsetSolver(
                eqn_set=0, atmosphere=0, threeD=True,
                ra_steps=(ra_start, ra_stop, ra_steps, ra_log),
                kx_steps=(kx_start, kx_stop, kx_steps, kx_log),
                ky_steps=(ky_start, ky_stop, ky_steps, ky_log),
                atmo_kwargs=defaults,
                eqn_args=eqn_defaults_args,
                eqn_kwargs=eqn_defaults_kwargs,
                bc_kwargs=bc_defaults)
elif args['--2.5D']:
    solver = OnsetSolver(
                eqn_set=0, atmosphere=0, threeD=True,
                ra_steps=(ra_start, ra_stop, ra_steps, ra_log),
                kx_steps=(kx_start, kx_stop, kx_steps, kx_log),
                atmo_kwargs=defaults,
                eqn_args=eqn_defaults_args,
                eqn_kwargs=eqn_defaults_kwargs,
                bc_kwargs=bc_defaults)
else:
    solver = OnsetSolver(
                eqn_set=0, atmosphere=0,
                ra_steps=(ra_start, ra_stop, ra_steps, ra_log),
                kx_steps=(kx_start, kx_stop, kx_steps, kx_log),
                atmo_kwargs=defaults,
                eqn_args=eqn_defaults_args,
                eqn_kwargs=eqn_defaults_kwargs,
                bc_kwargs=bc_defaults)


###############################################
#Add tasks (nrho, epsilon) in case we need to solve over
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

if args['--epsilon_start'] != None\
        and args['--epsilon_stop'] != None \
        and args['--epsilon_steps'] != None:
    start = float(args['--epsilon_start'])
    stop  = float(args['--epsilon_stop'])
    steps = int(args['--epsilon_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('epsilon', start, stop, n_steps=steps, log=log)
    file_name += '_eps{:s}-{:s}-{:s}'.format(
                    args['--epsilon_start'],
                    args['--epsilon_stop'],
                    args['--epsilon_steps'])

#############################################
#Crit find!
out_dir = args['--out_dir']
load    = args['--load']
solver.find_crits(out_dir=out_dir, out_file='{:s}'.format(file_name), load=load)
