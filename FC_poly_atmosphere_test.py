"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    
    --restart=<restart_file>             Restart from checkpoint

    --nz=<nz>                            vertical z (chebyshev) resolution 
    --nz_cz=<nz>                         vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3.5]

    --MHD                                Do MHD run
    --MagneticPrandtl=<MagneticPrandtl>  Magnetic Prandtl Number = nu/eta [default: 1]

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom)
    --fixed_Tz                           Fixed Temperature gradient boundary conditions (top and bottom)
        
    --label=<label>                      Additional label for run output directory

"""
import logging
logger = logging.getLogger(__name__)

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing = True
except:
    do_checkpointing = False
import matplotlib.pyplot as plt
    
def FC_constant_kappa(Rayleigh=1e6, Prandtl=1, MagneticPrandtl=1, MHD=False, n_rho_cz=3.5,
                      fixed_T=False, fixed_Tz=False, 
                      restart=None, nz=128, nx=None, data_dir='./'):
    import numpy as np
    import time
    import equations
    import os
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    nx = 2
    
    if MHD:
        atmosphere = equations.FC_MHD_polytrope(nx=nx, nz=nz, constant_kappa=True, n_rho_cz=n_rho_cz)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, MagneticPrandtl, include_background_flux=True)
    else:
        atmosphere = equations.FC_polytrope(nx=nx, nz=nz, constant_kappa=True, n_rho_cz=n_rho_cz)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, include_background_flux=True)
    if fixed_T:
        atmosphere.set_BC(fixed_temperature=fixed_T)
    elif fixed_Tz:
        atmosphere.set_BC(fixed_flux=fixed_Tz)
    else:
        atmosphere.set_BC()

    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    ts = de.timesteppers.RK443
    cfl_safety_factor = 0.2*4

    # Build solver
    solver = problem.build_solver(ts)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
    fig.savefig("sparsity_pattern.png", dpi=1200)

    atmosphere.set_IC(solver)
    dt = 1e-6
    solver.step(dt)
    
    import scipy.sparse.linalg as sla
    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='COLAMD')
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
    ax = fig.add_subplot(1,2,2)
    ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
    fig.savefig("sparsity_pattern_LU.png", dpi=1200)
    logger.info("{} nonzero entries in LU".format(LU.nnz))
    logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
    logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))

    
if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    
    import sys
    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    if args['--fixed_T']:
        data_dir +='_fixed'
    if args['--fixed_Tz']:
        data_dir +='_flux'
    data_dir += "_nrhocz{}_Ra{}".format(args['--n_rho_cz'], args['--Rayleigh'])
    if args['--MHD']:
        data_dir+= '_MHD'
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))
    
    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)
    nz = args['--nz']
    if nz is None:
        nz = args['--nz_cz']
    if nz is not None:
        nz = int(nz)

    FC_constant_kappa(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      MagneticPrandtl=float(args['--MagneticPrandtl']),
                      nz=nz,
                      nx=nx,
                      fixed_T=args['--fixed_T'],
                      fixed_Tz=args['--fixed_Tz'],                      
                      MHD=args['--MHD'],
                      restart=(args['--restart']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      data_dir=data_dir)
