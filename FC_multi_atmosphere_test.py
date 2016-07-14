"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    FC_multi_atmosphere_test.py [options]

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --stiffness=<stiffness>    Stiffness of radiative/convective interface [default: 1e4]
    --restart=<restart_file>   Restart from checkpoint
    --nz_rz=<nz_rz>            Vertical z (chebyshev) resolution in stable region   [default: 128]
    --nz_cz=<nz_cz>            Vertical z (chebyshev) resolution in unstable region [default: 128]
    --single_chebyshev         Use a single chebyshev domain across both stable and unstable regions.  Useful at low stiffness.
    --nx=<nx>                  Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>      Density scale heights across unstable layer [default: 3.5]
    --n_rho_rz=<n_rho_rz>      Density scale heights across stable layer   [default: 1]

    --width=<width>            Width of erf transition between two polytropes
    
    --MHD                                Do MHD run
    --MagneticPrandtl=<MagneticPrandtl>  Magnetic Prandtl Number = nu/eta [default: 1]

    
    --label=<label>            Additional label for run output directory
    --verbose                  Produce diagnostic plots
"""
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing=True
except:
    logger.info("No checkpointing available; disabling capability")
    do_checkpointing=False


def FC_constant_kappa(Rayleigh=1e6, Prandtl=1, stiffness=1e4,
                      MagneticPrandtl=1, MHD=False, 
                      n_rho_cz=3.5, n_rho_rz=1, 
                      nz_cz=128, nz_rz=128,
                      nx = None,
                      width=None,
                      single_chebyshev=False,
                      restart=None, data_dir='./', verbose=False):
    import numpy as np
    import time
    import equations
    import os
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))
    
    nx=2
    
    if single_chebyshev:
        nz = nz_cz
        nz_list = [nz_cz]
    else:
        nz = nz_rz+nz_cz
        nz_list = [nz_rz, nz_cz]

    if MHD:
        atmosphere = equations.FC_MHD_multitrope(nx=nx, nz=nz_list, stiffness=stiffness, 
                                                n_rho_cz=n_rho_cz, n_rho_rz=n_rho_rz, 
                                                verbose=verbose)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, MagneticPrandtl)
    else:
        atmosphere = equations.FC_multitrope(nx=nx, nz=nz_list, stiffness=stiffness, 
                                         n_rho_cz=n_rho_cz, n_rho_rz=n_rho_rz, 
                                         verbose=verbose, width=width)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=1e-8)
        
    atmosphere.set_BC()
    problem = atmosphere.get_problem()

    atmosphere.plot_atmosphere()
    #atmosphere.plot_scaled_atmosphere()

    ts = de.timesteppers.RK443
    # Build solver (to check NCCs)
    solver = problem.build_solver(ts)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
    fig.savefig("sparsity_pattern.png", dpi=600)

    atmosphere.set_IC(solver)
    dt = 1e-6
    solver.step(dt)
    
    import scipy.sparse.linalg as sla
    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='MMD_ATA')
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
    data_dir += "_nrhocz{}_Ra{}_S{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--stiffness'])
    if args['--width'] is not None:
        data_dir += "_erf{}".format(args['--width'])
        width = float(args['--width'])
    else:
        width = None
    if args['--MHD']:
        data_dir+= '_MHD'
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))

    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)
        
    FC_constant_kappa(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      stiffness=float(args['--stiffness']),
                      MHD=args['--MHD'],
                      MagneticPrandtl=float(args['--MagneticPrandtl']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      n_rho_rz=float(args['--n_rho_rz']),
                      nz_rz=int(args['--nz_rz']),
                      nz_cz=int(args['--nz_cz']),
                      single_chebyshev=args['--single_chebyshev'],
                      width=width,
                      nx=nx,
                      restart=(args['--restart']),
                      data_dir=data_dir,
                      verbose=args['--verbose'])
