"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.
Adds in passive scalar tracer and reacting scalar.

Usage:
    FC_multi_rxn.py [options]

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --ChemicalPrandtl=<ChemicalPrandtl>       ChemicalPrandtl number = nu/nu_chem [default: 1]
    --ChemicalReynolds=<ChemicalReynolds>     ChemicalReynolds number; controls k_chem [default: 10]
    --stiffness=<stiffness>    Stiffness of radiative/convective interface [default: 1e4]
    --restart=<restart_file>   Restart from checkpoint
    --nz_rz=<nz_rz>            Vertical z (chebyshev) resolution in stable region   [default: 128]
    --nz_cz=<nz_cz>            Vertical z (chebyshev) resolution in unstable region [default: 128]
    --single_chebyshev         Use a single chebyshev domain across both stable and unstable regions.  Useful at low stiffness.
    --nx=<nx>                  Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>      Density scale heights across unstable layer [default: 3.5]
    --n_rho_rz=<n_rho_rz>      Density scale heights across stable layer   [default: 1]

    --rk222                    Use RK222 as timestepper

    --superstep                Superstep equations by using average rather than actual vertical grid spacing
    --dense                    Oversample matching region with extra chebyshev domain
    --nz_dense=<nz_dense>      Vertical z (chebyshev) resolution in oversampling region   [default: 64]
   
    --oz                       Do system with convection zone on the bottom rather than top (exoplanets)

    --width=<width>            Width of erf transition between two polytropes
    
    --label=<label>            Additional label for run output directory
    --verbose                  Produce diagnostic plots
"""
import logging
logger = logging.getLogger(__name__)

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing=True
except:
    logger.info("No checkpointing available; disabling capability")
    do_checkpointing=False

def FC_convection(Rayleigh=1e6, Prandtl=1,
                  ChemicalPrandtl=1, ChemicalReynolds=10, stiffness=1e4,
                      n_rho_cz=3.5, n_rho_rz=1, 
                      nz_cz=128, nz_rz=128,
                      nx = None,
                      width=None,
                      single_chebyshev=False,
                      rk222=False,
                      superstep=False,
                      dense=False, nz_dense=64,
                      oz=False,
                      restart=None, data_dir='./', verbose=False):
    import numpy as np
    import time
    import equations
    import os
    from dedalus.core.future import FutureField
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if oz:
        constant_Prandtl=False
        stable_top=True
        mixed_temperature_flux=True
    else:
        constant_Prandtl=True
        stable_top=False
        mixed_temperature_flux=None
        
    # Set domain
    if nx is None:
        nx = nz_cz*4
        
    if single_chebyshev:
        nz = nz_cz
        nz_list = [nz_cz]
    else:
        if dense:
            nz = nz_rz+nz_dense+nz_cz
            #nz_list = [nz_rz, int(nz_dense/2), int(nz_dense/2), nz_cz]
            nz_list = [nz_rz, nz_dense, nz_cz]
        else:
            nz = nz_rz+nz_cz
            nz_list = [nz_rz, nz_cz]

    atmosphere = equations.FC_multitrope_rxn(nx=nx, nz=nz_list, stiffness=stiffness, 
                                         n_rho_cz=n_rho_cz, n_rho_rz=n_rho_rz, 
                                         verbose=verbose, width=width,
                                         constant_Prandtl=constant_Prandtl,
                                         stable_top=stable_top)
    
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, ChemicalPrandtl, ChemicalReynolds)
        
    atmosphere.set_BC(mixed_temperature_flux=mixed_temperature_flux)
    problem = atmosphere.get_problem()

    #atmosphere.plot_atmosphere()
    #atmosphere.plot_scaled_atmosphere()

        
    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk222:
        logger.info("timestepping using RK222")
        ts = de.timesteppers.RK222
        cfl_safety_factor = 0.2*2
    else:
        logger.info("timestepping using RK443")
        ts = de.timesteppers.RK443
        cfl_safety_factor = 0.2*4

    # Build solver
    solver = problem.build_solver(ts)

    if do_checkpointing:
        checkpoint = Checkpoint(data_dir)
        checkpoint.set_checkpoint(solver, wall_dt=1800)

    # initial conditions
    if restart is None:
        atmosphere.set_IC(solver)
    else:
        if do_checkpointing:
            logger.info("restarting from {}".format(restart))
            checkpoint.restart(restart, solver)
        else:
            logger.error("No checkpointing capability in this branch of Dedalus.  Aborting.")
            raise

    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time, atmosphere.top_thermal_time))
    
    max_dt = atmosphere.min_BV_time 
    max_dt = atmosphere.buoyancy_time*0.25

    report_cadence = 1
    output_time_cadence = 0.1*atmosphere.buoyancy_time
    solver.stop_sim_time = np.inf
    solver.stop_iteration= np.inf
    solver.stop_wall_time = 23.5*3600

    logger.info("output cadence = {:g}".format(output_time_cadence))

    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

    
    cfl_cadence = 1
    CFL = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

    if superstep:
        CFL_traditional = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

        CFL_traditional.add_velocities(('u', 'w'))
    
        vel_u = FutureField.parse('u', CFL.solver.evaluator.vars, CFL.solver.domain)
        delta_x = atmosphere.Lx/nx
        CFL.add_frequency(vel_u/delta_x)
        vel_w = FutureField.parse('w', CFL.solver.evaluator.vars, CFL.solver.domain)
        mean_delta_z_cz = atmosphere.Lz_cz/nz_cz
        CFL.add_frequency(vel_w/mean_delta_z_cz)
    else:
        CFL.add_velocities(('u', 'w'))


    
    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')

    try:
        start_time = time.time()
        while solver.ok:

            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            # update lists
            if solver.iteration % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), '.format(solver.iteration, solver.sim_time, solver.sim_time/atmosphere.buoyancy_time)
                log_string += 'dt: {:8.3e}'.format(dt)
                if superstep:
                    dt_traditional = CFL_traditional.compute_dt()
                    log_string += ' (vs {:8.3e})'.format(dt_traditional)
                log_string += ', '
                log_string += 'Re: {:8.3e}/{:8.3e}'.format(flow.grid_average('Re'), flow.max('Re'))
                logger.info(log_string)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()

        # Print statistics
        elapsed_time = end_time - start_time
        elapsed_sim_time = solver.sim_time
        N_iterations = solver.iteration 
        logger.info('main loop time: {:e}'.format(elapsed_time))
        logger.info('Iterations: {:d}'.format(N_iterations))
        logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
        logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
        
        logger.info('beginning join operation')
        if do_checkpointing:
            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks:
            logger.info(analysis_tasks[task].base_path)
            post.merge_analysis(analysis_tasks[task].base_path)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            logger.info('Iterations: {:d}'.format(N_iterations))
            logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
            logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
 
            N_TOTAL_CPU = atmosphere.domain.distributor.comm_cart.size

            # Print statistics
            print('-' * 40)
            total_time = end_time-initial_time
            main_loop_time = end_time - start_time
            startup_time = start_time-initial_time
            n_steps = solver.iteration-1
            print('  startup time:', startup_time)
            print('main loop time:', main_loop_time)
            print('    total time:', total_time)
            print('Iterations:', solver.iteration)
            print('Average timestep:', solver.sim_time / n_steps)
            print("          N_cores, Nx, Nz, startup     main loop,   main loop/iter, main loop/iter/grid, n_cores*main loop/iter/grid")
            print('scaling:',
                  ' {:d} {:d} {:d}'.format(N_TOTAL_CPU,nx,nz),
                  ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                                    main_loop_time, 
                                                                    main_loop_time/n_steps, 
                                                                    main_loop_time/n_steps/(nx*nz), 
                                                                    N_TOTAL_CPU*main_loop_time/n_steps/(nx*nz)))
            print('-' * 40)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    
    import sys
    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    if args['--oz']:
        data_dir += '_oz'
    data_dir += "_nrhocz{}_Ra{}_S{}_ChRe{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--stiffness'], args['--ChemicalReynolds'])
    if args['--width'] is not None:
        data_dir += "_erf{}".format(args['--width'])
        width = float(args['--width'])
    else:
        width = None
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))

    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)
        
    FC_convection(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      ChemicalPrandtl=float(args['--ChemicalPrandtl']),
                      ChemicalReynolds=float(args['--ChemicalReynolds']),
                      stiffness=float(args['--stiffness']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      n_rho_rz=float(args['--n_rho_rz']),
                      nz_rz=int(args['--nz_rz']),
                      nz_cz=int(args['--nz_cz']),
                      single_chebyshev=args['--single_chebyshev'],
                      width=width,
                      nx=nx,
                      restart=(args['--restart']),
                      data_dir=data_dir,
                      verbose=args['--verbose'],
                      oz=args['--oz'],
                      dense=args['--dense'],
                      nz_dense=int(args['--nz_dense']),
                      rk222=args['--rk222'],
                      superstep=args['--superstep'])
