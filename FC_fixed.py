"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>               Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>                 Prandtl number = nu/kappa [default: 1]
    --restart=<restart_file>            Restart from checkpoint
    --nz=<nz>                           vertical z (chebyshev) resolution 
    --nz_cz=<nz>                        vertical z (chebyshev) resolution
    --nx=<nx>                           Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>               Density scale heights across unstable layer [default: 3.5]
    --label=<label>                     Additional label for run output directory

"""
import logging
logger = logging.getLogger(__name__)

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
from dedalus.extras.checkpointing import Checkpoint

def FC_constant_kappa(Rayleigh=1e6, Prandtl=1, n_rho_cz=3.5, restart=None, nz=128, nx=None, data_dir='./'):
    import numpy as np
    import time
    import equations
    import os
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = nz*4
    
    atmosphere = equations.FC_polytrope(nx=nx, nz=nz, constant_kappa=True, n_rho_cz=n_rho_cz)
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, include_background_flux=True)
    atmosphere.set_BC(fixed_temperature=True)
    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    ts = de.timesteppers.RK443
    cfl_safety_factor = 0.2*4

    # Build solver
    solver = problem.build_solver(ts)

    max_dt = atmosphere.buoyancy_time*0.25
    checkpoint = Checkpoint(data_dir)
    if restart is None:
        atmosphere.set_IC(solver)        
        checkpoint_write = checkpoint_set = 1
        initial_dt = max_dt
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint_write, checkpoint_set, dt = checkpoint.restart(restart, solver)
        # these should be passed to init analysis
        #counts, sets = checkpoint.find_output_counts()
        logger.info("Restarting from time t = {0:10.5e}".format(solver.sim_time))
        checkpoint_write = checkpoint_set = 1
        initial_dt = dt

    checkpoint.set_checkpoint(solver,wall_dt=30*60.,write_num=checkpoint_write, set_num=checkpoint_set)

    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,
                                                                      atmosphere.top_thermal_time))
    #logger.info("full atm HS check")
    #atmosphere.check_atmosphere(make_plots = True, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))



    report_cadence = 1
    output_time_cadence = 0.1*atmosphere.buoyancy_time
    solver.stop_sim_time = atmosphere.thermal_time
    solver.stop_iteration= np.inf
    solver.stop_wall_time = 23.5*3600

    logger.info("output cadence = {:g}".format(output_time_cadence))

    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

    
    cfl_cadence = 1
    CFL = flow_tools.CFL(solver, initial_dt=initial_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

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
                log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, '.format(solver.iteration, solver.sim_time, dt)
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
            print('    iterations:', solver.iteration)
            print(' loop sec/iter:', main_loop_time/solver.iteration)
            print('    average dt:', solver.sim_time / n_steps)
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
    data_dir += "_nrhocz{}_Ra{}".format(args['--n_rho_cz'], args['--Rayleigh'])
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
                      nz=nz,
                      nx=nx, 
                      restart=(args['--restart']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      data_dir=data_dir)