"""
Dedalus script for 2D compressible convection in a multitrope,
here with convection below and stable region above.


Usage:
    FC_multi.py [options]

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --stiffness=<stiffness>    Stiffness of radiative/convective interface [default: 1e4]
    --restart=<restart_file>   Restart from checkpoint
    --nz_rz=<nz_rz>            Vertical z (chebyshev) resolution in stable region   [default: 128]
    --nz_cz=<nz_cz>            Vertical z (chebyshev) resolution in unstable region [default: 128]
    --n_rho_cz=<n_rho_cz>      Density scale heights across unstable layer [default: 1]
    --n_rho_rz=<n_rho_rz>      Density scale heights across stable layer   [default: 4.5]
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

def FC_constant_kappa(Rayleigh=1e6, Prandtl=1, stiffness=1e4, 
                      n_rho_cz=1.5, n_rho_rz=3, 
                      nz_cz=128, nz_rz=128,
                      restart=None, data_dir='./', verbose=False):
    import numpy as np
    import time
    import equations
    import os
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))
    
    # Set domain
    nz = nz_rz+nz_cz
    nx = nz_cz*4
    nz_list = [nz_rz, nz_cz]
    
    atmosphere = equations.FC_multitrope(nx=nx, nz=nz_list, stiffness=stiffness, 
                                         n_rho_cz=n_rho_cz, n_rho_rz=n_rho_rz, 
                                         stable_top=True, verbose=verbose)
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, include_background_flux=False)
    atmosphere.set_BC(mixed_temperature_flux=True)
    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    ts = de.timesteppers.RK443
    cfl_safety_factor = 0.2*4

    # Build solver
    solver = problem.build_solver(ts)


    if do_checkpointing:
        checkpoint = Checkpoint(data_dir)
        checkpoint.set_checkpoint(solver, wall_dt=1800)

    x = atmosphere.domain.grid(0)
    z = atmosphere.domain.grid(1)


    # initial conditions
    T = solver.state['T1']
    ln_rho = solver.state['ln_rho1']

    if restart is None:
         # Random perturbations, initialized globally for same results in parallel
        gshape = atmosphere.domain.dist.grid_layout.global_shape(scales=atmosphere.domain.dealias)
        slices = atmosphere.domain.dist.grid_layout.slices(scales=atmosphere.domain.dealias)
        rand = np.random.RandomState(seed=42)
        noise = rand.standard_normal(gshape)[slices]

        A0 = 1e-6
        
        T.set_scales(atmosphere.domain.dealias, keep_data=True)
        z_dealias = atmosphere.domain.grid(axis=1, scales=atmosphere.domain.dealias)
        T['g'] = A0*np.sin(np.pi*z_dealias/atmosphere.Lz)*noise*atmosphere.T0['g']

        logger.info("A0 = {:g}".format(A0))
        #logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))
        
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
                         max_change=1.5, min_change=0.5, max_dt=max_dt)

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
        if do_checkpointing:
            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks:
            logger.info(task.base_path)
            post.merge_analysis(task.base_path)

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
    data_dir += "_Ra{}_S{}".format(args['--Rayleigh'], args['--stiffness'])
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'

    logger.info("saving run in: {}".format(data_dir))
    
    FC_constant_kappa(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      stiffness=float(args['--stiffness']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      n_rho_rz=float(args['--n_rho_rz']),
                      nz_rz=int(args['--nz_rz']),
                      nz_cz=int(args['--nz_cz']),
                      restart=(args['--restart']),
                      data_dir=data_dir,
                      verbose=args['--verbose'])
