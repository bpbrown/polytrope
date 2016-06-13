"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>               Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>                 Prandtl number = nu/kappa [default: 1]
    --root_dir=<root_dir>               Root directory to save data dir in [default: ./]
    --restart=<restart_file>            Restart from checkpoint
    --nx=<nx>                           horizontal x (fourier) resolution
    --nz=<nz>                           vertical z (chebyshev) resolution [default: 128]
    --n_rho_cz=<n_rho_cz>               Density scale heights across unstable layer [default: 3.5]
    --epsilon=<epsilon>                 The level of superadiabaticity of our polytrope background [default: 1e-4]
    --run_time=<run_time>               Run time in hours [default: 12]
    --label=<label>                     Additional label for run output directory
    --safety_factor=<safety_factor>     Determines CFL Danger.  Higher=Faster [default: 0.14]
    --start_new_files=<start_new_files> Start new files while checkpointing [default: False]
    --start_dt=<start_dt>               Start timestep, if None set it manually [default: None]
    --out_cadence=<out_cadence>         The fraction of a buoyancy time to output data at
                                                            [default: 0.1]
    --zero_velocities=<zero_vels>       If True, set all velocities to zero [default: False]
    --bootstrap_file=<bootstrap_file>   A file to bootstrap from
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
from tools.bootstrapping import Bootstrapper

checkpointing_installed=True
CHECKPOINT_MIN=30*60 #30 min checkpoints
if checkpointing_installed:
    from dedalus.extras.checkpointing import Checkpoint


def FC_constant_kappa(Rayleigh=1e6, Prandtl=1, n_rho_cz=3.5, epsilon=1e-4, restart=None, run_time=12, \
                            nz=128, nx=False, data_dir='./', safety_factor=0.2, grid_dtype=np.float64,
                            start_new_files = False, start_dt=None, out_cadence=0.1, zero_velocities=False,
                            bootstrap_file=None):
    import time
    import equations
    import os
    import sys
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx == None:
        nx = nz*4
    
    atmosphere = equations.FC_polytrope(nx=nx, nz=nz, constant_kappa=True, constant_mu=True,\
                                        epsilon=epsilon, n_rho_cz=n_rho_cz,\
                                        grid_dtype=grid_dtype, fig_dir='./FC_poly_atmosphere/')
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, include_background_flux=True)
    atmosphere.set_BC(T1_z_left = 0, T1_right=0, mixed_flux_temperature=True, stress_free=True)
    problem = atmosphere.get_problem()
    
    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    ts = de.timesteppers.RK222
    cfl_safety_factor = safety_factor*2

    ts = de.timesteppers.RK443
    cfl_safety_factor = safety_factor*4

    # Build solver
    solver = problem.build_solver(ts)

    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,\
                                                                    atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    
    atmosphere.check_atmosphere(make_plots = False, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))
    max_dt = atmosphere.buoyancy_time*0.25
  
    if checkpointing_installed:
        logger.info('checkpointing in {}'.format(data_dir))
        do_checkpointing=True
        checkpoint = Checkpoint(data_dir, allowed_dirs=['slices', 'profiles', 'scalar', 'coeffs'])
        chk_write = chk_set = 1
        if restart is None:
            atmosphere.set_IC(solver)
            dt = max_dt
            slices_count    = 1
            slices_set      = 1
            profiles_count  = 1
            profiles_set    = 1
            scalar_count    = 1
            scalar_set      = 1 
            coeffs_count, coeffs_set = 1,1
        else:
            logger.info("restarting from {}".format(restart))
            chk_write, chk_set, dt = checkpoint.restart(restart, solver)
            if not start_new_files:
                counts, sets = checkpoint.find_output_counts()
                slices_count    = counts['slices']
                slices_set      = sets['slices']
                profiles_count  = counts['profiles']
                profiles_set    = sets['profiles']
                scalar_count    = counts['scalar']
                scalar_set      = sets['scalar']
                try:
                    coeffs_count, coeffs_set = counts['coeffs'], sets['coeffs']
                except:
                    coeffs_count, coeffs_set = 1, 1
            else:
                slices_count    = 1
                slices_set      = 1
                profiles_count  = 1
                profiles_set    = 1
                scalar_count    = 1
                scalar_set      = 1
                coeffs_count, coeffs_set = 1, 1
                chk_write = chk_set = 1
        checkpoint.set_checkpoint(solver, wall_dt = CHECKPOINT_MIN, write_num=chk_write, set_num=chk_set)
    else:
        dt = max_dt
        slices_count    = 1
        slices_set      = 1
        profiles_count  = 1
        profiles_set    = 1
        scalar_count    = 1
        scalar_set      = 1 
        coeffs_count, coeffs_set = 1,1
        chk_write = chk_set = 1


    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    threshold=0.1
        
    if epsilon > 0.5:
        solver.stop_sim_time = 10*atmosphere.thermal_time
    else:
        solver.stop_sim_time = atmosphere.thermal_time
    if 500*atmosphere.buoyancy_time >  atmosphere.thermal_time:
        solver.stop_sim_time = 500*atmosphere.buoyancy_time
    solver.stop_iteration= np.inf
    solver.stop_wall_time = run_time*3600
   
    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, full_output=True,\
                                slices=[slices_count, slices_set], profiles=[profiles_count, profiles_set], scalar=[scalar_count, scalar_set],\
                                coeffs=[coeffs_count, coeffs_set])
 
    if bootstrap_file != None:
        booter = Bootstrapper(bootstrap_file, ['ln_rho1'])
        booter.bootstrap(atmosphere.z, solver)
        dt = atmosphere.buoyancy_time/200
        max_dt = dt
        noise = atmosphere.global_noise()
        T_IC = solver.state['T1']
        T_IC.set_scales(atmosphere.domain.dealias, keep_data=True)
        atmosphere.T0.set_scales(atmosphere.domain.dealias, keep_data=True)
        z_dealias = atmosphere.domain.grid(axis=1, scales=atmosphere.domain.dealias)
        T_IC['g'] = 1e-6*np.sin(np.pi*z_dealias/atmosphere.Lz)*noise*(atmosphere.T0['g'])#+atmosphere.T_IC['g'])
   
    if start_dt != None:
        dt = start_dt
    cfl_cadence = 1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=threshold)

    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    flow.add_property("Pe_rms", name='Pe')

    logger.info("Thermal time: {}, max_dt: {}".format(atmosphere.thermal_time, max_dt))
    logger.info("output cadence = {:g}".format(output_time_cadence))
    
    if zero_velocities:
        u = solver.state['u']
        w = solver.state['w']
        u['g'] *= 0
        w['g'] *= 0

    start_iter=solver.iteration
    try:
        start_time = time.time()
        while solver.ok:

            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            # update lists
            if solver.iteration % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, solver.sim_time/atmosphere.buoyancy_time, dt)
                log_string += '\n\t\tRe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Re'), flow.max('Re'))
                log_string += '; Pe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
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
        if checkpointing_installed:
            if do_checkpointing:
                logger.info(data_dir+'/checkpoint/')
                post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks.keys():
            logger.info(analysis_tasks[task].base_path)
            post.merge_analysis(analysis_tasks[task].base_path)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            if start_iter > 2:
                logger.info('Iterations (this run): {:d}'.format(N_iterations - start_iter))
                logger.info('Iterations (total): {:d}'.format(N_iterations - start_iter))
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
    data_dir = args['--root_dir']
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]
    data_dir += "_nrhocz{}_Ra{}_Pr{}_eps{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--Prandtl'], args['--epsilon'])
    if args['--label'] == None:
        data_dir += '/'
    else:
        data_dir += '_{}/'.format(args['--label'])
    logger.info("saving run in: {}".format(data_dir))
    
    if args['--start_new_files'] == 'True':
        start_new_files = True
    else:
        start_new_files = False
    if args['--start_dt'] != 'None':
        start_dt = float(args['--start_dt'])
    else:
        start_dt = None

    if args['--zero_velocities'] == 'True':
        zero_velocities=True
    else:
        zero_velocities=False

    nx = args['--nx']
    if nx != None:
        nx = int(nx)
    nz = int(args['--nz'])


    FC_constant_kappa(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      nx = nx,
                      nz = nz,
                      restart=(args['--restart']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      epsilon=float(args['--epsilon']),
                      run_time=float(args['--run_time']),
                      safety_factor=float(args['--safety_factor']),
                      data_dir=data_dir,
                      start_new_files=start_new_files,
                      start_dt = start_dt,
                      out_cadence=float(args['--out_cadence']),
                      zero_velocities=zero_velocities,
                      bootstrap_file=args['--bootstrap_file'])
