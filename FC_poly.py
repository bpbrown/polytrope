"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3.5]
    --Lx=<Lx>                            Physical width of the atmosphere
    --aspect_ratio=<aspect_ratio>        Physical aspect ratio of the atmosphere [default: 4]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --run_time=<run_time>                Run time, in hours [default: 23.5]

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom)
    --fixed_Tz                           Fixed Temperature gradient boundary conditions (top and bottom)
    --const_mu                           If flagged, use constant mu 
    --const_chi                          If flagged, use constant chi 

    --restart=<restart_file>             Restart from checkpoint
    --start_new_files                    Start new files while checkpointing
    --start_dt=<start_dt>                Start timestep, if None set it manually
    --zero_velocities                    If True, set all velocities to zero

    --timestepper=<timestepper>          Runge-Kutta. 2nd or 4th order (rk222/rk443) [default: rk222]
    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]
    
    --root_dir=<root_dir>                Root directory to save data dir in [default: ./]
    --label=<label>                      Additional label for run output directory
    --out_cadence=<out_cadence>          The fraction of a buoyancy time to output data at [default: 0.1]
    --no_coeffs                          If flagged, coeffs will not be output
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing = True
    checkpoint_min   = 0.5
except:
    print('not importing checkpointing')
    do_checkpointing = False

def FC_constant_kappa(  Rayleigh=1e6, Prandtl=1, Lx=None, aspect_ratio=None,\
                        nz=128, nx=False, n_rho_cz=3.5, epsilon=1e-4, run_time=23.5, \
                        fixed_T=False, fixed_Tz=False, const_mu=False, const_kappa=True, \
                        restart=None, start_new_files=False, start_dt=None, zero_velocities=False, \
                        rk443=False, safety_factor=0.2, sim_time_buoyancies=None, \
                        data_dir='./', out_cadence=0.1, no_coeffs=False):
    import time
    import equations
    import os
    import sys
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx == None:
        nx = nz*4
    
    atmosphere = equations.FC_polytrope(nx=nx, nz=nz, constant_kappa=const_kappa, constant_mu=const_mu,\
                                        epsilon=epsilon, n_rho_cz=n_rho_cz, Lx=Lx, aspect_ratio=aspect_ratio,\
                                        fig_dir='./FC_poly_atmosphere/')
    atmosphere.set_IVP_problem(Rayleigh, Prandtl)

    if fixed_T:
        atmosphere.set_BC(T1_left=0, T1_right=0, fixed_temperature=True, stress_free=True)
    elif fixed_Tz:
        atmosphere.set_BC(T1_z_left=0, T1_z_right=0, fixed_flux=True, stress_free=True)
    else:
        atmosphere.set_BC(T1_z_left=0, T1_right=0, mixed_flux_temperature=True, stress_free=True)

    problem = atmosphere.get_problem()
    
    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk443:
        ts = de.timesteppers.RK443
        cfl_safety_factor = safety_factor*4
    else:
        ts = de.timesteppers.RK222
        cfl_safety_factor = safety_factor*2

    # Build solver
    solver = problem.build_solver(ts)

    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,\
                                                                    atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    
    atmosphere.check_atmosphere(make_plots = True, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))
    dt = max_dt = atmosphere.buoyancy_time*0.25

  
    if restart is None or start_new_files or not do_checkpointing:
        slices_count, slices_set        = 1,1
        profiles_count, profiles_set    = 1,1
        scalar_count, scalar_set        = 1,1
        coeffs_count, coeffs_set        = 1,1
        chk_write = chk_set = 1
    if do_checkpointing:
        logger.info('checkpointing in {}'.format(data_dir))

        #Find all of the directories we don't want to checkpoint in
        import glob
        good_dirs = ['slices', 'profiles', 'scalar', 'coeffs', 'checkpoint']
        dirs = glob.glob('{:s}/*/'.format(data_dir))
        found_dirs = [s_dir.split(data_dir)[-1].split('/')[0] for s_dir in dirs]
        excluded_dirs = []
        for found_dir in found_dirs:
            if found_dir not in good_dirs: excluded_dirs.append(found_dir)

        #Checkpoint
        try:
            checkpoint = Checkpoint(data_dir, excluded_dirs=excluded_dirs)
        except:
            checkpoint = Checkpoint(data_dir)
        if restart is None :
            atmosphere.set_IC(solver)
        else:
            logger.info("restarting from {}".format(restart))
            chk_write, chk_set, dt = checkpoint.restart(restart, solver)
            if not start_new_files:
                counts, sets = checkpoint.find_output_counts()
                #All of the +1s make it so that we make a new file rather than overwriting the previous.
                slices_count, slices_set            = counts['slices']+1,sets['slices']+1
                profiles_count, profiles_set      = counts['profiles']+1,sets['profiles']+1
                scalar_count, scalar_set            = counts['scalar']+1,sets['scalar']+1
                try: #Allows for runs without coeffs
                    coeffs_count, coeffs_set = counts['coeffs']+1, sets['coeffs']+1
                except:
                    coeffs_count, coeffs_set = 1, 1
                chk_write += 1
                chk_set   += 1
            else:
                chk_write = chk_set = 1
        checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, write_num=chk_write, set_num=chk_set)
    else:
        atmosphere.set_IC(solver)
    report_cadence = 1
    output_time_cadence     = out_cadence*atmosphere.buoyancy_time
    if sim_time_buoyancies != None:
        solver.stop_sim_time    = solver.sim_time + sim_time_buoyancies*atmosphere.buoyancy_time
    else:
        solver.stop_sim_time    = atmosphere.thermal_time*10
    solver.stop_iteration   = np.inf
    solver.stop_wall_time   = run_time*3600
        
    logger.info("output cadence = {:g}".format(output_time_cadence))
    
    if no_coeffs:
        coeffs_output=False
    else:
        coeffs_output=True
    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=coeffs_output,\
                                slices=[slices_count, slices_set], profiles=[profiles_count, profiles_set], scalar=[scalar_count, scalar_set],\
                                coeffs=[coeffs_count, coeffs_set])
    
    if start_dt != None:
        dt = start_dt

    cfl_cadence = 1
    cfl_threshold=0.1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=cfl_threshold)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    flow.add_property("Pe_rms", name='Pe')
    
    if zero_velocities:
        u = solver.state['u']
        w = solver.state['w']
        u['g'] *= 0
        w['g'] *= 0

    start_iter=solver.iteration
    start_sim_time = solver.sim_time

    try:
        start_time = time.time()
        logger.info('starting main loop')
        while solver.ok:
            
            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            # update lists
            if solver.iteration % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, (solver.sim_time-start_sim_time)/atmosphere.buoyancy_time, dt)
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
        if do_checkpointing:
            try:
                final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
                final_checkpoint.set_checkpoint(solver, wall_dt=1, write_num=1, set_num=1)
                solver.step(dt) #clean this up in the future...works for now.
            except:
                print('cannot save final checkpoint')

            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/final_checkpoint/')

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
    #   these lines really are all about setting up the output directory name
    data_dir = args['--root_dir']
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]
    #BCs
    if args['--fixed_T']:
        data_dir += '_fixed'
    elif args['--fixed_Tz']:
        data_dir += '_flux'
    #Diffusivities
    if args['--const_mu']:
        data_dir += '_constMu'
    else:
        data_dir += '_constNu'
    if args['--const_chi']:
        data_dir += '_constChi'
    else:
        data_dir += '_constKappa'
    #Base atmosphere
    data_dir += "_nrhocz{}_Ra{}_Pr{}_eps{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--Prandtl'], args['--epsilon'])
    if args['--label'] == None:
        data_dir += '/'
    else:
        data_dir += '_{}/'.format(args['--label'])
    logger.info("saving run in: {}".format(data_dir))
  

    #Timestepper type
    if args['--timestepper'] == 'rk443':
        rk443=True
    else:
        rk443=False

    #Restarting options
    if args['--start_new_files']:
        start_new_files = True
    else:
        start_new_files = False
    if args['--start_dt'] != None:
        start_dt = float(args['--start_dt'])
    else:
        start_dt = None
    if args['--zero_velocities']:
        zero_velocities=True
    else:
        zero_velocities=False

    #Resolution
    nx = args['--nx']
    if nx != None:
        nx = int(nx)
    nz = int(args['--nz'])

    #Diffusivity flags
    const_mu    = False
    const_kappa = True
    if args['--const_mu']:
        const_mu   = True
    if args['--const_chi']:
        const_kappa = False

    Lx = args['--Lx']
    aspect_ratio = float(args['--aspect_ratio'])
    if Lx != None:
        Lx = float(Lx)


    FC_constant_kappa(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      nx = nx,
                      nz = nz,
                      Lx = Lx,
                      aspect_ratio = aspect_ratio,
                      n_rho_cz=float(args['--n_rho_cz']),
                      epsilon=float(args['--epsilon']),
                      run_time=float(args['--run_time']),
                      fixed_T=args['--fixed_T'],
                      fixed_Tz=args['--fixed_Tz'],
                      const_mu=const_mu,
                      const_kappa=const_kappa,
                      restart=(args['--restart']),
                      zero_velocities=zero_velocities,
                      start_new_files=start_new_files,
                      start_dt = start_dt,
                      rk443=rk443,
                      safety_factor=float(args['--safety_factor']),
                      out_cadence=float(args['--out_cadence']),
                      data_dir=data_dir,
                      no_coeffs=args['--no_coeffs'])
