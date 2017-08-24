"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    Dash2.py [options]

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --stiffness=<stiffness>    Stiffness of radiative/convective interface [default: 3]
    --restart=<restart_file>   Restart from checkpoint
    --nz_rz=<nz_rz>            Vertical z (chebyshev) resolution in stable region   [default: 128]
    --nz_cz=<nz_cz>            Vertical z (chebyshev) resolution in unstable region [default: 128]
    --single_chebyshev         Use a single chebyshev domain across both stable and unstable regions.  Useful at low stiffness.
    --nx=<nx>                  Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>      Density scale heights across unstable layer [default: 1]
    --n_rho_rz=<n_rho_rz>      Density scale heights across stable layer   [default: 5]

    --width=<width>            Width of erf transition between two polytropes

    --rk222                    Use RK222 as timestepper

    --MHD                                Do MHD run
    --MagneticPrandtl=<MagneticPrandtl>  Magnetic Prandtl Number = nu/eta [default: 1] 
    --B0_amplitude=<B0_amplitude>        Strength of B0_field, scaled to isothermal sound speed at top of domain [default: 1]

    --root_dir=<root_dir>      Root directory to save data dir in [default: ./]    
    --label=<label>            Additional label for run output directory
    --verbose                  Produce diagnostic plots
"""
import logging

def FC_convection(Rayleigh=1e6, Prandtl=1, stiffness=3,
                  MHD=False, MagneticPrandtl=1, B0_amplitude=1,
                      n_rho_cz=1, n_rho_rz=5, 
                      nz_cz=128, nz_rz=128,
                      nx = None,
                      width=None,
                      single_chebyshev=True,
                      rk222=False,
                      restart=None, data_dir='./', verbose=False, label=None):
    import numpy as np
    import time
    import os

    def format_number(number, no_format_min=0.1, no_format_max=10):
        if number > no_format_max or number < no_format_min:
            try:
                mantissa = "{:e}".format(number).split("+")[0].split("e")[0].rstrip("0") or "0"
                power    = "{:e}".format(number).split("+")[1].lstrip("0") or "0"
            except:
                mantissa = "{:e}".format(number).split("-")[0].split("e")[0].rstrip("0") or "0"
                power    = "{:e}".format(number).split("-")[1].lstrip("0") or "0"
                power    = "-"+power
            if mantissa[-1]==".":
                mantissa = mantissa[:-1]
            mantissa += "e"
        else:
            mantissa = "{:f}".format(number).rstrip("0") or "0"
            if mantissa[-1]==".":
                mantissa = mantissa[:-1]
            power = ""
        number_string = mantissa+power
        return number_string
             
    data_dir = './'
    # save data in directory named after script
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]
    data_dir += "_nrhocz{}_Ra{}_S{}".format(format_number(n_rho_cz), format_number(Rayleigh), format_number(stiffness))
    if width:
        data_dir += "_erf{}".format(format_number(width))
    if args['--MHD']:
        data_dir+= '_MHD'
    if label:
        data_dir += "_{}".format(label)
    data_dir += '/'

    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'

    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    logger = logging.getLogger(__name__)
    logger.info("saving run in: {}".format(data_dir))
    
    import dedalus.public as de
    from dedalus.tools  import post
    from dedalus.extras import flow_tools


    from dedalus.core.future import FutureField
    from stratified_dynamics import multitropes
    from tools.checkpointing import Checkpoint

        
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    constant_Prandtl=True
    stable_top=True
    mixed_temperature_flux=True
        
    # Set domain
    if nx is None:
        nx = nz_cz*4
        
    if single_chebyshev:
        nz = nz_cz
        nz_list = [nz_cz]
    else:
        nz = nz_rz+nz_cz
        nz_list = [nz_rz, nz_cz]
    
    eqns_dict = {'stiffness' : stiffness,
                 'nx' : nx,
                 'nz' : nz_list, 
                 'n_rho_cz' : n_rho_cz,
                 'n_rho_rz' : n_rho_rz,
                 'verbose'   : verbose,
                 'width'    : width,
                 'constant_Prandtl' : constant_Prandtl,
                 'stable_top' : stable_top}
    if MHD:
         atmosphere = multitropes.FC_MHD_multitrope_guidefield_2d(**eqns_dict)
         
         atmosphere.set_IVP_problem(Rayleigh, Prandtl, MagneticPrandtl, guidefield_amplitude=B0_amplitude)
    else:
         atmosphere = multitropes.FC_multitrope(**eqns_dict)      
         atmosphere.set_IVP_problem(Rayleigh, Prandtl)
        
    atmosphere.set_BC()
    problem = atmosphere.get_problem()
        
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

    if restart is None:
        mode = "overwrite"
    else:
        mode = "append"

    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800, mode=mode)

    # initial conditions
    if restart is None:
        atmosphere.set_IC(solver)
        dt = None
    else:
        logger.info("restarting from {}".format(restart))
        dt = checkpoint.restart(restart, solver)
        
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time, atmosphere.top_thermal_time))
    
    max_dt = atmosphere.min_BV_time 
    max_dt = atmosphere.buoyancy_time*0.25
    if dt is None: dt = max_dt
    
    report_cadence = 1
    output_time_cadence = 0.1*atmosphere.buoyancy_time
    solver.stop_sim_time = np.inf
    solver.stop_iteration= np.inf
    solver.stop_wall_time = 23.*3600

    logger.info("output cadence = {:g}".format(output_time_cadence))

    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

    
    cfl_cadence = 1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

    CFL.add_velocities(('u', 'w'))
    if MHD:
        CFL.add_velocities(('Bx/sqrt(4*pi*rho_full)', 'Bz/sqrt(4*pi*rho_full)'))
        
    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    if MHD:
        flow.add_property("abs(dx(Bx) + dz(Bz))", name='divB')
        
    try:
        start_time = time.time()
        while solver.ok:

            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            # update lists
            if solver.iteration % report_cadence == 0:
                Re_avg = flow.grid_average('Re')
                if not np.isfinite(Re_avg):
                    solver.ok = False
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration, solver.sim_time, solver.sim_time/atmosphere.buoyancy_time, dt)
                log_string += 'Re: {:8.3e}/{:8.3e}'.format(Re_avg, flow.max('Re'))
                if MHD:
                     log_string += ', divB: {:8.3e}/{:8.3e}'.format(flow.grid_average('divB'), flow.max('divB'))

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
        post.merge_process_files(data_dir+'/checkpoint/', cleanup=True)

        for task in analysis_tasks:
            logger.info(analysis_tasks[task].base_path)
            post.merge_process_files(analysis_tasks[task].base_path, cleanup=True)

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
    
    if args['--width'] is not None:
        data_dir += "_erf{}".format(args['--width'])
        width = float(args['--width'])
    else:
        width = None
        
    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)
            
    FC_convection(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      stiffness=float(args['--stiffness']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      n_rho_rz=float(args['--n_rho_rz']),
                      nz_rz=int(args['--nz_rz']),
                      nz_cz=int(args['--nz_cz']),
                      single_chebyshev=args['--single_chebyshev'],
                      width=width,
                      nx=nx,
                      restart=(args['--restart']),
                      rk222=args['--rk222'],
                      data_dir=args['--root_dir'],
                      verbose=args['--verbose'],
                      MHD=args['--MHD'],
                      B0_amplitude=float(args['--B0_amplitude']),
                      label=args['--label'])
