"""
Dedalus script for 2D compressible convection in a polytrope,
with 3.5 density scale heights of stratification.

Usage:
    Dash2.py [options] bootstrap
    Dash2.py [options]

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --stiffness=<stiffness>    Stiffness of radiative/convective interface [default: 3]
    --m_rz=<m_rz>              Polytropic index of stable layer [default: 3]
    --gamma=<gamma>            Gamma of ideal gas (cp/cv) [default: 5/3]

    --MHD                                Do MHD run
    --MagneticPrandtl=<MagneticPrandtl>  Magnetic Prandtl Number = nu/eta [default: 1] 
    --B0_amplitude=<B0_amplitude>        Strength of B0_field, scaled to isothermal sound speed at top of domain [default: 1]
    
    --restart=<restart_file>   Restart from checkpoint
    --nz_rz=<nz_rz>            Vertical z (chebyshev) resolution in stable region   [default: 128]
    --nz_cz=<nz_cz>            Vertical z (chebyshev) resolution in unstable region [default: 128]
    --single_chebyshev         Use a single chebyshev domain across both stable and unstable regions.  Useful at low stiffness.
    --nx=<nx>                  Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --n_rho_cz=<n_rho_cz>      Density scale heights across unstable layer [default: 1]
    --n_rho_rz=<n_rho_rz>      Density scale heights across stable layer   [default: 5]

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    
    --fixed_flux               Fixed flux thermal BCs
    --dynamic_diffusivities    If flagged, use equations formulated in terms of dynamic diffusivities (μ,κ)

    --rk222                    Use RK222 as timestepper

    --superstep                Superstep equations by using average rather than actual vertical grid spacing
    --dense                    Oversample matching region with extra chebyshev domain
    --nz_dense=<nz_dense>      Vertical z (chebyshev) resolution in oversampling region   [default: 64]
    
    --width=<width>            Width of erf transition between two polytropes
    
    --root_dir=<root_dir>      Root directory to save data dir in [default: ./]    
    --label=<label>            Additional label for run output directory
    --out_cadence=<out_cad>    The fraction of a buoyancy time to output data at [default: 0.1]
    --writes=<writes>          Writes per file [default: 20]
    --no_coeffs                If flagged, coeffs will not be output
    --no_join                  If flagged, skip join operation at end of run

    --verbose                  Produce diagnostic plots

    --init_file=<init_file>    The equilibrated, low Ra run from which the bootstrap process begins
    --ra_end=<ra_end>          Ending Rayleigh number [default: 1e6]
    --bsp_nx=<bsp_nx>          If supplied, a list of length equal to the number of steps giving x resolutions
    --bsp_nz=<bsp_nz>          If supplied, a list of length equal to the number of steps giving z resolutions
    --bsp_step=<bsp_step> The time in buoyancy  [default: 50.]
"""
import logging
import numpy as np
import time
import os
import sys
from fractions import Fraction

def FC_convection(Rayleigh=1e6, Prandtl=1, stiffness=3, m_rz=3, gamma=5/3,
                  MHD=False, MagneticPrandtl=1, B0_amplitude=1,
                  n_rho_cz=1, n_rho_rz=5, 
                  nz_cz=128, nz_rz=128,
                  nx = None,
                  width=None,
                  single_chebyshev=False,
                  rk222=False,
                  superstep=False,
                  dense=False, nz_dense=64,
                  oz=False,
                  fixed_flux=False,
                  run_time=23.5, run_time_buoyancies=np.inf, run_time_iter=np.inf,
                  dynamic_diffusivities=False,
                  max_writes=20,out_cadence=0.1, no_coeffs=False, no_join=False,
                  restart=None, data_dir='./', verbose=False, label=None):
    
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

    checkpoint_min = 30
        
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
                 'stable_top' : stable_top,
                 'gamma': gamma,
                 'm_rz':m_rz}
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

    # initial conditions
    if restart is None:
        atmosphere.set_IC(solver)
        dt = None
    else:
        logger.info("restarting from {}".format(restart))
        dt = checkpoint.restart(restart, solver)
        
    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
        
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time, atmosphere.top_thermal_time))
    
    max_dt = atmosphere.min_BV_time 
    max_dt = atmosphere.buoyancy_time*out_cadence
    if dt is None: dt = max_dt
    
    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    solver.stop_sim_time  = solver.sim_time + run_time_buoyancies*atmosphere.buoyancy_time
    solver.stop_iteration = solver.iteration + run_time_iter
    solver.stop_wall_time = run_time*3600
    
    logger.info("output cadence = {:g}".format(output_time_cadence))

    analysis_tasks = atmosphere.initialize_output(solver, data_dir, coeffs_output=not(no_coeffs), sim_dt=output_time_cadence, max_writes=max_writes, mode=mode)

    
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
    from bootstrap import bootstrap
    from docopt import docopt
    args = docopt(__doc__)
    logger = logging.getLogger(__name__)

    if args['--width'] is not None:
        data_dir += "_erf{}".format(args['--width'])
        width = float(args['--width'])
    else:
        width = None
        
    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)

    run_time_buoy = args['--run_time_buoy']
    if run_time_buoy != None:
        run_time_buoy = float(run_time_buoy)
    else:
        run_time_buoy = np.inf
        
    run_time_iter = args['--run_time_iter']
    if run_time_iter != None:
        run_time_iter = int(float(run_time_iter))
    else:
        run_time_iter = np.inf
        
    kwargs = {"Rayleigh":float(args['--Rayleigh']),
              "Prandtl":float(args['--Prandtl']),
              "stiffness":float(args['--stiffness']),
              "m_rz":float(args['--m_rz']),
              "gamma":float(Fraction(args['--gamma'])),
              "MHD":args['--MHD'],
              "MagneticPrandtl":float(args['--MagneticPrandtl']),
              "B0_amplitude":float(args['--B0_amplitude']),
              "n_rho_cz":float(args['--n_rho_cz']),
              "n_rho_rz":float(args['--n_rho_rz']),
              "nz_rz":int(args['--nz_rz']),
              "nz_cz":int(args['--nz_cz']),
              "single_chebyshev":args['--single_chebyshev'],
              "width":width,
              "nx":nx,
              "restart":(args['--restart']),
              "data_dir":args['--root_dir'],
              "verbose":args['--verbose'],
              "no_coeffs":args['--no_coeffs'],
              "no_join":args['--no_join'],
              "out_cadence":float(args['--out_cadence']),
              "fixed_flux":args['--fixed_flux'],
              "dynamic_diffusivities":args['--dynamic_diffusivities'],
              "dense":args['--dense'],
              "nz_dense":int(args['--nz_dense']),
              "rk222":args['--rk222'],
              "max_writes":int(float(args['--writes'])),
              "superstep":args['--superstep'],
              "run_time":float(args['--run_time']),
              "run_time_buoyancies":run_time_buoy,
              "run_time_iter":run_time_iter,
              "label":args['--label']}
    
    if args['bootstrap']:
        logger.info("Bootstrapping...")
        if args['--init_file']:
            init_file = args['--init_file']
        else:
            raise ValueError("Must specify a starting file if bootstrapping")
        ra_end = float(args['--ra_end'])
        
        bsp_step_time = float(args['--bsp_step'])
        bsp_nx = args['--bsp_nx']
        bsp_nz = args['--bsp_nz']
        bootstrap(init_file,ra_end,kwargs,nx=bsp_nx,nz=bsp_nz,step_run_time=bsp_step_time)
    else:
        FC_convection(**kwargs)
