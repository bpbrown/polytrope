"""
Dedalus script for 2D or 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification, 
based on the evolution of a previous run, with or without tracers/chemistry.

Usage:
    FC_poly_rxn_inject.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 7/5]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]


                                         Rotation keywords
    --rotating                           Solve f-plane equations
    --Co=<Co>                            Convective Rossby number
    --Taylor=<Taylor>                    Taylor number [default: 1e4]
    --theta=<theta>                      Angle between z and rotation vector omega [default: 0]
    
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz
    --ny=<ny>                            Horizontal y (Fourier) resolution; if not set, ny=nx (3D only) 
    --3D                                 Do 3D run
    --mesh=<mesh>                        Processor mesh if distributing 3D run in 2D 

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom; default if no BCs specified)
    --mixed_flux_T                       Fixed T (top) and flux (bottom) BCs
    --fixed_flux                         Fixed flux boundary conditions (top and bottom)
    --const_nu                           If flagged, use constant nu 
    --const_chi                          If flagged, use constant chi 
    --dynamic_diffusivities              If flagged, use equations formulated in terms of dynamic diffusivities (μ,κ)
    
    --restart=<restart_file>             Restart from checkpoint
    --start_new_files                    Start new files while checkpointing

    --rk222                              Use RK222 as timestepper
    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]
    --split_diffusivities                If True, split the chi and nu between LHS and RHS to lower bandwidth
    
    --root_dir=<root_dir>                Root directory to save data dir in [default: ./]
    --label=<label>                      Additional label for run output directory
    --out_cadence=<out_cad>              The fraction of a buoyancy time to output data at [default: 0.1]
    --writes=<writes>                    Writes per file [default: 20]
    --no_coeffs                          If flagged, coeffs will not be output
    --no_join                            If flagged, skip join operation at end of run.

    --verbose                            Do extra output (Peclet and Nusselt numbers) to screen

    --chemistry                          Do chemistry in injected run
    --ChemicalPrandtl=<ChemicalPrandtl>  Ratio of chemical diffusivities to fluid viscosity [default: 1]
    --Qu_0=<Qu_0>                        Knob for controlling location of quench point combined with phi_0 [default: 5e-8]
    --phi_0=<phi_0>                      Knob for controlling ratio of chemical to density scale heights [default: 10]

    --scalar_file=<scalar_file>          Scalar slice file with initial slice to start from [default: None]
    --dynamics_file=<dynamics_file>      Dynamics coeff file with final slice to start from
"""
import logging
import numpy as np
import h5py

def FC_polytrope(dynamics_file,
                 Rayleigh=1e4, Prandtl=1, aspect_ratio=4,
                 Taylor=None, theta=0,
                 nz=128, nx=None, ny=None, threeD=False, mesh=None,
                 n_rho_cz=3, epsilon=1e-4, gamma=5/3,
                 run_time=23.5, run_time_buoyancies=None, run_time_iter=np.inf,
                 fixed_T=False, fixed_flux=False, mixed_flux_T=False,
                 const_mu=True, const_kappa=True,
                 dynamic_diffusivities=False, split_diffusivities=False,
                 chemistry=True,ChemicalPrandtl=1,Qu_0=5e-8,phi_0=10,
                 restart=None, start_new_files=False,
                 scalar_file=None, 
                 rk222=False, safety_factor=0.2,
                 max_writes=20,
                 data_dir='./', out_cadence=0.1, no_coeffs=False, no_join=False,
                 verbose=False):
    
    import dedalus.public as de
    from dedalus.tools  import post
    from dedalus.extras import flow_tools

    import time
    import os
    import sys
    from stratified_dynamics import polytropes    
    from tools.checkpointing import Checkpoint
    
    checkpoint_min   = 30
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = int(np.round(nz*aspect_ratio))
    if threeD and ny is None:
        ny = nx

    eqn_dict = {'nx' : nx,
                'nz' : nz,
                'constant_kappa' : const_kappa,
                'constant_mu' : const_mu,
                'epsilon' : epsilon,
                'gamma' : gamma,
                'n_rho_cz' : n_rho_cz,
                'aspect_ratio' : aspect_ratio,
                'fig_dir' : data_dir}
    if threeD:
        eqn_dict['mesh'] = mesh
        eqn_dict['nz'] = nz
        atmosphere = polytropes.FC_polytrope_rxn_3d(**eqn_dict)
        
    else:
        if dynamic_diffusivities:
            atmosphere = polytropes.FC_polytrope_2d_kappa(**eqn_dict)
        else:
            if chemistry:
                atmosphere = polytropes.FC_polytrope_rxn_2d(**eqn_dict)
            else:
                atmosphere = polytropes.FC_polytrope_2d(**eqn_dict)

    if epsilon < 1e-4:
        ncc_cutoff = 1e-14
    elif epsilon > 1e-1:
        ncc_cutoff = 1e-6
    else:
        ncc_cutoff = 1e-10

    problem_dict = {'ncc_cutoff' : ncc_cutoff,
                    'split_diffusivities' : split_diffusivities}
    if threeD:
        problem_dict['Taylor'] = Taylor
        problem_dict['theta']  = theta
    if chemistry:
        problem_dict['ChemicalPrandtl'] = ChemicalPrandtl
        problem_dict['Qu_0'] = Qu_0
        problem_dict['phi_0'] = phi_0
        
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, **problem_dict)
    
    if fixed_flux:
        atmosphere.set_BC(fixed_flux=True, stress_free=True)
    elif mixed_flux_T:
        atmosphere.set_BC(mixed_flux_temperature=True, stress_free=True)
    else:
        atmosphere.set_BC(fixed_temperature=True, stress_free=True)

    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk222:
        logger.info("timestepping using RK222")
        ts = de.timesteppers.RK222
        cfl_safety_factor = safety_factor*2
    else:
        logger.info("timestepping using RK443")
        ts = de.timesteppers.RK443
        cfl_safety_factor = safety_factor*4

    # Build solver
    solver = problem.build_solver(ts)

    #Check atmosphere
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,\
                                                                    atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    atmosphere.check_atmosphere(make_plots = False, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))

    if restart is None:
        mode = "overwrite"
    else:
        mode = "append"

    logger.info('checkpointing in {}'.format(data_dir))
    checkpoint = Checkpoint(data_dir)

    if restart is None:
        atmosphere.set_IC(solver)
        dt = None
    else:
        logger.info("restarting from {}".format(restart))
        dt = checkpoint.restart(restart, solver)

    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
    
    if run_time_buoyancies != None:
        solver.stop_sim_time    = solver.sim_time + run_time_buoyancies*atmosphere.buoyancy_time
    else:
        solver.stop_sim_time    = 100*atmosphere.thermal_time
    
    solver.stop_iteration   = solver.iteration + run_time_iter
    solver.stop_wall_time   = run_time*3600
    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    Hermitian_cadence = 100
    
    logger.info("stopping after {:g} time units".format(solver.stop_sim_time))
    logger.info("output cadence = {:g}".format(output_time_cadence))
    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=not(no_coeffs), mode=mode,max_writes=max_writes)

    # Reinjecting dynamics and tracers if desired
    logger.info("Re-injecting scalars")

    def reset_variable(key,h5_val,grid=False,grad=False):
        # Get variable
        k = solver.state[key]
        k.set_scales(1, keep_data=True)

        # Set grid or coefficient space
        if grid:
            gc = 'g'
            slice = solver.domain.dist.grid_layout.slices(k.meta[:]['scale'])
        else:
            gc = 'c'
            slice = solver.domain.dist.coeff_layout.slices(k.meta[:]['scale'])
        
        # Set initial value of variable
        k[gc] = h5_val[slice]

        if grad:  # Set gradient if called for
            k.differentiate('z',out=k)

        logger.info("Re-injecting: {}".format(key))
        return k

    objects = {}
    # Set all the dynamic variables

    if threeD:
        keys = ['u','u_z','v','v_z','w','w_z','T1','T1_z','ln_rho1']
        grads = [False,True,False,True,False,True,False,True,False,False,True,False,True]
        h5_keys = ['u','u','v','v','w','w','T','T','ln_rho']
    else:
        keys = ['u','u_z','w','w_z','T1','T1_z','ln_rho1']
        grads = [False,True,False,True,False,True,False,False,True,False,True]
        h5_keys = ['u','u','w','w','T','T','ln_rho']
    h5File_c = h5py.File(dynamics_file,'r')
    dt = h5File_c['scales']['timestep'][-1]
    h5File_c = h5File_c['tasks']
    for i,K in enumerate(keys):
        # Restarting with final profile of run
        objects[K] = reset_variable(K, h5File_c[h5_keys[i]][-1],grad=grads[i])

    if scalar_file != 'None':
        keys = ['f','f_z','C','C_z','G','G_z']
        grads = [False,True,False,True]
        h5_keys = ['f','f','C','C','G','G']
        h5File_g = h5py.File(scalar_file,'r')['tasks']
        for i,K in enumerate(keys):
            # Restarting with initial profile
            objects[K] = reset_variable(K, h5File_g[h5_keys[i]][0],grid=True,grad=grads[i])
        
    
    #Set up timestep defaults
    max_dt = output_time_cadence/2
    if dt is None: dt = max_dt
        
    cfl_cadence = 1
    cfl_threshold = 0.1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=cfl_threshold)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    if verbose:
        flow.add_property("Pe_rms", name='Pe')
        flow.add_property("Nusselt_AB17", name='Nusselt')
    
    start_iter=solver.iteration
    start_sim_time = solver.sim_time

    try:
        start_time = time.time()
        start_iter = solver.iteration
        logger.info('starting main loop')
        good_solution = True
        first_step = True
        while solver.ok and good_solution:
            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            effective_iter = solver.iteration - start_iter

            if threeD and effective_iter % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            # update lists
            if effective_iter % report_cadence == 0:
                Re_avg = flow.grid_average('Re')
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, (solver.sim_time-start_sim_time)/atmosphere.buoyancy_time, dt)
                if verbose:
                    log_string += '\n\t\tRe: {:8.5e}/{:8.5e}'.format(Re_avg, flow.max('Re'))
                    log_string += '; Pe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
                    log_string += '; Nu: {:8.5e}/{:8.5e}'.format(flow.grid_average('Nusselt'), flow.max('Nusselt'))
                else:
                    log_string += 'Re: {:8.3e}/{:8.3e}'.format(Re_avg, flow.max('Re'))
                logger.info(log_string)
                
            if not np.isfinite(Re_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_avg))
                    
            if first_step:
                if verbose:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern.png", dpi=1200)

                    import scipy.sparse.linalg as sla
                    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
                    fig = plt.figure()
                    ax = fig.add_subplot(1,2,1)
                    ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
                    ax = fig.add_subplot(1,2,2)
                    ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=1200)

                    logger.info("{} nonzero entries in LU".format(LU.nnz))
                    logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
                    logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))
                first_step = False
                start_time = time.time()
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()

        # Print statistics
        elapsed_time = end_time - start_time
        elapsed_sim_time = solver.sim_time
        N_iterations = solver.iteration-1
        logger.info('main loop time: {:e}'.format(elapsed_time))
        logger.info('Iterations: {:d}'.format(N_iterations))
        logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
        if N_iterations > 0:
            logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
        
        if not no_join:
            logger.info('beginning join operation')
            try:
                final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
                final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
                solver.step(dt) #clean this up in the future...works for now.
                post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=True)
            except:
                print('cannot save final checkpoint')
                
            logger.info(data_dir+'/checkpoint/')
            post.merge_process_files(data_dir+'/checkpoint/', cleanup=True)
            
            for task in analysis_tasks.keys():
                logger.info(analysis_tasks[task].base_path)
                post.merge_process_files(analysis_tasks[task].base_path, cleanup=True)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            if start_iter > 1:
                logger.info('Iterations (this run): {:d}'.format(N_iterations - start_iter))
                logger.info('Iterations (total): {:d}'.format(N_iterations - start_iter))
            logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
            if N_iterations > 0:
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
            if n_steps > 0:
                print('    iterations:', n_steps)
                print(' loop sec/iter:', main_loop_time/n_steps)
                print('    average dt:', solver.sim_time/n_steps)
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
    from numpy import inf as np_inf
    from fractions import Fraction

    import os
    import sys
    # save data in directory named after script
    #   these lines really are all about setting up the output directory name
    data_dir = args['--root_dir']
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]

    #BCs

    if args['--fixed_T'] and args['--verbose']:
        data_dir += '_fixed'
    elif args['--fixed_flux']:
        data_dir += '_flux'

    if args['--dynamic_diffusivities']:
        data_dir += '_dynamic'
    if args['--verbose']:
        #Diffusivities
        if args['--const_nu']:
            data_dir += '_constNu'
        else:
            data_dir += '_constMu'
        if args['--const_chi']:
            data_dir += '_constChi'
        else:
            data_dir += '_constKappa'

    if args['--3D']:
        data_dir +='_3D'
    else:
        data_dir +='_2D'

    #Base atmosphere
    data_dir += "_nrhocz{}_Ra{}_Pr{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--Prandtl'])
    if args['--rotating']:
        if args['--Co']:
            Co = float(args['--Co'])
            Taylor = float(args['--Rayleigh'])/float(args['--Prandtl'])/Co**2
            data_dir += "_Co{}".format(args['--Co'])
        else:
            Taylor = float(args['--Taylor'])
            Co = np.sqrt(float(args['--Rayleigh'])/float(args['--Prandtl'])/Taylor)
            data_dir += "_Ta{}".format(args['--Taylor'])
    else:
        Taylor = None
    data_dir += "_eps{}_a{}".format(args['--epsilon'], args['--aspect'])

    if args['--chemistry']:
        Qu_0 = float(args['--Qu_0'])
        phi_0 = float(args['--phi_0'])
        data_dir += "_Qu{}_phi{}".format(args['--Qu_0'],args['--phi_0'])
    
    if args['--label'] == None:
        data_dir += '/'
    else:
        data_dir += '_{}/'.format(args['--label'])

    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'

    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    logger = logging.getLogger(__name__)
    logger.info("saving run in: {}".format(data_dir))


    #Timestepper type
    if args['--rk222']:
        rk222=True
    else:
        rk222=False

    #Restarting options
    if args['--start_new_files']:
        start_new_files = True
    else:
        start_new_files = False

    #Resolution
    nx = args['--nx']
    if nx is not None:
        nx = int(nx)
    ny =  args['--ny']
    if ny is not None:
        ny = int(ny)
    nz = int(args['--nz'])

    #Diffusivity flags
    const_mu    = True
    const_kappa = True
    if args['--const_nu']:
        const_mu   = False
    if args['--const_chi']:
        const_kappa = False

    mesh = args['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]

    run_time_buoy = args['--run_time_buoy']
    if run_time_buoy != None:
        run_time_buoy = float(run_time_buoy)
        
    run_time_iter = args['--run_time_iter']
    if run_time_iter != None:
        run_time_iter = int(float(run_time_iter))
    else:
        run_time_iter = np_inf
        
    FC_polytrope(args['--dynamics_file'],
                 scalar_file=args['--scalar_file'],
                 Rayleigh=float(args['--Rayleigh']),
                 Prandtl=float(args['--Prandtl']),
                 Taylor=Taylor,
                 chemistry=args['--chemistry'],
                 ChemicalPrandtl=float(args['--ChemicalPrandtl']),
                 Qu_0=float(args['--Qu_0']),
                 phi_0=float(args['--phi_0']),
                 theta=float(args['--theta']),
                 threeD=args['--3D'],
                 mesh=mesh,
                 nx = nx,
                 ny = ny,
                 nz = nz,
                 aspect_ratio=float(args['--aspect']),
                 n_rho_cz=float(args['--n_rho_cz']),
                 epsilon=float(args['--epsilon']),
                 gamma=float(Fraction(args['--gamma'])),
                 run_time=float(args['--run_time']),
                 run_time_buoyancies=run_time_buoy,
                 run_time_iter=run_time_iter,
                 fixed_T=args['--fixed_T'],
                 fixed_flux=args['--fixed_flux'],
                 mixed_flux_T=args['--mixed_flux_T'],
                 const_mu=const_mu,
                 const_kappa=const_kappa,
                 dynamic_diffusivities=args['--dynamic_diffusivities'],
                 restart=(args['--restart']),
                 start_new_files=start_new_files,
                 rk222=rk222,
                 safety_factor=float(args['--safety_factor']),
                 out_cadence=float(args['--out_cadence']),
                 max_writes=int(float(args['--writes'])),
                 data_dir=data_dir,
                 no_coeffs=args['--no_coeffs'],
                 no_join=args['--no_join'],
                 split_diffusivities=args['--split_diffusivities'],
                 verbose=args['--verbose'])
