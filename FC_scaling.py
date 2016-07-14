"""
Dedalus script for 2D compressible convection

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the isothermal sound crossing
time at the top of the layer and the temperature gradient length scale.

This version of the script is intended for scaling and performance tests.

Usage:
    FC_scaling.py [options] 

Options:
    --nz=<nz>                 Number of Chebyshev modes [default: 128]
    --nx=<nx>                 Number of Fourier modes; default is aspect*nz
    --aspect=<aspect>         Aspect ratio [default: 2]
    --Rayleigh=<Rayleigh>     Rayleigh number of the convection [default: 1e6]
"""


import numpy as np
import time
import os
import sys
import equations

import logging
logger = logging.getLogger(__name__)

from dedalus.public import *
from dedalus.tools  import post
from dedalus.extras import flow_tools

from docopt import docopt
args = docopt(__doc__)
nz = int(args['--nz'])
nx = args['--nx']
aspect = int(args['--aspect'])
if nx is None:
        nx = nz*aspect
else:
        nx = int(nx)
Rayleigh_string = args['--Rayleigh']
Rayleigh = float(Rayleigh_string)        


initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

#Rayleigh = 4e4
Prandtl = 1

n_rho_cz=3

atmosphere = equations.FC_polytrope(nx=nx, nz=nz, constant_kappa=True, n_rho_cz=n_rho_cz)
atmosphere.set_IVP_problem(Rayleigh, Prandtl)

atmosphere.set_BC()
problem = atmosphere.get_problem()


ts = timesteppers.RK443
cfl_safety_factor = 0.2*4

ts = timesteppers.RK222
cfl_safety_factor = 0.2*2


# Build solver
solver = problem.build_solver(ts)

atmosphere.set_IC(solver)

max_dt = atmosphere.buoyancy_time*0.25

report_cadence = 1
output_time_cadence = 0.1*atmosphere.buoyancy_time
solver.stop_sim_time = 0.05*atmosphere.thermal_time
solver.stop_iteration= 100+1
solver.stop_wall_time = 0.25*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

    
cfl_cadence = 1
CFL = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

    

CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u + w*w)*Lz/ nu", name='Re')


while solver.ok:

    dt = CFL.compute_dt()
    # advance
    solver.step(dt)
        
    # update lists
    if solver.iteration % report_cadence == 0:
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, Re: {:8.3e}/{:8.3e}'.format(solver.iteration, solver.sim_time, dt,
                                                                                                flow.grid_average('Re'), flow.max('Re'))
        logger.info(log_string)

    if solver.iteration == 1:
        # pull out transpose init costs from startup time.
        start_time = time.time()        

        
end_time = time.time()

# Print statistics
elapsed_time = end_time - start_time
elapsed_sim_time = solver.sim_time
N_iterations = solver.iteration 
logger.info('main loop time: {:e}'.format(elapsed_time))
logger.info('Iterations: {:d}'.format(N_iterations))
logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))

if (atmosphere.domain.distributor.rank==0):

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


