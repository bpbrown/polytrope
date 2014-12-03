import numpy as np
import time
import os
import sys
import equations

import logging
logger = logging.getLogger(__name__)

from dedalus2.public import *
from dedalus2.tools  import post
from dedalus2.extras import flow_tools
#from dedalus2.extras.checkpointing import Checkpoint

initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]+'/'

Rayleigh = 4e8
Prandtl = 1
poly_n = 2
gamma = 5/3

epsilon = 1/(gamma-1) - poly_n 


# Set domain
Lz = 100
Lx = 3*Lz

nx = np.int(64*3/2)
nz = np.int(32*3/2)

x_basis = Fourier(nx,   interval=[0., Lx], dealias=2/3)
z_basis = Chebyshev(nz, interval=[0., Lz], dealias=2/3)
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))
    
atmosphere = equations.polytrope(domain, gamma=gamma, epsilon=epsilon)
pde = atmosphere.set_anelastic_problem(Rayleigh, Prandtl)

ts = timesteppers.RK443
cfl_safety_factor = 0.2*4

# Build solver
solver = solvers.IVP(pde, domain, ts)

x = domain.grid(0)
z = domain.grid(1)

# initial conditions
u = solver.state['u']
w = solver.state['w']
s = solver.state['s']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

A0 = 1e-6

s['g'] = A0*np.sin(5*np.pi*z/Lz)*np.sin(2*np.pi*x/Lx)

logger.info("A0 = {:g}".format(A0))
logger.info("s = {:g} -- {:g}".format(np.min(s['g']), np.max(s['g'])))


# determine characteristic timescales
chi = solver.evaluator.vars['chi']
thermal_time = Lz**2/chi
top_thermal_time = 1/chi
logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(thermal_time, top_thermal_time))


max_dt = 1
cfl_cadence = 1
cfl = flow_tools.CFL_conv_2D(solver, max_dt, cfl_cadence=cfl_cadence)


report_cadence = 1
output_time_cadence = 1
solver.stop_sim_time = 100
solver.stop_iteration= np.inf
solver.stop_wall_time = 0.25*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", sim_dt=output_time_cadence, max_writes=20, parallel=False)

analysis_slice.add_task("s", name="s")
analysis_slice.add_task("s - Integrate(s, dx)/Lx", name="s'")
analysis_slice.add_task("u", name="u")
analysis_slice.add_task("w", name="w")
analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")


do_checkpointing=False
if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

solver.dt = max_dt

start_time = time.time()
while solver.ok:

    # advance
    solver.step(solver.dt)
    
    if solver.iteration % cfl_cadence == 0 and solver.iteration>=2*cfl_cadence:
        domain.distributor.comm_world.Barrier()
        solver.dt = cfl.compute_dt(cfl_safety_factor)
    
    # update lists
    if solver.iteration % report_cadence == 0:
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e},'.format(solver.iteration, solver.sim_time, solver.dt)
        logger.info(log_string)
        
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
logger.info(analysis_slice.base_path)
post.merge_analysis(analysis_slice.base_path)

if (domain.distributor.rank==0):

    N_TOTAL_CPU = domain.distributor.comm_world.size
    
    # Print statistics
    print('-' * 40)
    total_time = end_time-initial_time
    main_loop_time = end_time - start_time
    startup_time = start_time-initial_time
    print('  startup time:', startup_time)
    print('main loop time:', main_loop_time)
    print('    total time:', total_time)
    print('Iterations:', solver.iteration)
    print('Average timestep:', solver.sim_time / solver.iteration)
    print('scaling:',
          ' {:d} {:d} {:d} {:d} {:d} {:d}'.format(N_TOTAL_CPU, 0, N_TOTAL_CPU,nx, 0, nz),
          ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                            main_loop_time, 
                                                            main_loop_time/solver.iteration, 
                                                            main_loop_time/solver.iteration/(nx*nz), 
                                                            N_TOTAL_CPU*main_loop_time/solver.iteration/(nx*nz)))
    print('-' * 40)


