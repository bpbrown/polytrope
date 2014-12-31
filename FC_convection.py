import numpy as np
import time
import os
import sys
import equations

import matplotlib.pyplot as plt

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

Rayleigh = 4e4
Prandtl = 1

# Set domain
Lz = 100
Lx = 3*Lz
    
atmosphere = equations.polytrope(nx=96, nz=96, Lx=Lx, Lz=Lz)
problem = atmosphere.set_FC_problem(Rayleigh, Prandtl)

if atmosphere.domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

ts = timesteppers.RK443
cfl_safety_factor = 0.2*4

logger.info("--------------------------------------")
logger.info("pre:  domain {}".format(atmosphere.domain.dealias))
logger.info("pre:  T0    {}".format(atmosphere.T0['g'].shape))
logger.info("pre:  rho0  {}".format(atmosphere.rho0['g'].shape))
logger.info("pre:  scale {}".format(atmosphere.scale['g'].shape))
logger.info("pre:  del_ln_rho0 {}".format(atmosphere.del_ln_rho0['g'].shape))
logger.info("pre:  del_s0 {}".format(atmosphere.del_s0['g'].shape))
logger.info("      ---------------------")
logger.info("pre:  T0 scales   {}".format(atmosphere.T0.meta[:]['scale']))
logger.info("pre:  rho0 scales   {}".format(atmosphere.rho0.meta[:]['scale']))
logger.info("pre:  scale scales {}".format(atmosphere.scale.meta[:]['scale']))
logger.info("pre:  del_ln_rho0 scales {}".format(atmosphere.del_ln_rho0.meta[:]['scale']))
logger.info("pre:  del_s0 scales{}".format(atmosphere.del_s0.meta[:]['scale']))
logger.info("--------------------------------------")
logger.info("pre:  scale    {}".format(atmosphere.scale['c'][0,:]))

logger.info("building the solver now")
# Build solver
solver = problem.build_solver(ts)

logger.info("--------------------------------------")
logger.info("post: domain {}".format(atmosphere.domain.dealias))
logger.info("post: T0    {}".format(atmosphere.T0['g'].shape))
logger.info("post: rho0  {}".format(atmosphere.rho0['g'].shape))
logger.info("post: scale {}".format(atmosphere.scale['g'].shape))
logger.info("post: del_ln_rho0 {}".format(atmosphere.del_ln_rho0['g'].shape))
logger.info("post: del_s0 {}".format(atmosphere.del_s0['g'].shape))
logger.info("      ---------------------")
logger.info("post: T0 scales   {}".format(atmosphere.T0.meta[:]['scale']))
logger.info("post: rho0 scales   {}".format(atmosphere.rho0.meta[:]['scale']))
logger.info("post: scale scales {}".format(atmosphere.scale.meta[:]['scale']))
logger.info("post: del_ln_rho0 scales {}".format(atmosphere.del_ln_rho0.meta[:]['scale']))
logger.info("post: del_s0 scales{}".format(atmosphere.del_s0.meta[:]['scale']))
logger.info("--------------------------------------")
logger.info("post: scale    {}".format(atmosphere.scale['c'][0,:]))

x = atmosphere.domain.grid(0)
z = atmosphere.domain.grid(1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(solver.evaluator.vars['del_ln_rho0']['g'][0,:])
fig.savefig("del_ln_rho0_build_{:d}.png".format(atmosphere.domain.distributor.rank))


# initial conditions
u = solver.state['u']
w = solver.state['w']
T = solver.state['T1']
s = solver.state['s']
ln_rho = solver.state['ln_rho1']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

for key in solver.problem.ncc_manager.ncc_strings:
    logger.info("NCC: {}".format(key)) #, solver.problem.ncc_manager.ncc_matrices[key]))

A0 = 1e-6
np.random.seed(1+atmosphere.domain.distributor.rank)

#atmosphere.T0.set_scales(domain.dealias, keep_data=True)
T.set_scales(atmosphere.domain.dealias, keep_data=True)
z_dealias = atmosphere.domain.grid(axis=1, scales=atmosphere.domain.dealias)
T['g'] = A0*np.sin(np.pi*z_dealias/Lz)*np.random.randn(*T['g'].shape)*atmosphere.T0['g']
#ln_rho['g'] = A0*np.sin(np.pi*z/Lz)*np.random.randn(*s['g'].shape)/atmosphere.rho0['g']

logger.info("A0 = {:g}".format(A0))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))

logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time, atmosphere.top_thermal_time))


max_dt = atmosphere.buoyancy_time

report_cadence = 1
output_time_cadence = 0.1*atmosphere.buoyancy_time
solver.stop_sim_time = 0.25*atmosphere.thermal_time
solver.stop_iteration= np.inf
solver.stop_wall_time = 0.25*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", sim_dt=output_time_cadence, max_writes=20, parallel=False)

analysis_slice.add_task("s", name="s")
analysis_slice.add_task("s - integ(s, 'x')/Lx", name="s'")
analysis_slice.add_task("u", name="u")
analysis_slice.add_task("w", name="w")
analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")


do_checkpointing=False
if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)


    
cfl_cadence = 1
CFL = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=max_dt)

CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u + w*w)*Lz/ nu", name='Re')


start_time = time.time()
while solver.ok:

    dt = CFL.compute_dt()
    # advance
    solver.step(dt)
        
    # update lists
    if solver.iteration % report_cadence == 0:
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, Re: {:8.3e}/{:8.3e}'.format(solver.iteration, solver.sim_time, dt,
                                                                                                flow.grid_average('Re'), flow.max('Re'))
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

if (atmosphere.domain.distributor.rank==0):

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


