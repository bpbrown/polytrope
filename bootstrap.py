import logging
import numpy as np
from FC_multi import FC_convection
def bootstrap(Ra_start, Ra_end, flags, nx=None, nz=None, nproc=None, run_function=FC_convection, step_run_time=50.):
    """bootstrap run_function in steps of 10 from Ra_start to Ra_end.

    nx: if supplied, a list of length equal to the number of steps 
    nz: if supplied, a list of length equal to the number of steps 
    nproc: if supplied, a list of length equal to the number of steps 

    """

    logger = logging.getLogger(__name__)

    n_steps = int(np.log10(Ra_end/Ra_start))
    logger.info("Bootstrapping from Ra = {:5.2e} to Ra = {:5.2e} in {:d} steps".format(Ra_start, Ra_end, n_steps))

    if nx:
        assert len(nx) == n_steps
    else:
        nx = [None for i in range(n_steps)]
    if nz:
        assert len(nz) == n_steps
    else:
        nz = [None for i in range(n_steps)]

    if nproc:
        assert len(nproc) == n_steps
        raise NotImplementedError("Variable numbers of processors is not yet implemented.")
    else:
        nproc = [None for i in range(n_steps)]

    # apply the supplied run_time_buoyancies only to the Ra_end run
    try:
        old_run_time_buoyancies = flags['run_time_buoyancies']
    except KeyError:
        old_run_time_buoyancies = np.inf

    flags['run_time_buoyancies'] = step_run_time
    
    Ra = Ra_start
    for i in range(n_steps-1):
        print("Step {}".format(i))
        print("================")
        Ra *= 10
        flags['Rayleigh'] = Ra
        if nx[i]:
            flags['nx'] = nx[i]
        if nz[i]:
            flags['nz']  = nz[i]
        run_function(**flags)

    print("================")
    print("Beginning target run.")
    flags['Rayleigh'] = Ra_end
    if nx[-1]:
        flags['nx'] = nx[-1]
    if nz[-1]:
        flags['nz'] = nz[-1]
    flags['run_time_buoyancies'] = old_run_time_buoyancies

    run_function(**flags)

def test_function(**kwargs):
    for k,v in kwargs.items():
        print("{}: {}".format(k,v))
    print()

if __name__ == "__main__":
    flags = {"Rayleigh": 1e3, "Stiffness":10}
    bootstrap(1e3,1e8,flags,run_function=test_function,nx = [512,512,512,1024,1536])
