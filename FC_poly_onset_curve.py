from FC_onset_solver import *

#Set up parameters
nx = 32
nz = 32
aspect_ratio=10
epsilon=1e-4
out_dir = './'
ra_steps = 2


#Values to solve over. Note that the program is parallelizable up to (nx/2 * # of steps in ra_range).
#   If nx = 32 and ra_range takes 100 steps, it can be run on 16*100 = 1600 processors.
n_rho_cz = [1, 3.5, 5, 7]
ra_range = np.logspace(1, 4, ra_steps) #Runs from 10 - 10,000 over 200 log-space steps

ONSET_CURVE_INDIVIDUAL = False
ONSET_CURVE_ALL = True
GROWTH_MODES    = True


for n_rho in n_rho_cz:
    #Set up solver and solve ra range at this n_rho
    solver = FC_onset_solver(ra_range, profiles=['u','w','T1'], nx=nx, nz=nz, 
                            aspect_ratio=aspect_ratio, n_rho_cz=n_rho, epsilon=epsilon,
                            constant_kappa=True,
                            comm=MPI.COMM_SELF, out_dir=out_dir)
    solver.full_parallel_solve() #Merges files by default.
   
    #Read info and plot
    wavenumbers, profiles, filename, atmosphere = solver.read_file()
    if wavenumbers != None and GROWTH_MODES:
        solver.plot_growth_modes(wavenumbers, filename)
    if wavenumbers != None and ONSET_CURVE_INDIVIDUAL:
        atmosphere = (atmosphere[0], atmosphere[1], solver.Lx)
        solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=True, save=True)

if ONSET_CURVE_ALL:
    for i in range(len(n_rho_cz)):
        #Set up solver again for grabbing proper Lx
        solver = FC_onset_solver(ra_range, profiles=['u','w','T1'], nx=nx, nz=nz, 
                                aspect_ratio=aspect_ratio, n_rho_cz=n_rho_cz[i], epsilon=epsilon, 
                                constant_kappa=True,
                                comm=MPI.COMM_SELF, out_dir=out_dir)
        wavenumbers, profiles, filename, atmosphere = solver.read_file(n_rho=n_rho_cz[i])
        if atmosphere != None:
            atmosphere = (atmosphere[0], atmosphere[1], solver.Lx)
        if wavenumbers == None:
            continue
        if i == 0 and len(n_rho_cz) > 1:
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=True, save=False)
        elif i == len(n_rho_cz)-1:
            if len(n_rho_cz) == 1:
                clear = True
            else:
                clear = False
            #Optional figure name specification
            figname = 'onset_{:.04g}x{:.04g}_nrhos_{:.04g}-{:.04g}.png'.format(nx, nz, n_rho_cz[0], n_rho_cz[-1])
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=clear, save=True, figname=figname)
        else:
            solver.plot_onset_curve(wavenumbers, filename, atmosphere, clear=False, save=False)
