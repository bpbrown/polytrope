import numpy as np
import os
from mpi4py import MPI

from dedalus import public as de

#import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class polytrope:
    def __init__(self, nx=256, nz=128, Lx=30, Lz=10, epsilon=1e-4, gamma=5/3, constant_diffusivities=True):
        
        x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        z_basis = de.Chebyshev('z', nz, interval=[0., Lz], dealias=3/2)
        self.domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
        
        self.constant_diffusivities = constant_diffusivities
        self._set_atmosphere(epsilon, gamma)

    def _new_ncc(self):
        field = self.domain.new_field()
        field.meta['x']['constant'] = True
        return field
        
    def _set_atmosphere(self, epsilon, gamma):
        # polytropic atmosphere characteristics
        self.epsilon = epsilon
        self.gamma = gamma
        self.poly_n = 1/(gamma-1) - epsilon

        self.x = self.domain.grid(0)
        self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
        self.nx = self.domain.bases[0].coeff_size

        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z0 = 1. + self.Lz

        self.del_ln_rho_factor = -self.poly_n
        
        self.del_ln_rho0 = self._new_ncc()
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)
        
        self.rho0 = self._new_ncc()
        self.rho0['g'] = (self.z0 - self.z)**self.poly_n

        self.del_s0_factor = - self.epsilon/self.gamma

        self.del_s0 = self._new_ncc()
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)

        self.delta_s = self.del_s0_factor*np.log(self.z0)

        self.del_T0 = -1

        self.T0 = self._new_ncc()
        self.T0['g'] = self.z0 - self.z       

        self.scale = self._new_ncc()
        if self.constant_diffusivities:
            self.scale['g'] = self.z0 - self.z
        else:
            # this may be a better scale factor for the diffusion terms.  Wow.  None of these work particularly well.
            self.scale['g'] = (self.z0 - self.z)**(3)
            self.scale['g'] = (self.z0 - self.z)**(self.poly_n)
            self.scale['g'] = (self.z0 - self.z)**(self.poly_n+1)

        self.g = self.poly_n + 1
        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi = self._new_ncc()
        self.phi['g'] = -self.g*(self.z0 - self.z)
        
        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_n = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_n, self.epsilon, self.gamma))
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        
        logger.info("   density scale heights = {:g}".format(np.log(self.Lz**self.poly_n)))
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format((self.z0-self.Lz)/self.poly_n, (self.z0)/self.poly_n))
        logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(((self.z0-self.Lz)/self.poly_n)/(self.Lx/self.nx), ((self.z0)/self.poly_n)/(self.Lx/self.nx)))

        # min of global quantity
        self.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']))), op=MPI.MIN)
        self.freefall_time = np.sqrt(self.Lz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,self.freefall_time,self.buoyancy_time))

        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.plot(self.z[0,:], self.del_ln_rho0['g'][0,:])
        #ax.plot(self.del_ln_rho0['g'][0,:])
        #fig.savefig("del_ln_rho0_{:d}.png".format(self.domain.distributor.rank))
        
    def _set_diffusivity(self, Rayleigh, Prandtl):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))

        self.nu = self._new_ncc()
        self.chi = self._new_ncc()

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s)*self.g)/Rayleigh)
            chi = nu/Prandtl

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))

            # determine characteristic timescales
            self.thermal_time = self.Lz**2/chi
            self.top_thermal_time = 1/chi

            self.viscous_time = self.Lz**2/nu
            self.top_viscous_time = 1/nu
        
            logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time, self.top_thermal_time))
        else:
        #elif constant_diffusion:
            # take constant mu, kappa based on setting a top-of-domain Rayleigh number
            nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s)*self.g)/Rayleigh)
            chi_top = nu_top/Prandtl

            # we're using internal chebyshev grid points... right.
            nu  =  nu_top/(self.rho0['g'])
            chi = chi_top/(self.rho0['g'])

            #nu  =  nu_top/(self.rho0['g']/self.rho0['g'][...,-1][0])
            #chi = chi_top/(self.rho0['g']/self.rho0['g'][...,-1][0])
            logger.info("   using constant mu, kappa")
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu[...,-1][0], chi[...,-1][0]))
            logger.info("   nu_mid = {:g}, chi_mid = {:g}".format(nu[...,self.nz/2][0], chi[...,self.nz/2][0]))
            logger.info("   nu_bot = {:g}, chi_bot = {:g}".format(nu[...,0][0], chi[...,0][0]))

            # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
            self.thermal_time = self.Lz**2/chi[...,self.nz/2][0]
            self.top_thermal_time = 1/chi_top

            self.viscous_time = self.Lz**2/nu[...,self.nz/2][0]
            self.top_viscous_time = 1/nu_top

            logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time, self.top_thermal_time))

        self.nu['g'] = nu
        self.chi['g'] = chi

        
        return nu, chi

    def _set_parameters(self):
        self.problem.parameters['nu'] = self.nu
        self.problem.parameters['chi'] = self.chi
        
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['del_T0'] = self.del_T0
        
        self.problem.parameters['rho0'] = self.rho0
        self.problem.parameters['del_ln_rho0'] = self.del_ln_rho0
        
        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['Cv'] = 1/(self.gamma-1)

        self.problem.parameters['del_s0'] = self.del_s0

        self.problem.parameters['g']  = self.g
        self.problem.parameters['phi']  = self.phi
        
        self.problem.parameters['scale'] = self.scale

        self.problem.parameters['Lz'] = self.Lz
        self.problem.parameters['Lx'] = self.Lx

        
    def get_problem(self):
        return self.problem
    
class polytrope_flux(polytrope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_diffusivity(self, Rayleigh, Prandtl):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        
        # take constant nu, chi
        nu = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s)*self.g)/Rayleigh)
        chi = nu/Prandtl

        logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))

        # determine characteristic timescales
        self.thermal_time = self.Lz**2/chi
        self.top_thermal_time = 1/chi

        self.viscous_time = self.Lz**2/nu
        self.top_viscous_time = 1/nu

        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time, self.top_thermal_time))

        self.nu = nu
        self.chi = chi

        
        return nu, chi        

    def set_BC(self):
        self.problem.add_bc( "left(Q_z) = 1")
        self.problem.add_bc("right(Q_z) = 1")
            
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")
        
class AN_polytrope(polytrope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_IVP_problem(self, Rayleigh, Prandtl):

        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'], cutoff=1e-10)

        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")

        # Lecoanet et al 2014, ApJ, eqns D23-D31
        self.viscous_term_w = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z)))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"
        
        self.problem.add_equation("(scale)  * (dt(w)  - "+self.viscous_term_w+" + dz(pomega) - s*g) = -(scale)  * (u*dx(w) + w*w_z)")
        self.problem.add_equation("(scale)  * (dt(u)  - "+self.viscous_term_u+" + dx(pomega)      ) = -(scale)  * (u*dx(u) + w*u_z)")
        self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*dz(s))")
        # seems to not help speed --v
        # self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*(-Q_z/T0_local))")

        self.problem.add_equation("(scale)*(dx(u) + w_z + w*del_ln_rho0) = 0")

    def set_eigenvalue_problem(self, Rayleigh, Prandtl):

        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'], cutoff=1e-6)

        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")

        # Lecoanet et al 2014, ApJ, eqns D23-D31
        self.viscous_term_w = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z)))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"
        
        self.problem.add_equation("(scale)  * (dt(w)  - "+self.viscous_term_w+" + dz(pomega) - s*g) = -(scale)  * (u*dx(w) + w*w_z)")
        self.problem.add_equation("(scale)  * (dt(u)  - "+self.viscous_term_u+" + dx(pomega)      ) = -(scale)  * (u*dx(u) + w*u_z)")
        self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*dz(s))")
        # seems to not help speed --v
        # self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*(-Q_z/T0_local))")

        self.problem.add_equation("(scale)*(dx(u) + w_z + w*del_ln_rho0) = 0")

    def set_BC(self, fixed_flux=False):
        if fixed_flux:
            self.problem.add_bc( "left(Q_z) = 0")
            self.problem.add_bc("right(Q_z) = 0")
        else:
            self.problem.add_bc( "left(s) = 0")
            self.problem.add_bc("right(s) = 0")
            
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0", condition="nx != 0")
        self.problem.add_bc( "left(pomega) = 0", condition="nx == 0")
        self.problem.add_bc("right(w) = 0")


class FC_polytrope(polytrope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_IVP_problem(self, Rayleigh, Prandtl):

        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','T1', 'Q_z', 'ln_rho1', 's'])
        
        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()
        self._set_subs()
        
        # here, nu and chi are constants        
        self.viscous_term_w = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z)))"

        self.nonlinear_viscous_w = " nu*(    u_z*dx(ln_rho1) + 2*w_z*dz(ln_rho1) + dx(ln_rho1)*dx(w) - 2/3*dz(ln_rho1)*(dx(u)+w_z))"
        self.nonlinear_viscous_u = " nu*(2*dx(u)*dx(ln_rho1) + dx(w)*dz(ln_rho1) + dz(ln_rho1)*u_z   - 2/3*dx(ln_rho1)*(dx(u)+w_z))"
        
        self.thermal_diff   = " Cv_inv*chi*(dx(dx(T1)) - dz(Q_z) - Q_z*del_ln_rho0)"
        self.nonlinear_thermal_diff = "Cv_inv*chi*(dx(T1)*dx(ln_rho1) - Q_z*dz(ln_rho1))"

        self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2/3*(dx(u)+w_z)**2)"
        
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) + Q_z = 0")
        
        self.problem.add_equation(("(scale)*( dt(w) - Q_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - " + self.viscous_term_w + ") = "
                                   "(scale)*(-T1*dz(ln_rho1) - u*dx(w) - w*w_z + "+self.nonlinear_viscous_w+")"))

        self.problem.add_equation(("(scale)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - " + self.viscous_term_u + ") = "
                                   "(scale)*(-T1*dx(ln_rho1) - u*dx(u) - w*u_z + "+self.nonlinear_viscous_u+")"))

        self.problem.add_equation(("(scale)*( dt(ln_rho1)   + w*del_ln_rho0 + dx(u) + w_z ) = "
                                   "(scale)*(-u*dx(ln_rho1) - w*dz(ln_rho1))"))

        # here we have assumed chi = constant in both rho and radius
        self.problem.add_equation(("(scale)*( dt(T1)   + w*del_T0 + (gamma-1)*T0*(dx(u) + w_z) - " + self.thermal_diff+") = "
                                   "(scale)*(-u*dx(T1) + w*Q_z   - (gamma-1)*T1*(dx(u) + w_z) + "
                                   +self.nonlinear_thermal_diff+" + "+self.viscous_heating+" )")) 
        
        logger.info("using nonlinear EOS for entropy")
        # non-linear EOS for s, where we've subtracted off
        # Cv_inv*∇s0 =  del_T0/(T0 + T1) - (gamma-1)*del_ln_rho0
        self.problem.add_equation(("(scale)*(Cv_inv*s - T1/T0 + (gamma-1)*ln_rho1) = "
                                   "(scale)*(log(1+T1/T0) - T1/T0)"))

    def set_BC(self, fixed_flux=False):
        if fixed_flux:
            self.problem.add_bc( "left(Q_z) = 0")
            self.problem.add_bc("right(Q_z) = 0")
        else:
            self.problem.add_bc( "left(s) = 0")
            self.problem.add_bc("right(s) = 0")
            
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")

    def _set_subs(self):
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'

        self.problem.substitutions['KE'] = 'rho_full*(u**2+w**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'rho_full*Cv*T1+rho_fluc*Cv*T0'
        self.problem.substitutions['P'] = 'rho_full*(T1+T0)'
        self.problem.substitutions['P_fluc'] = 'rho_full*T1+rho_fluc*T0'
        self.problem.substitutions['h'] = 'IE + P'
        self.problem.substitutions['h_fluc'] = 'IE_fluc + P_fluc'
        self.problem.substitutions['u_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['w_rms'] = 'sqrt(w*w)'
        self.problem.substitutions['Re_rms'] = 'sqrt(u**2+w**2)*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'sqrt(u**2+w**2)*Lz/chi'
        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

    def initialize_output(self, solver, data_dir, **kwargs):
        analysis_tasks = []
        self.analysis_tasks = analysis_tasks
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("s", name="s")
        analysis_slice.add_task("s - plane_avg(s)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")
        analysis_tasks.append(analysis_slice)

        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(IE)", name="IE")
        analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
        analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
        analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
        analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_profile.add_task("plane_avg(w*(KE))", name="KE_flux_z")
        analysis_profile.add_task("plane_avg(w*(PE))", name="PE_flux_z")
        analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
        analysis_profile.add_task("plane_avg(w*(P))",  name="P_flux_z")
        analysis_profile.add_task("plane_avg(w*(h))",  name="enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_tasks.append(analysis_profile)

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=20, parallel=False, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(PE)", name="PE")
        analysis_scalar.add_task("vol_avg(IE)", name="IE")
        analysis_scalar.add_task("vol_avg(PE_fluc)", name="PE_fluc")
        analysis_scalar.add_task("vol_avg(IE_fluc)", name="IE_fluc")
        analysis_scalar.add_task("vol_avg(KE + PE + IE)", name="TE")
        analysis_scalar.add_task("vol_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        analysis_scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
        analysis_tasks.append(analysis_scalar)

        return self.analysis_tasks
