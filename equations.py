import numpy as np
import os
from mpi4py import MPI

from dedalus2 import public as de

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class polytrope:
    def __init__(self, nx=256, nz=128, Lx=30, Lz=10, epsilon=1e-4, gamma=5/3):
        
        x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        z_basis = de.Chebyshev('z', nz, interval=[0., Lz], dealias=3/2)
        self.domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
                
        self._set_atmosphere(epsilon, gamma)
        
    def _set_atmosphere(self, epsilon, gamma):
        # polytropic atmosphere characteristics
        self.epsilon = epsilon
        self.gamma = gamma
        self.poly_n = 1/(gamma-1) - epsilon

        self.x = self.domain.grids(0)
        self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
        self.nx = self.domain.bases[0].coeff_size

        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size
        
        self.z0 = 1. + self.Lz

        self.del_ln_rho_factor = -self.poly_n
        
        self.del_ln_rho0 = self.domain.new_field()
        self.del_ln_rho0.meta['x']['constant'] = True
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)

        self.del_s0_factor = - self.epsilon/self.gamma

        self.del_s0 = self.domain.new_field()
        self.del_s0.meta['x']['constant'] = True
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)

        self.delta_s = self.del_s0_factor*np.log(self.z0)

        self.T0 = self.domain.new_field()
        self.T0.meta['x']['constant'] = True
        self.T0['g'] = self.z0 - self.z
        
        self.del_T0 = -1

        self.rho0 = self.domain.new_field()
        self.rho0.meta['x']['constant'] = True
        self.rho0['g'] = (self.z0 - self.z)**self.poly_n

        
        self.scale = self.domain.new_field()
        self.scale.meta['x']['constant'] = True
        self.scale['g'] = self.z0 - self.z

        self.g = self.poly_n + 1

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

    def _set_parameters(self):
        self.problem.parameters['nu'] = self.nu
        self.problem.parameters['chi'] = self.chi
        
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['del_T0'] = self.del_T0
        
        self.problem.parameters['rho0'] = self.rho0
        self.problem.parameters['del_ln_rho0'] = self.del_ln_rho0
        
        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['gamma'] = self.gamma

        self.problem.parameters['del_s0'] = self.del_s0

        self.problem.parameters['g']  = self.g
        
        self.problem.parameters['scale'] = self.scale
        
    def set_anelastic_problem(self, Rayleigh, Prandtl):
                
        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'], cutoff=1e-6)

        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()

        self.problem.parameters['one_third'] = 1/3
        
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")

        # Lecoanet et al 2014, ApJ, eqns D23-D31
        self.viscous_term_w = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + one_third*(dx(u_z) + dz(w_z)) - 2*one_third*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + one_third*(dx(dx(u)) + dx(w_z)))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"
        
        self.problem.add_equation("(scale)  * (dt(w)  - "+self.viscous_term_w+" + dz(pomega) - s*g) = -(scale)  * (u*dx(w) + w*w_z)")
        self.problem.add_equation("(scale)  * (dt(u)  - "+self.viscous_term_u+" + dx(pomega)      ) = -(scale)  * (u*dx(u) + w*u_z)")
        self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*dz(s))")
        # seems to not help speed --v
        # self.problem.add_equation("(scale)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(scale)**2*(u*dx(s) + w*(-Q_z/T0_local))")

        self.problem.add_equation("(scale)*(dx(u) + w_z + w*del_ln_rho0) = 0")
            

        self.problem.add_bc( "left(s) = 0")
        self.problem.add_bc("right(s) = 0")
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0", condition="nx != 0")
        self.problem.add_bc( "left(pomega) = 0", condition="nx == 0")
        self.problem.add_bc("right(w) = 0")
        
        return self.problem


    def set_FC_problem(self, Rayleigh, Prandtl):

        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','T1', 'Q_z', 'ln_rho1', 's'], cutoff=1e-12)
        
        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()

        self.problem.parameters['one_third'] = 1/3

        # here, nu and chi are constants
        viscous_term_w = (" - nu*(dx(dx(w)) + dz(w_z)) - nu*one_third.*(dx(u_z)   + dz(w_z)) " 
                          " - 2*nu*w_z*del_ln_rho0 + 2*one_third*nu*del_ln_rho0*(dx(u) + w_z) ")
        
        viscous_term_u = (" - nu*(dx(dx(u)) + dz(u_z)) - nu*one_third.*(dx(dx(u)) + dx(w_z)) "
                          " - nu*dx(w)*del_ln_rho0 - nu*del_ln_rho0*u_z ")

        nonlinear_viscous_w = (" + nu*u_z*dx(ln_rho1) + 2*nu*w_z*dz(ln_rho1) "
                               " + nu*dx(ln_rho1)*dx(w) "
                               " - 2*one_third*nu*dz(ln_rho1)*(dx(u)+w_z) ")
        
        nonlinear_viscous_u = (" + 2*nu*dx(u)*dx(ln_rho1) + nu*dx(w)*dz(ln_rho1) "
                               " + nu*dz(ln_rho1)*u_z "
                               " - 2*one_third*nu*dx(ln_rho1)*(dx(u)+w_z) ")

        viscous_heating_term = ""

        
        self.viscous_term_w = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + one_third*(dx(u_z) + dz(w_z)) - 2*one_third*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + one_third*(dx(dx(u)) + dx(w_z)))"

        self.nonlinear_viscous_w = " nu*(    u_z*dx(ln_rho1) + 2*w_z*dz(ln_rho1) + dx(ln_rho1)*dx(w) - 2*one_third*dz(ln_rho1)*(dx(u)+w_z))"
        self.nonlinear_viscous_u = " nu*(2*dx(u)*dx(ln_rho1) + dx(w)*dz(ln_rho1) + dz(ln_rho1)*u_z   - 2*one_third*dx(ln_rho1)*(dx(u)+w_z))"
        
        self.thermal_diff   = " Cv_inv*chi*(dx(dx(T1)) - dz(Q_z) - Q_z*del_ln_rho0)"
        self.nonlinear_thermal_diff = "Cv_inv*chi*(dx(T1)*dx(ln_rho1) - Q_z*dz(ln_rho1))"

        self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2*one_third*(dx(u)+w_z)**2)"
        
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) + Q_z = 0")
        
        self.problem.add_equation(("(scale)*( dt(w) + Q_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - " + self.viscous_term_w + ") = "
                                   "(scale)*(-T1*dz(ln_rho1) - u*dx(w) - w*w_z + "+self.nonlinear_viscous_w+")"))

        self.problem.add_equation(("(scale)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - " + self.viscous_term_u + ") = "
                                   "(scale)*(-T1*dx(ln_rho1) - u*dx(u) - w*u_z + "+self.nonlinear_viscous_u+")"))

        self.problem.add_equation(("(scale)*( dt(ln_rho1)   + w*del_ln_rho0 + dx(u) + w_z ) = "
                                   "(scale)*(-u*dx(ln_rho1) - w*dz(ln_rho1))"))

        # here we have assumed chi = constant in both rho and radius
        self.problem.add_equation(("(scale)*( dt(T1)   + w*del_T0 + (gamma-1)*T0*(dx(u) + w_z) - " + self.thermal_diff+") = "
                                   "(scale)*(-u*dx(T1) - w*Q_z   - (gamma-1)*T1*(dx(u) + w_z) + "
                                   +self.nonlinear_thermal_diff+" + "+self.viscous_heating+" )")) 
        
        logger.info("using nonlinear EOS for entropy")
        # non-linear EOS for s, where we've subtracted off
        # Cv_inv*∇s0 =  del_T0/(T0 + T1) - (gamma-1)*del_ln_rho0
        self.problem.add_equation(("(scale)*(Cv_inv*s - T1/T0 + (gamma-1)*ln_rho1) = "
                                   "(scale)*(log(1+T1/T0) - T1/T0)"))

        self.problem.add_bc( "left(s) = 0")
        self.problem.add_bc("right(s) = 0")
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")

        return self.problem
