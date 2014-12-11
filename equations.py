import numpy as np
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class polytrope:
    def __init__(self, domain, epsilon=1e-4, gamma=5/3):
        self.domain = domain
        self._set_atmosphere(epsilon, gamma)
        
    def _set_atmosphere(self, epsilon, gamma):
        # polytropic atmosphere characteristics
        self.epsilon = epsilon
        self.gamma = gamma
        self.poly_n = 1/(gamma-1) - epsilon

        self.z = self.domain.bases[-1].grid # need to access globally-sized z-basis
        self.Lz = np.max(self.z)-np.min(self.z) # global size of Lz
        self.z0 = 1. + self.Lz

        self.del_ln_rho_factor = -self.poly_n
        self.del_ln_rho0 = self.del_ln_rho_factor/(self.z0 - self.z)

        self.del_s0_factor = - self.epsilon/self.gamma
        self.del_s0 = self.del_s0_factor/(self.z0 - self.z)

        self.delta_s = self.del_s0_factor*np.log(self.z0)
        
        self.T0 = self.z0 - self.z
        self.del_T0 = -1

        
        self.g = self.poly_n + 1

        self.x = self.domain.bases[0].grid
        nx = self.x.shape[0]
        self.Lx = np.max(self.x)-np.min(self.x) # global size of Lx

        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_n = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_n, self.epsilon, self.gamma))
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        
        logger.info("   density scale heights = {:g}".format(np.log(self.Lz**self.poly_n)))
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format((self.z0-self.Lz)/self.poly_n, (self.z0)/self.poly_n))
        logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(((self.z0-self.Lz)/self.poly_n)/(self.Lx/nx), ((self.z0)/self.poly_n)/(self.Lx/nx)))

        self.min_BV_time = np.min(np.sqrt(np.abs(self.g*self.del_s0)))
        self.freefall_time = np.sqrt(self.Lz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))

        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,self.freefall_time,self.buoyancy_time))

        # processor local values
        x = self.domain.grid(0)
        z = self.domain.grid(1)
        self.T0_local = self.domain.new_field()
        self.T0_local['g'] = self.z0 - z
        self.rho0_local = self.domain.new_field()
        self.rho0_local['g'] = (self.z0 - z)**self.poly_n

    def _calc_diffusivity(self, Rayleigh, Prandtl):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        
        # take constant nu, chi
        nu = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s)*self.g)/Rayleigh)
        chi = nu/Prandtl

        logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        
        return nu, chi
    
    def set_anelastic_problem(self, Rayleigh, Prandtl):

        nu, chi = self._calc_diffusivity(Rayleigh, Prandtl)
                
        self.problem = ParsedProblem( axis_names=['x', 'z'],
                                field_names=['u','u_z','w','w_z','s', 'Q_z', 'pomega'],
                                param_names=['nu', 'chi', 'del_ln_rho0', 'del_s0', 'T0', 'T0_local', 'g', 'z0'])
        
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")

        # Lecoanet et al 2014, ApJ, eqns D23-D31
        self.viscous_term_z = " nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_x = " nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z)))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"
        
        self.problem.add_equation("(z-z0)  * (dt(w)  - "+self.viscous_term_z+" + dz(pomega) - s*g) = -(z-z0)  * (u*dx(w) + w*w_z)")
        self.problem.add_equation("(z-z0)  * (dt(u)  - "+self.viscous_term_x+" + dx(pomega)      ) = -(z-z0)  * (u*dx(u) + w*u_z)")
        self.problem.add_equation("(z-z0)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(z-z0)**2*(u*dx(s) + w*dz(s))")
        # seems to not help speed --v
        # self.problem.add_equation("(z-z0)**2*(dt(s)  - "+self.thermal_diff  +" + w*del_s0        ) = -(z-z0)**2*(u*dx(s) + w*(-Q_z/T0_local))")

        self.problem.add_equation("(z-z0)*(dx(u) + w_z + w*del_ln_rho0) = 0")
            
        self.problem.parameters['nu']  = nu
        self.problem.parameters['chi'] = chi
        self.problem.parameters['del_ln_rho0']  = self.del_ln_rho0
        self.problem.parameters['del_s0']  = self.del_s0
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['T0_local'] = self.T0_local

        self.problem.parameters['g']  = self.g
        self.problem.parameters['z0']  = self.z0

        self.problem.add_left_bc( "s = 0")
        self.problem.add_right_bc("s = 0")
        self.problem.add_left_bc( "u = 0")
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0", condition="dx != 0")
        self.problem.add_left_bc( "pomega = 0", condition="dx == 0")
        self.problem.add_right_bc("w = 0")

        self.problem.expand(self.domain, order=3)

        return self.problem


    def set_FC_problem(self, Rayleigh, Prandtl):

        nu, chi = self._calc_diffusivity(Rayleigh, Prandtl)
                
        self.problem = ParsedProblem(axis_names=['x', 'z'],
                                     field_names=['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1', 's'], #'ln_rho1_z', 's', 's_z'],
                                     param_names=['T0', 'del_T0', 'del_ln_rho0', 'nu', 'chi', 'gamma', 'Cv_inv', 'z0', 'T0_local'])

        # here, nu and chi are constants
        viscous_term_w = (" - nu*(dx(dx(w)) + dz(w_z)) - nu/3.*(dx(u_z)   + dz(w_z)) " 
                          " - 2*nu*w_z*del_ln_rho0 + 2/3*nu*del_ln_rho0*(dx(u) + w_z) ")
        
        viscous_term_u = (" - nu*(dx(dx(u)) + dz(u_z)) - nu/3.*(dx(dx(u)) + dx(w_z)) "
                          " - nu*dx(w)*del_ln_rho0 - nu*del_ln_rho0*u_z ")

        nonlinear_viscous_w = (" + nu*u_z*dx(ln_rho1) + 2*nu*w_z*dz(ln_rho1) "
                               " + nu*dx(ln_rho1)*dx(w) "
                               " - 2/3*nu*dz(ln_rho1)*(dx(u)+w_z) ")
        
        nonlinear_viscous_u = (" + 2*nu*dx(u)*dx(ln_rho1) + nu*dx(w)*dz(ln_rho1) "
                               " + nu*dz(ln_rho1)*u_z "
                               " - 2/3*nu*dx(ln_rho1)*(dx(u)+w_z) ")

        viscous_heating_term = ""

        self.viscous_term_w = " - nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z))"
        self.viscous_term_u = " - nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z)))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"

        
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
        #self.problem.add_equation("dz(ln_rho1) - ln_rho1_z = 0")
        #self.problem.add_equation("dz(s) - s_z = 0")

        
        self.problem.add_equation(("(z0-z)*(dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 " + self.viscous_term_w + ") = "
                                   "(z0-z)*(-T1*dz(ln_rho1) - u*dx(w) - w*w_z " + nonlinear_viscous_w +")"))

        self.problem.add_equation(("(z0-z)*(dt(u) + dx(T1) + T0*dx(ln_rho1)                    " + self.viscous_term_u + ") = "
                                   "(z0-z)*(-T1*dx(ln_rho1) - u*dx(u) - w*u_z " + nonlinear_viscous_u+")"))

        self.problem.add_equation(("(z0-z)*(dt(ln_rho1) + w*del_ln_rho0 + dx(u) + w_z ) = "
                                   "(z0-z)*(-u*dx(ln_rho1) -w*dz(ln_rho1) )"))

        # here we have assumed chi = constant in both rho and radius
        self.problem.add_equation(("(z0-z)*(dt(T1) + w*del_T0 + (gamma-1)*T0*(dx(u) + w_z) - Cv_inv*chi*(dx(dx(T1)) + dz(T1_z)) - Cv_inv*chi*T1_z*del_ln_rho0 ) = "
                                   "(z0-z)*(-u*dx(T1) - w*T1_z - (gamma-1)*T1*(dx(u) + w_z) + Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1)) )")) #+ " + vicous_heating
        
        logger.info("using non-differential, nonlinear EOS for entropy")
        # non-linear EOS for s, where we've subtracted off
        # Cv_inv*∇s0 =  del_T0/(T0 + T1) - (gamma-1)*del_ln_rho0
        self.problem.add_equation(("(z0-z)*(Cv_inv*s - T1/T0 + (gamma-1)*ln_rho1) = "
                                   "(z0-z)*(log(1+T1/T0_local) - T1/T0_local)"))

        self.problem.add_left_bc( "s = 0")
        self.problem.add_right_bc("s = 0")
        self.problem.add_left_bc( "u = 0")
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0")
        self.problem.add_right_bc("w = 0")

        self.problem.parameters['nu']  = nu
        self.problem.parameters['chi'] = chi
        self.problem.parameters['del_ln_rho0']  = self.del_ln_rho0
        self.problem.parameters['del_T0'] = self.del_T0
        self.problem.parameters['T0']  = self.T0
        
        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['z0']  = self.z0

        # Local variables for RHS
        self.problem.parameters['T0_local']  = self.T0_local

        self.problem.expand(self.domain, order=3)

        return self.problem
