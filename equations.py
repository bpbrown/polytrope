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

        self.del_s0_factor = - self.epsilon/self.gamma
        self.del_ln_rho_factor = -self.poly_n

        self.del_ln_rho0 = self.del_ln_rho_factor/(self.z0 - self.z)
        self.del_s0 = self.del_s0_factor/(self.z0 - self.z)
        
        self.g = self.poly_n + 1

        logger.info("polytropic atmosphere parameters:")
        logger.info("poly_n = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_n, self.epsilon, self.gamma))
        logger.info("Lz = {:g}".format(self.Lz))

        self.min_BV_time = np.min(np.sqrt(np.abs(self.g*self.del_s0)))
        self.freefall_time = np.sqrt(self.Lz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))

        logger.info("atmospheric timescales:")
        logger.info("min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,self.freefall_time,self.buoyancy_time))
        
    def set_anelastic_problem(self, Rayleigh, Prandtl):

        logger.info("problem parameters:")
        logger.info("Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        
        delta_s = self.del_s0_factor*np.log(self.z0)

        # take constant nu, chi
        nu = np.sqrt(Prandtl*(self.Lz**3*np.abs(delta_s)*self.g)/Rayleigh)
        chi = nu/Prandtl

        logger.info("nu = {:g}, chi = {:g}".format(nu, chi))
                
        self.problem = ParsedProblem( axis_names=['x', 'z'],
                                field_names=['u','u_z','w','w_z','s', 's_z', 'pomega'],
                                param_names=['nu', 'chi', 'del_ln_rho0', 'del_s0', 'g', 'z0'])
        
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(s) - s_z = 0")

        self.problem.add_equation("        dt(w)  - nu*(dx(dx(w)) + dz(w_z)) + dz(pomega) - s*g  =        -(u*dx(w) + w*w_z)")
        self.problem.add_equation("        dt(u)  - nu*(dx(dx(u)) + dz(u_z)) + dx(pomega)        =        -(u*dx(u) + w*u_z)")
        self.problem.add_equation("(z-z0)*(dt(s) - chi*(dx(dx(s)) + dz(s_z)) + w*del_s0)         = -(z-z0)*(u*dx(s) + w*s_z)")
        self.problem.add_equation("(z-z0)*(dx(u) + w_z + w*del_ln_rho0) = 0")
            
        self.problem.parameters['nu']  = nu
        self.problem.parameters['chi'] = chi
        self.problem.parameters['del_ln_rho0']  = self.del_ln_rho0
        self.problem.parameters['del_s0']  = self.del_s0

        self.problem.parameters['g']  = self.g
        self.problem.parameters['z0']  = self.z0

        self.problem.add_left_bc( "s = 0")
        self.problem.add_right_bc("s = 0")
        self.problem.add_left_bc( "u = 0")
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0", condition="dx != 0")
        self.problem.add_left_bc( "pomega = 0", condition="dx == 0")
        self.problem.add_right_bc("w = 0")

        self.problem.expand(self.domain, order=2)

        return self.problem
    
