import numpy as np
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class anelastic_polytrope:
    def __init__(self, domain):
        self.domain = domain
        
    def set_problem(self, Rayleigh, Prandtl, epsilon=1e-4, gamma=5/3):
        
        logger.info("Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        
        poly_n = 1/(gamma-1) - epsilon
        
        del_s0_factor = - epsilon/gamma
        del_ln_rho_factor = -poly_n

        z = self.domain.bases[-1].grid # need to access globally-sized z-basis
        Lz = np.max(z)-np.min(z) # global size of Lz
        
        z0 = 1. + Lz

        del_ln_rho0 = del_ln_rho_factor/(z0 - z)
        del_s0 = del_s0_factor/(z0 - z)
        
        g_atmosphere = poly_n + 1
        delta_s = del_s0_factor*np.log(z0)

        # take constant nu, chi
        nu = np.sqrt(Prandtl*(Lz**3*np.abs(delta_s)*g_atmosphere)/Rayleigh)
        chi = nu/Prandtl

        logger.info("Lz = {:g}".format(Lz))
        logger.info("nu = {:g}, chi = {:g}".format(nu, chi))
        logger.info("poly_n = {:g}, epsilon = {:g}, gamma = {:g}".format(poly_n, epsilon, gamma))
                
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
        self.problem.parameters['del_ln_rho0']  = del_ln_rho0
        self.problem.parameters['del_s0']  = del_s0

        self.problem.parameters['g']  = g_atmosphere
        self.problem.parameters['z0']  = z0

        self.problem.add_left_bc( "s = 0")
        self.problem.add_right_bc("s = 0")
        self.problem.add_left_bc( "u = 0")
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0", condition="dx != 0")
        self.problem.add_left_bc( "pomega = 0", condition="dx == 0")
        self.problem.add_right_bc("w = 0")

        self.problem.expand(self.domain, order=2)

        return self.problem
    
