import numpy as np
import os
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__.split('.')[-1])

try:
    from equations import *
    from atmospheres import *
except:
    from sys import path
    path.insert(0, './stratified_dynamics')
    from stratified_dynamics.equations import *
    from stratified_dynamics.atmospheres import *

class FC_multitrope(FC_equations_2d, Multitrope):
    def __init__(self, dimensions=2, *args, **kwargs):
        super(FC_multitrope, self).__init__(dimensions=dimensions) 
        Multitrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_multitrope,self).set_equations(*args, **kwargs)

    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        if self.stable_bottom:
            # set taper safely in the mid-CZ to avoid leakage of coeffs into RZ chebyshev coeffs
            taper = 1-self.match_Phi_multi(z_dealias, center=(self.Lz_rz+self.Lz_cz/2), width=0.1*self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias-self.Lz_rz)/self.Lz_cz)
        else:
            taper = self.match_Phi_multi(z_dealias, center=self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias)/self.Lz_cz)

        # this will broadcast power back into relatively high Tz; consider widening taper.
        self.T_IC['g'] = self.epsilon*A0*noise['g']*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0
        
        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))

class FC_multitrope_rxn(FC_equations_rxn, Multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_multitrope_rxn, self).__init__() 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_multitrope_rxn,self).set_equations(*args, **kwargs)

    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        self.f_IC = solver.state['f']
        self.c_IC = solver.state['c']
        self.c_IC.set_scales(self.domain.dealias, keep_data=True)
        self.f_IC.set_scales(self.domain.dealias, keep_data=True)

        
        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        if self.stable_bottom:
            # set taper safely in the mid-CZ to avoid leakage of coeffs into RZ chebyshev coeffs
            taper = 1-self.match_Phi(z_dealias, center=(self.Lz_rz+self.Lz_cz/2), width=0.1*self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias-self.Lz_rz)/self.Lz_cz)
        else:
            taper = self.match_Phi(z_dealias, center=self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias)/self.Lz_cz)

        # this will broadcast power back into relatively high Tz; consider widening taper.
        self.T_IC['g'] = self.epsilon*A0*noise['g']*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0
        
        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))

        # we need to add c and f ICs here
        # for now we just hijack the taper function, which puts the quench point in
        # the middle of the CZ; in time the location and width of this should be
        # a kwarg so we can control this profile.
        #
        # Right now we're setting it to be 1 in wave region and adjoining CZ,
        # and zero in the other part of the CZ (furthest from wave region)
        #
        # This is a hack to get things up and running
        c0 = 1
        self.c_IC['g'] = c0*(1-taper)
        self.f_IC['g'] = self.c_IC['g']

class FC_multitrope_2d_kappa_mu(FC_equations_2d_kappa_mu, Multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_multitrope_2d_kappa, self).__init__() 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        if self.stable_bottom:
            # set taper safely in the mid-CZ to avoid leakage of coeffs into RZ chebyshev coeffs
            taper = 1-self.match_Phi(z_dealias, center=(self.Lz_rz+self.Lz_cz/2), width=0.1*self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias-self.Lz_rz)/self.Lz_cz)
        else:
            taper = self.match_Phi(z_dealias, center=self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias)/self.Lz_cz)

        # this will broadcast power back into relatively high Tz; consider widening taper.
        self.T_IC['g'] = self.epsilon*A0*noise['g']*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0
        
        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))


    def initialize_output(self, solver, data_dir, *args, **kwargs):
        super(FC_multitrope_2d_kappa, self).initialize_output(solver, data_dir, *args, **kwargs)

        #This creates an output file that contains all of the useful atmospheric info at the beginning of the run
        import h5py
        import os
        from dedalus.core.field import Field
        dir = data_dir + '/atmosphere/'
        file = dir + 'atmosphere.h5'
        if self.domain.dist.comm_cart.rank == 0:
            if not os.path.exists('{:s}'.format(dir)):
                os.mkdir('{:s}'.format(dir))
        if self.domain.dist.comm_cart.rank == 0:
            f = h5py.File('{:s}'.format(file), 'w')
        key_set = list(self.problem.parameters.keys())
        logger.debug("Outputing atmosphere parameters for {}".format(key_set))
        for key in key_set:
            if 'scale' in key:
                continue
            if type(self.problem.parameters[key]) == Field:
                field_key = True
                self.problem.parameters[key].set_scales(1, keep_data=True)
            else:
                field_key = False
            if field_key:
                try:
                    array = self.problem.parameters[key]['g'][0,:]
                except:
                    logger.error("key error on atmosphere output {}".format(key))
                        
                this_chunk      = np.zeros(self.nz)
                global_chunk    = np.zeros(self.nz)
                n_per_cpu       = int(self.nz/self.domain.dist.comm_cart.size)
                i_chunk_0 = self.domain.dist.comm_cart.rank*(n_per_cpu)
                i_chunk_1 = (self.domain.dist.comm_cart.rank+1)*(n_per_cpu)
                this_chunk[i_chunk_0:i_chunk_1] = array
                self.domain.dist.comm_cart.Allreduce(this_chunk, global_chunk, op=MPI.SUM)
                if self.domain.dist.comm_cart.rank == 0:
                    f[key] = global_chunk                        
            elif self.domain.dist.comm_cart.rank == 0:
                f[key] = self.problem.parameters[key]
                
        if self.domain.dist.comm_cart.rank == 0:
            f['dimensions']     = 2
            f['nx']             = self.nx
            f['nz']             = self.nz
            f['z']              = self.domain.grid(axis=-1, scales=1)
            f['m_ad']           = self.m_ad
            f['m']              = self.m_ad - self.epsilon
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f.close()
            
        return self.analysis_tasks

class FC_MHD_multitrope(FC_MHD_equations, Multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_MHD_multitrope, self).__init__() 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_MHD_multitrope, self).set_equations(*args, **kwargs)
    
    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        if self.stable_bottom:
            # set taper safely in the mid-CZ to avoid leakage of coeffs into RZ chebyshev coeffs
            taper = 1-self.match_Phi(z_dealias, center=(self.Lz_rz+self.Lz_cz/2), width=0.1*self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias-self.Lz_rz)/self.Lz_cz)
        else:
            taper = self.match_Phi(z_dealias, center=self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias)/self.Lz_cz)

        # this will broadcast power back into relatively high Tz; consider widening taper.
        self.T_IC['g'] = self.epsilon*A0*noise['g']*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0

        self.Bx_IC = solver.state['Bx']
        self.Ay_IC = solver.state['Ay']

        # not in HS balance
        B0 = 1
        self.Bx_IC.set_scales(self.domain.dealias, keep_data=True)

        self.Bx_IC['g'] = A0*B0*np.cos(np.pi*self.z_dealias/self.Lz)*taper
        self.Bx_IC.antidifferentiate('z',('left',0), out=self.Ay_IC)
        self.Ay_IC['g'] *= -1
        
        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))

            
class FC_MHD_multitrope_guidefield_2d(FC_equations_MHD_guidefield_2d, Multitrope):
    def __init__(self, *args, dimensions=2, **kwargs):
        super(FC_MHD_multitrope_guidefield_2d, self).__init__(dimensions=dimensions) 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super().set_equations(*args, **kwargs)
    
    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        if self.stable_bottom:
            # set taper safely in the mid-CZ to avoid leakage of coeffs into RZ chebyshev coeffs
            taper = 1-self.match_Phi_multi(z_dealias, center=(self.Lz_rz+self.Lz_cz/2), width=0.1*self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias-self.Lz_rz)/self.Lz_cz)
        else:
            taper = self.match_Phi_multi(z_dealias, center=self.Lz_cz)
            taper *= np.sin(np.pi*(z_dealias)/self.Lz_cz)

        # this will broadcast power back into relatively high Tz; consider widening taper.
        self.T_IC['g'] = self.epsilon*A0*noise['g']*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0

        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))

