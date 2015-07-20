import numpy as np
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de


class atmosphere:
    def __init__(self, gamma=5/3, verbose=False, **kwargs):
        self._set_domain(**kwargs)
        
        self.gamma = gamma
        self.make_plots = verbose
        
    def _set_domain(self, nx=256, Lx=4, nz=128, Lz=1, grid_dtype=np.float64, comm=MPI.COMM_WORLD):
        x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        z_basis = de.Chebyshev('z', nz, interval=[0., Lz], dealias=3/2)
        self.domain = de.Domain([x_basis, z_basis], grid_dtype=grid_dtype, comm=comm)
        
        self.x = self.domain.grid(0)
        self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
        self.nx = self.domain.bases[0].coeff_size
        self.delta_x = self.Lx/self.nx
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        
    def _new_ncc(self):
        field = self.domain.new_field()
        field.meta['x']['constant'] = True
        return field

    def get_problem(self):
        return self.problem

    def _set_atmosphere(self):
        self.necessary_quantities = OrderedDict()

        self.phi = self._new_ncc()
        self.necessary_quantities['phi'] = self.phi

        self.del_ln_rho0 = self._new_ncc()
        self.rho0 = self._new_ncc()
        self.necessary_quantities['del_ln_rho0'] = self.del_ln_rho0
        self.necessary_quantities['rho0'] = self.rho0

        self.del_s0 = self._new_ncc()
        self.necessary_quantities['del_s0'] = self.del_s0
        
        self.T0_zz = self._new_ncc()
        self.del_T0 = self._new_ncc()
        self.T0 = self._new_ncc()
        self.necessary_quantities['T0_zz'] = self.T0_zz
        self.necessary_quantities['del_T0'] = self.del_T0
        self.necessary_quantities['T0'] = self.T0

        self.del_P0 = self._new_ncc()
        self.P0 = self._new_ncc()
        self.necessary_quantities['del_P0'] = self.del_P0
        self.necessary_quantities['P0'] = self.P0

        self.nu = self._new_ncc()
        self.chi = self._new_ncc()
        self.del_chi = self._new_ncc()
        self.necessary_quantities['nu'] = self.nu
        self.necessary_quantities['chi'] = self.chi
        self.necessary_quantities['del_chi'] = self.del_chi

        self.scale = self._new_ncc()
        self.necessary_quantities['scale'] = self.scale


    def _set_parameters(self):
        '''
        Basic parameters needed for any stratified atmosphere.
        '''
        self.problem.parameters['Lz'] = self.Lz
        self.problem.parameters['Lx'] = self.Lx

        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['Cv'] = 1/(self.gamma-1)

        # the following quantities must be calculated and are missing
        # from the atmosphere stub.

        # thermodynamic quantities
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['del_T0'] = self.del_T0
        self.problem.parameters['T0_zz'] = self.T0_zz
        
        self.problem.parameters['rho0'] = self.rho0
        self.problem.parameters['del_ln_rho0'] = self.del_ln_rho0
                    
        self.problem.parameters['del_s0'] = self.del_s0

        # gravity
        self.problem.parameters['g']  = self.g
        self.problem.parameters['phi']  = self.phi

        # scaling factor to reduce NCC bandwidth of all equations
        self.problem.parameters['scale'] = self.scale

        # diffusivities
        self.problem.parameters['nu'] = self.nu
        self.problem.parameters['chi'] = self.chi

        if not self.constant_diffusivities:
            self.problem.parameters['del_chi'] = self.del_chi

    def plot_atmosphere(self):
        fig_atm = plt.figure()
        axT = fig_atm.add_subplot(2,2,1)
        axT.plot(self.z[0,:], self.T0['g'][0,:])
        axT.set_ylabel('T0')
        axP = fig_atm.add_subplot(2,2,2)
        axP.plot(self.z[0,:], self.P0['g'][0,:])
        axP.set_ylabel('P0')
        axR = fig_atm.add_subplot(2,2,3)
        axR.plot(self.z[0,:], self.rho0['g'][0,:])
        axR.set_ylabel(r'$\rho0$')
        axS = fig_atm.add_subplot(2,2,4)
        mask = (self.del_s0['g'][0,:]>0)
        axS.semilogy(self.z[0,mask], self.del_s0['g'][0,mask])
        mask = (self.del_s0['g'][0,:]<0)
        axS.semilogy(self.z[0,mask], np.abs(self.del_s0['g'][0,mask]), linestyle='dashed', color='red')
        
        axS.set_ylabel(r'$\nabla s0$')
        fig_atm.savefig("atmosphere_quantities_p{}.png".format(self.domain.distributor.rank), dpi=300)

        for key in self.necessary_quantities:
            fig_q = plt.figure()
            ax = fig_q.add_subplot(1,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            ax.set_xlabel('z')
            ax.set_ylabel(key)
            fig_q.savefig("atmosphere_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)
            
    def check_that_atmosphere_is_set(self):
        for key in self.necessary_quantities:
            quantity = self.necessary_quantities[key]['g']
            quantity_set = quantity.any()
            if not quantity_set:
                logger.info("WARNING: atmosphere {} is all zeros".format(key))
        
    def test_hydrostatic_balance(self, make_plots=False):
        # error in hydrostatic balance diagnostic
        HS_balance = self.del_P0['g']+self.g*self.rho0['g']
        relative_error = HS_balance/self.del_P0['g']
        
        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(self.z[0,:], self.del_P0['g'][0,:])
            ax1.plot(self.z[0,:], -self.g*self.rho0['g'][0,:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig("atmosphere_HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)

        max_rel_err = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error)), op=MPI.MAX)
        logger.info('max error in HS balance: {}'.format(max_rel_err))

    def check_atmosphere(self):
        if self.make_plots:
            self.plot_atmosphere()
        self.test_hydrostatic_balance()
        self.check_that_atmosphere_is_set()


class multi_layer_atmosphere(atmosphere):
    def __init__(self, *args, **kwargs):
        super(multi_layer_atmosphere, self).__init__(*args, **kwargs)
        
    def _set_domain(self, nx=256, Lx=4, nz=[128, 128], Lz=[1,1], grid_dtype=np.float64, comm=MPI.COMM_WORLD):
        '''
        Specify 2-D domain, with compund basis in z-direction.

        First entries in nz, Lz are the bottom entries (build upwards).
        '''
        if len(nz) != len(Lz):
            logger.error("nz {} has different number of elements from Lz {}".format(nz, Lz))
            raise
                         
        x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)

        logger.info("Setting compound basis in vertical (z) direction")
        z_basis_list = []
        Lz_interface = 0.
        for iz, nz_i in enumerate(nz):
            Lz_top = Lz[iz]+Lz_interface
            z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=3/2)
            z_basis_list.append(z_basis)
            Lz_interface = Lz_top

        z_basis = de.Compound('z', tuple(z_basis_list),  dealias=3/2)

        logger.info("    Using nx = {}, Lx = {}".format(nx, Lx))
        logger.info("          nz = {}, nz_tot = {}, Lz = {}".format(nz, np.sum(nz), Lz))
       
        self.domain = de.Domain([x_basis, z_basis], grid_dtype=grid_dtype, comm=comm)
        
        self.x = self.domain.grid(0)
        self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
        self.nx = self.domain.bases[0].coeff_size
        self.delta_x = self.Lx/self.nx
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)

class polytrope(atmosphere):
    '''
    Single polytrope, stable or unstable.
    '''
    def __init__(self,
                 nx=256, nz=128,
                 Lx=None, aspect_ratio=4,
                 Lz=None, n_rho_cz = 3.5,
                 m_cz=None, epsilon=1e-4, gamma=5/3,
                 constant_diffusivities=True, constant_kappa=False,
                 **kwargs):
        
        self.atmosphere_name = 'single polytrope'

        self.m_ad = 1/(gamma-1)
        
        if m_cz is None:
            if epsilon is not None:
                m_cz = self.m_ad - epsilon
                self.epsilon = epsilon
            else:
                logger.error("Either m_cz or epsilon must be set")
                raise
        if Lz is None:
            if n_rho_cz is not None:
                Lz = self._calculate_Lz_cz(n_rho_cz, m_cz)
            else:
                logger.error("Either Lz or n_rho must be set")
                raise
        if Lx is None:
            Lx = Lz*aspect_ratio
            
        super(polytrope, self).__init__(gamma=gamma, nx=nx, nz=nz, Lx=Lx, Lz=Lz, **kwargs)
        
        self.constant_diffusivities = constant_diffusivities
        if constant_kappa:
            self.constant_diffusivities = False
            
        self._set_atmosphere()
        
    def _calculate_Lz_cz(self, n_rho_cz, m_cz):
        '''
        Calculate Lz based on the number of density scale heights and the initial polytrope.
        '''
        Lz_cz = np.exp(n_rho_cz/m_cz)-1
        return Lz_cz
        
    def _set_atmosphere(self):
        super(polytrope, self)._set_atmosphere()
        
        # polytropic atmosphere characteristics
        self.poly_n = 1/(self.gamma-1) - self.epsilon

        self.z0 = 1. + self.Lz

        self.del_ln_rho_factor = -self.poly_n
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)
        self.rho0['g'] = (self.z0 - self.z)**self.poly_n

        self.del_s0_factor = - self.epsilon/self.gamma
        self.delta_s = self.del_s0_factor*np.log(self.z0)
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)
 
        self.T0_zz['g'] = 0        
        self.del_T0['g'] = -1
        self.T0['g'] = self.z0 - self.z       

        self.P0['g'] = (self.z0 - self.z)**(self.poly_n+1)
        self.P0.differentiate('z', out=self.del_P0)
        self.del_P0.set_scales(1, keep_data=True)
        self.P0.set_scales(1, keep_data=True)
        
        if self.constant_diffusivities:
            self.scale['g'] = self.z0 - self.z
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g'] = self.z0 - self.z

        self.g = self.poly_n + 1
        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z0 - self.z)
        
        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_n = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_n, self.epsilon, self.gamma))
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        
        logger.info("   density scale heights = {:g}".format(np.log(self.Lz**self.poly_n)))
        H_rho_top = (self.z0-self.Lz)/self.poly_n
        H_rho_bottom = (self.z0)/self.poly_n
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format(H_rho_top,H_rho_bottom))
        logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))

        # min of global quantity
        self.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']))), op=MPI.MIN)
        self.freefall_time = np.sqrt(self.Lz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,
                                                                                               self.freefall_time,
                                                                                               self.buoyancy_time))
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))

        # set nu and chi at top based on Rayleigh number
        nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s)*self.g)/Rayleigh)
        chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))

            # determine characteristic timescales
            self.thermal_time = self.Lz**2/chi
            self.top_thermal_time = 1/chi

            self.viscous_time = self.Lz**2/nu
            self.top_viscous_time = 1/nu
        
        else:
            self.constant_kappa = True

            if self.constant_kappa:
                # take constant nu, constant kappa (non-constant chi); Prandtl changes.
                nu  = nu_top
                logger.info("   using constant nu, kappa")
            else:
                # take constant mu, kappa based on setting a top-of-domain Rayleigh number
                # nu  =  nu_top/(self.rho0['g'])
                logger.error("   using constant mu, kappa <DISABLED>")
                raise
            
            chi = chi_top/(self.rho0['g'])
        
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))

            # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
            # broken in parallel runs.
            self.thermal_time = self.Lz**2/chi_top #chi[...,self.nz/2][0]
            self.top_thermal_time = 1/chi_top

            self.viscous_time = self.Lz**2/nu_top #nu[...,self.nz/2][0]
            self.top_viscous_time = 1/nu_top
            
        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))
        
        #Allows for atmosphere reuse
        self.chi.set_scales(1, keep_data=True)
        
        self.nu['g'] = nu
        self.chi['g'] = chi
        if not self.constant_diffusivities:
            self.chi.differentiate('z', out=self.del_chi)
            self.chi.set_scales(1, keep_data=True)
            

class multitrope(multi_layer_atmosphere):
    '''
    Multiple joined polytropes.  Currently two are supported, unstable on top, stable below.  To be generalized.

    When specifying the z resolution, use a list, with the stable layer
    as the first entry and the unstable layer as the second list entry.
    e.g.,
    nz = [nz_rz, nz_cz]
    
    '''
    def __init__(self, nx=256, nz=[128, 128],
                 aspect_ratio=4,
                 gamma=5/3,
                 n_rho_cz=3.5, n_rho_rz=2, 
                 m_rz=3, stiffness=100,
                 **kwargs):

        self.atmosphere_name = 'multitrope'
        
        # gamma = c_p/c_v
        # n_rho_cz = number of density scale heights in CZ
        # n_rho_rz = number of density scale heights in RZ
        # m_rz = polytropic index of radiative zone
        # stiffness = (m_rz - m_ad)/(m_ad - m_cz) = (m_rz - m_ad)/epsilon

        self.m_ad = 1/(gamma-1)
        self.m_rz = m_rz
        self.stiffness = stiffness
        self.epsilon = (self.m_rz - self.m_ad)/self.stiffness
        self.m_cz = self.m_ad - self.epsilon


        self.n_rho_cz = n_rho_cz
        self.n_rho_rz = n_rho_rz
        
        Lz_cz, Lz_rz, Lz = self._calculate_Lz(n_rho_cz, self.m_cz, n_rho_rz, self.m_rz)
        self.Lz_cz = Lz_cz
        self.Lz_rz = Lz_rz
        
        Lx = Lz_cz*aspect_ratio
        
        super(multitrope, self).__init__(gamma=gamma, nx=nx, nz=nz, Lx=Lx, Lz=[Lz_rz, Lz_cz], **kwargs)

        self._set_atmosphere()
        
    def _calculate_Lz(self, n_rho_cz, m_cz, n_rho_rz, m_rz):
        '''
        Estimate the depth of the CZ and the RZ.
        '''
        # T = del_T*(z-z_interface) + T_interface
        # del_T = -g/(m+1) = -(m_cz+1)/(m+1)
        # example: cz: z_interface = L_cz (top), T_interface = 1, del_T = -1
        #     .: T = -1*(z - L_cz) + 1 = (L_cz + 1 - z) = (z0 - z)
        # this recovers the Lecoanet et al 2014 notation
        #
        # T_bot = -del_T*z_interface + T_interface
        # n_rho = ln(rho_bot/rho_interface) = m*ln(T_bot/T_interface)
        #       = m*ln(-del_T*z_interface/T_interface + 1)
        # 
        # z_interface = (T_interface/(-del_T))*(np.exp(n_rho/m)-1)

        Lz_cz = np.exp(n_rho_cz/m_cz)-1

        del_T_rz = -(m_cz+1)/(m_rz+1)
        T_interface = (Lz_cz+1) # T at bottom of CZ
        Lz_rz = T_interface/(-del_T_rz)*(np.exp(n_rho_rz/m_rz)-1)

        Lz = Lz_cz + Lz_rz
        logger.info("Calculating scales {}".format((Lz_cz, Lz_rz, Lz)))
        return (Lz_cz, Lz_rz, Lz)

    def _compute_kappa_profile(self, kappa_ratio, tanh_center=None, tanh_width=1):
        if tanh_center is None:
            tanh_center = self.Lz_rz

        # start with a simple profile, adjust amplitude later (in _set_diffusivities)
        kappa_top = 1
    
        phi = (1/2*(1-np.tanh((self.z-tanh_center)/tanh_width)))
        inv_phi = 1-phi
        self.kappa = self._new_ncc() 
        self.kappa['g'] = (phi*kappa_ratio+inv_phi)*kappa_top
        self.necessary_quantities['kappa'] = self.kappa
        
    def _set_atmosphere(self):
        super(multi_layer_atmosphere, self)._set_atmosphere()
        
        kappa_ratio = (self.m_rz + 1)/(self.m_cz + 1)
        
        self.z_cz =self.Lz_cz + 1

        self.delta_s = self.epsilon*(self.gamma-1)/self.gamma*np.log(self.z_cz)
        logger.info("Atmosphere delta s is {}".format(self.delta_s))

        self.g = (self.m_cz + 1)
        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z_cz - self.z)

        # this doesn't work: numerical instability blowup, and doesn't reduce bandwidth much at all
        #self.scale['g'] = (self.z_cz - self.z)
        # this seems to work fine; bandwidth only a few terms worse.
        self.scale['g'] = 1.
                
        self._compute_kappa_profile(kappa_ratio, tanh_center=self.Lz_rz, tanh_width=0.25)

        logger.info("Solving for T0")
        # start with an arbitrary -1 at the top, which will be rescaled after _set_diffusivites
        flux_top = -1
        self.del_T0['g'] = flux_top/self.kappa['g']
        self.del_T0.antidifferentiate('z',('right',0), out=self.T0)
        self.T0['g'] += 1
        self.T0.set_scales(1, keep_data=True)
    
        self.del_ln_P0 = self._new_ncc()
        self.ln_P0 = self._new_ncc()
        self.necessary_quantities['ln_P0'] = self.ln_P0
        self.necessary_quantities['del_ln_P0'] = self.del_ln_P0
        
        logger.info("Solving for P0")
        # assumes ideal gas equation of state
        self.del_ln_P0['g'] = -self.g/self.T0['g']
        self.del_ln_P0.antidifferentiate('z',('right',0),out=self.ln_P0)
        self.ln_P0.set_scales(1, keep_data=True)
        self.P0['g'] = np.exp(self.ln_P0['g'])
        self.del_ln_P0.set_scales(1, keep_data=True)
        self.del_P0['g'] = self.del_ln_P0['g']*self.P0['g']
        self.del_P0.set_scales(1, keep_data=True)
        
        self.rho0['g'] = self.P0['g']/self.T0['g']

        self.rho0.differentiate('z', out=self.del_ln_rho0)
        self.del_ln_rho0['g'] = self.del_ln_rho0['g']/self.rho0['g']

        self.rho0.set_scales(1, keep_data=True)         
        self.del_ln_P0.set_scales(1, keep_data=True)        
        self.del_ln_rho0.set_scales(1, keep_data=True)        
        self.del_s0['g'] = 1/self.gamma*self.del_ln_P0['g'] - self.del_ln_rho0['g']

        logger.info("multitrope atmosphere parameters:")
        logger.info("   m_cz = {:g}, epsilon = {:g}, gamma = {:g}".format(self.m_cz, self.epsilon, self.gamma))
        logger.info("   m_rz = {:g}, stiffness = {:g}".format(self.m_rz, self.stiffness))

        logger.info("   Lx = {:g}, Lz = {:g} (Lz_cz = {:g}, Lz_rz = {:g})".format(self.Lx, self.Lz, self.Lz_cz, self.Lz_rz))

        T0_max = self.domain.dist.comm_cart.allreduce(np.max(self.T0['g']), op=MPI.MAX)
        T0_min = self.domain.dist.comm_cart.allreduce(np.min(self.T0['g']), op=MPI.MIN)
        logger.info("   temperature: min {}  max {}".format(T0_min, T0_max))

        P0_max = self.domain.dist.comm_cart.allreduce(np.max(self.P0['g']), op=MPI.MAX)
        P0_min = self.domain.dist.comm_cart.allreduce(np.min(self.P0['g']), op=MPI.MIN)
        logger.info("   pressure: min {}  max {}".format(P0_min, P0_max))

        rho0_max = self.domain.dist.comm_cart.allreduce(np.max(self.rho0['g']), op=MPI.MAX)
        rho0_min = self.domain.dist.comm_cart.allreduce(np.min(self.rho0['g']), op=MPI.MIN)
        rho0_ratio = rho0_max/rho0_min
        logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))
        logger.info("   density scale heights = {:g}".format(np.log(rho0_ratio)))
        logger.info("   target n_rho_cz = {:g} n_rho_rz = {:g}".format(self.n_rho_cz, self.n_rho_rz))
        logger.info("   target n_rho_total = {:g}".format(self.n_rho_cz+self.n_rho_rz))
        H_rho_top = (self.z_cz-self.Lz_cz)/self.m_cz
        H_rho_bottom = (self.z_cz)/self.m_cz
        logger.info("   H_rho = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top,H_rho_bottom))
        logger.info("   H_rho/delta x = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))

        # min of global quantity
        self.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']))), op=MPI.MIN)
        self.freefall_time = np.sqrt(self.Lz_cz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz_cz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,
                                                                                               self.freefall_time,
                                                                                               self.buoyancy_time))

    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1):
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        Rayleigh_top = Rayleigh
        Prandtl_top = Prandtl
        # inputs:
        # Rayleigh_top = g dS L_cz**3/(chi_top**2 * Pr_top)
        # Prandtl_top = nu_top/chi_top
        self.chi_top = np.sqrt((self.g*self.delta_s*self.Lz_cz**3)/(Rayleigh_top*Prandtl_top))
        self.nu_top = self.chi_top*Prandtl_top

        self.constant_diffusivities = False

        self.nu['g'] = self.nu_top
        # rescale kappa to correct values based on Rayleigh number derived chi
        self.kappa['g'] *= self.chi_top
        self.chi['g'] = self.kappa['g']/self.rho0['g']
        self.chi.differentiate('z', out=self.del_chi)
        self.chi.set_scales(1, keep_data=True)

        self.top_thermal_time = 1/self.chi_top
        self.thermal_time = self.Lz_cz**2/self.chi_top

        logger.info("   nu_top = {:g}, chi_top = {:g}".format(self.nu_top, self.chi_top))            
        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))

# need to implement flux-based Rayleigh number here.
class polytrope_flux(polytrope):
    def __init__(self, *args, **kwargs):
        super(polytrope, self).__init__(*args, **kwargs)
        self.atmosphere_name = 'single polytrope'
        
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1):
        
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

        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))

        self.nu['g'] = nu
        self.chi['g'] = chi
        
        return nu, chi        

    def set_BC(self):
        self.problem.add_bc( "left(Q_z) = 1")
        self.problem.add_bc("right(Q_z) = 1")
            
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")


class equations():
    def __init__(self):
        pass
    
    def set_IVP_problem(self, *args, **kwargs):
        self.problem = de.IVP(self.domain, variables=self.variables)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, **kwargs):
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega')
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def _set_subs(self):
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'

        self.problem.parameters['delta_s_atm'] = self.delta_s
        self.problem.substitutions['s_fluc'] = '(1/Cv_inv*log(1+T1/T0) - 1/Cv_inv*(gamma-1)*ln_rho1)'
        self.problem.substitutions['s_mean'] = '(1/Cv_inv*log(T0) - 1/Cv_inv*(gamma-1)*ln_rho0)'
        
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

        self.problem.substitutions['h_flux'] = 'w*h'
        self.problem.substitutions['kappa_flux_mean'] = '-rho_full*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '-rho_full*chi*dz(T1)'
        self.problem.substitutions['kappa_flux'] = '((kappa_flux_mean) + (kappa_flux_fluc))'
        self.problem.substitutions['KE_flux'] = 'w*KE'

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)/(nu*chi)'
        
        self.problem.substitutions['enstrophy'] = '(dx(w) - u_z)**2'
        self.problem.substitutions['vorticity'] = '(dx(w) - u_z)'        

        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

    def initialize_output(self, solver, data_dir, **kwargs):
        analysis_tasks = []
        self.analysis_tasks = analysis_tasks
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("s_fluc", name="s")
        analysis_slice.add_task("s_fluc - plane_avg(s_fluc)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("vorticity", name="vorticity")
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
        analysis_profile.add_task("plane_avg(kappa_flux)", name="kappa_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_fluc)", name="kappa_flux_fluc_z")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)", name="Rayleigh_local")
        
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
        analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")

        analysis_tasks.append(analysis_scalar)

        # workaround for issue #29
        self.problem.namespace['enstrophy'].store_last = True

        return self.analysis_tasks
    
class FC_equations(equations):
    def __init__(self):
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes'
        self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1']
        
    def set_equations(self, Rayleigh, Prandtl, include_background_flux=True):
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        self._set_subs()
        
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dz(f_z))"
        self.problem.substitutions['Div(f, f_z)'] = "(dx(f) + f_z)"
        self.problem.substitutions['Div_u'] = "Div(u, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + w*(f_z))"
        
        # here, nu and chi are constants        
        self.viscous_term_w = " nu*(Lap(w, w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*Div_u)"
        self.viscous_term_u = " nu*(Lap(u, u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*Div(dx(u), dx(w_z)))"
        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u

        self.nonlinear_viscous_w = " nu*(    u_z*dx(ln_rho1) + 2*w_z*dz(ln_rho1) + dx(ln_rho1)*dx(w) - 2/3*dz(ln_rho1)*Div_u)"
        self.nonlinear_viscous_u = " nu*(2*dx(u)*dx(ln_rho1) + dx(w)*dz(ln_rho1) + dz(ln_rho1)*u_z   - 2/3*dx(ln_rho1)*Div_u)"
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u

        # double check implementation of variabile chi and background coupling term.
        self.problem.substitutions['Q_z'] = "(-T1_z)"
        self.thermal_diff           = " Cv_inv*chi*(Lap(T1, T1_z)      + T1_z*del_ln_rho0)"
        self.nonlinear_thermal_diff = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source = ""
        if include_background_flux:
            self.source +=    " Cv_inv*chi*(T0_zz + del_T0*del_ln_rho0 + del_T0*dz(ln_rho1))"
        else:
            self.source += " +0 "
        if not self.constant_diffusivities:
            self.thermal_diff +=    " + Cv_inv*del_chi*dz(T1) "
            if include_background_flux:
                self.source += " + Cv_inv*del_chi*del_T0"
                
        self.problem.substitutions['L_thermal'] = self.thermal_diff 
        self.problem.substitutions['NL_thermal'] = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source
        
        self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2/3*Div_u**2)"
        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
        
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
        
        self.problem.add_equation(("(scale)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))

        self.problem.add_equation(("(scale)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))

        self.problem.add_equation(("(scale)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        # here we have assumed chi = constant in both rho and radius
        self.problem.add_equation(("(scale)*( dt(T1)   + w*del_T0 + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 
        
        logger.info("using nonlinear EOS for entropy, via substitution")
        # non-linear EOS for s, where we've subtracted off
        # Cv_inv*∇s0 =  del_T0/(T0 + T1) - (gamma-1)*del_ln_rho0
        # move entropy to a substitution; no need to solve for it.
        #self.problem.add_equation(("(scale)*(Cv_inv*s - T1/T0 + (gamma-1)*ln_rho1) = "
        #                           "(scale)*(log(1+T1/T0) - T1/T0)"))

                
    def set_BC(self,
               fixed_flux=False, fixed_temperature=False, mixed_flux_temperature=True,
               stress_free=True, no_slip=False):
        
        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (T1_z)")
            self.problem.add_bc( "left(Q_z) = 0")
            self.problem.add_bc("right(Q_z) = 0")
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")            
        elif mixed_flux_temperature:
            logger.info("Thermal BC: mixed flux/temperature (T1_z/T1)")
            self.problem.add_bc("left(Q_z) = 0")
            self.problem.add_bc("right(T1) = 0")
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc( "left(u_z) = 0")
            self.problem.add_bc("right(u_z) = 0")
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")


class FC_polytrope(FC_equations, polytrope):
    def __init__(self, *args, **kwargs):
        super(FC_polytrope, self).__init__() 
        polytrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_polytrope, self).set_equations(*args, **kwargs)
        self.check_atmosphere()
        
class FC_multitrope(FC_equations, multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_multitrope, self).__init__() 
        multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_multitrope,self).set_equations(*args, **kwargs)
        self.check_atmosphere()

                        
# needs to be tested again and double-checked
class AN_polytrope(polytrope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_IVP_problem(self, Rayleigh, Prandtl):
        
        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'], cutoff=1e-10)

        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
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
        pass
        
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
