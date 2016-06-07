import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from tools.EVP import EVP_homogeneous
from dedalus import public as de

class Atmosphere:
    def __init__(self, verbose=False, **kwargs):
        self._set_domain(**kwargs)
        
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

    def filter_field(self, field,frac=0.25):
        logger.info("filtering field with frac={}".format(frac))
        dom = field.domain
        local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
        coeff = []
        for i in range(dom.dim)[::-1]:
            coeff.append(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False))
        cc = np.meshgrid(*coeff)

        field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

        for i in range(dom.dim):
            field_filter = field_filter | (cc[i][local_slice] > frac)
        field['c'][field_filter] = 0j

    def _new_ncc(self):
        field = self.domain.new_field()
        field.meta['x']['constant'] = True
        return field

    def _new_field(self):
        field = self.domain.new_field()
        return field

    def get_problem(self):
        return self.problem

    def evaluate_at_point(self, f, z=0):
        return f.interpolate(z=z)

    def value_at_boundary(self, field):
        orig_scale = field.meta[:]['scale']
        try:
            field_top    = self.evaluate_at_point(field, z=self.Lz)['g'][0][0]
            if not np.isfinite(field_top):
                logger.info("Likely interpolation error at top boundary; setting field=1")
                logger.info("orig_scale: {}".format(orig_scale))
                field_top = 1
            field_bottom = self.evaluate_at_point(field, z=0)['g'][0][0]
            field.set_scales(orig_scale, keep_data=True)
        except:
            logger.debug("field at top shape {}".format(field['g'].shape))
            field_top = None
            field_bottom = None
        
        return field_bottom, field_top
    
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
        self.T0_z = self._new_ncc()
        self.T0 = self._new_ncc()
        self.necessary_quantities['T0_zz'] = self.T0_zz
        self.necessary_quantities['T0_z'] = self.T0_z
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
        self.problem.parameters['T0_z'] = self.T0_z
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
        self.problem.parameters['del_chi'] = self.del_chi

    def copy_atmosphere(self, atmosphere):
        '''
        Copies values from a target atmosphere into the current atmosphere.
        '''
        self.necessary_quantities = atmosphere.necessary_quantities
            
    def plot_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            ax.set_xlabel('z')
            ax.set_ylabel(key)

            ax = fig_q.add_subplot(2,1,2)
            power_spectrum = np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:]))
            ax.plot(np.arange(len(quantity['c'][0,:])), power_spectrum)
            ax.axhline(y=1e-20, color='black', linestyle='dashed') # ncc_cutoff = 1e-10
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')
            
            fig_q.savefig("atmosphere_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)

            
        fig_atm = plt.figure()
        axT = fig_atm.add_subplot(2,2,1)
        axT.plot(self.z[0,:], self.T0['g'][0,:])
        axT.set_ylabel('T0')
        axP = fig_atm.add_subplot(2,2,2)
        axP.semilogy(self.z[0,:], self.P0['g'][0,:]) 
        axP.set_ylabel('P0')
        axR = fig_atm.add_subplot(2,2,3)
        axR.semilogy(self.z[0,:], self.rho0['g'][0,:])
        axR.set_ylabel(r'$\rho0$')
        axS = fig_atm.add_subplot(2,2,4)
        analysis.semilogy_posneg(axS, self.z[0,:], self.del_s0['g'][0,:], color_neg='red')
        
        axS.set_ylabel(r'$\nabla s0$')
        fig_atm.savefig("atmosphere_quantities_p{}.png".format(self.domain.distributor.rank), dpi=300)

        fig_atm = plt.figure()
        axS = fig_atm.add_subplot(2,2,1)
        axdelS = fig_atm.add_subplot(2,2,2)
        axlnP = fig_atm.add_subplot(2,2,3)
        axdellnP = fig_atm.add_subplot(2,2,4)

        Cv_inv = self.gamma-1
        axS.plot(self.z[0,:], 1/Cv_inv*np.log(self.T0['g'][0,:]) - 1/Cv_inv*(self.gamma-1)*np.log(self.rho0['g'][0,:]), label='s0', linewidth=2)
        axS.plot(self.z[0,:], (1+(self.gamma-1)/self.gamma*self.g)*np.log(self.T0['g'][0,:]), label='s based on lnT', linewidth=2)
        axS.plot(self.z[0,:], np.log(self.T0['g'][0,:]) - (self.gamma-1)/self.gamma*np.log(self.P0['g'][0,:]), label='s based on lnT and lnP', linewidth=2)
        
        axdelS.plot(self.z[0,:], self.del_s0['g'][0,:], label=r'$\nabla s0$', linewidth=2)
        axdelS.plot(self.z[0,:], self.T0_z['g'][0,:]/self.T0['g'][0,:] + self.g*(self.gamma-1)/self.gamma*1/self.T0['g'][0,:],
                    label=r'$\nabla s0$ from T0', linewidth=2, linestyle='dashed',color='red')
         
        axlnP.plot(self.z[0,:], np.log(self.P0['g'][0,:]), label='ln(P)', linewidth=2)
        axlnP.plot(self.z[0,:], self.ln_P0['g'][0,:], label='lnP', linestyle='dashed', linewidth=2)
        axlnP.plot(self.z[0,:], -self.g*np.log(self.T0['g'][0,:])*(self.T0_z['g'][0,:]), label='-g*lnT', linewidth=2, linestyle='dotted')
        
        axdellnP.plot(self.z[0,:], self.del_ln_P0['g'][0,:], label='dellnP', linewidth=2)
        axdellnP.plot(self.z[0,:], -self.g/self.T0['g'][0,:], label='-g/T', linestyle='dashed', linewidth=2, color='red')
        
        #axS.legend()
        axS.set_ylabel(r'$s0$')
        fig_atm.savefig("atmosphere_s0_p{}.png".format(self.domain.distributor.rank), dpi=300)

    def plot_scaled_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity['g'] *= self.scale['g']
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            ax.set_xlabel('z')
            ax.set_ylabel(key+'*scale')

            ax = fig_q.add_subplot(2,1,2)
            ax.plot(np.arange(len(quantity['c'][0,:])), np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:])))
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}*scale".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')
            
            fig_q.savefig("atmosphere_{}scale_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)
                        
    def check_that_atmosphere_is_set(self):
        for key in self.necessary_quantities:
            quantity = self.necessary_quantities[key]['g']
            quantity_set = quantity.any()
            if not quantity_set:
                logger.info("WARNING: atmosphere {} is all zeros".format(key))
                
    def test_hydrostatic_balance(self, P_z=None, P=None, T=None, rho=None, make_plots=False):

        if rho is None:
            logger.error("HS balance test requires rho (currently)")
            raise
        
        if P_z is None:
            if P is None:
                if T is None:
                    logger.error("HS balance test requires P_z, P or T")
                    raise
                else:
                    T_scales = T.meta[:]['scale']
                    rho_scales = rho.meta[:]['scale']
                    P = self._new_field()
                    T.set_scales(self.domain.dealias, keep_data=True)
                    rho.set_scales(self.domain.dealias, keep_data=True)
                    P.set_scales(self.domain.dealias, keep_data=False)
                    P['g'] = T['g']*rho['g']
                    T.set_scales(T_scales, keep_data=True)
                    rho.set_scales(rho_scales, keep_data=True)

            P_z = self._new_field()
            P.differentiate('z', out=P_z)
            P_z.set_scales(1, keep_data=True)

        rho_scales = rho.meta[:]['scale']
        rho.set_scales(1, keep_data=True)
        # error in hydrostatic balance diagnostic
        HS_balance = P_z['g']+self.g*rho['g']
        relative_error = HS_balance/P_z['g']
        rho.set_scales(rho_scales, keep_data=True)
        
        HS_average = self._new_field()
        HS_average['g'] = HS_balance
        HS_average.integrate('x')
        HS_average['g'] /= self.Lx
        HS_average.set_scales(1, keep_data=True)

        relative_error_avg = self._new_field()
        relative_error_avg['g'] = relative_error
        relative_error_avg.integrate('x')
        relative_error_avg['g'] /= self.Lx
        relative_error_avg.set_scales(1, keep_data=True)

        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(self.z[0,:], P_z['g'][0,:])
            ax1.plot(self.z[0,:], -self.g*rho['g'][0,:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
            ax2.semilogy(self.z[0,:], np.abs(relative_error_avg['g'][0,:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig("atmosphere_HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)
            
        #print("rank {} : |{}|".format(self.domain.distributor.rank,np.max(np.abs(relative_error))))
        max_rel_err = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error)), op=MPI.MAX)
        max_rel_err_avg = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error_avg['g'])), op=MPI.MAX)
        logger.info('max error in HS balance: point={} avg={}'.format(max_rel_err, max_rel_err_avg))

    def check_atmosphere(self, make_plots=False, **kwargs):
        if self.make_plots or make_plots:
            try:
                self.plot_atmosphere()
            except:
                logger.info("Problems in plot_atmosphere: atm full of NaNs?")
        self.test_hydrostatic_balance(make_plots=make_plots, **kwargs)
        self.check_that_atmosphere_is_set()


class MultiLayerAtmosphere(Atmosphere):
    def __init__(self, *args, **kwargs):
        super(MultiLayerAtmosphere, self).__init__(*args, **kwargs)
        
    def _set_domain(self, nx=256, Lx=4, nz=[128, 128], Lz=[1,1], grid_dtype=np.float64, comm=MPI.COMM_WORLD):
        '''
        Specify 2-D domain, with compund basis in z-direction.

        First entries in nz, Lz are the bottom entries (build upwards).
        '''
        if len(nz) != len(Lz):
            logger.error("nz {} has different number of elements from Lz {}".format(nz, Lz))
            raise
                         
        x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)

        if len(nz)>1:
            logger.info("Setting compound basis in vertical (z) direction")
            z_basis_list = []
            Lz_interface = 0.
            for iz, nz_i in enumerate(nz):
                Lz_top = Lz[iz]+Lz_interface
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=3/2)
                z_basis_list.append(z_basis)
                Lz_interface = Lz_top

            z_basis = de.Compound('z', tuple(z_basis_list),  dealias=3/2)
        elif len(nz)==1:
            logger.info("Setting single chebyshev basis in vertical (z) direction")
            z_basis = de.Chebyshev('z', nz[0], interval=[0, Lz[0]], dealias=3/2)
        else:
            logger.error("error in specification of vertical basis")
            
             
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
        self.nz_set = nz
        
        self.z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)

    def filter_field(self, field,frac=0.25):
        logger.info("compound domain: filtering field with frac={}".format(frac))
        dom = field.domain
        local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
        coeff = []
        first_subbasis = True
        for i in range(dom.dim)[::-1]:
            if i == np.max(dom.dim)-1:
                # special operation on the compound basis
                for nz in self.nz_set:
                    logger.info("nz: {} out of {}".format(nz, self.nz_set))
                    if first_subbasis:
                        compound_set = np.linspace(0,1,nz,endpoint=False)
                        first_subbasis=False
                    else:
                        compound_set = np.append(compound_set, np.linspace(0,1,nz,endpoint=False))
                logger.debug("compound set shape {}".format(compound_set.shape))
                logger.debug("target shape {}".format(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False).shape))
                coeff.append(compound_set)
            else:
                coeff.append(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False))
        cc = np.meshgrid(*coeff)
        for i in range(len(cc)):
            logger.debug("cc {} shape {}".format(i, cc[i].shape))
        field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

        for i in range(dom.dim):
            field_filter = field_filter | (cc[i][local_slice] > frac)
        field['c'][field_filter] = 0j

class Polytrope(Atmosphere):
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

        self._set_atmosphere_parameters(gamma=gamma, epsilon=epsilon, poly_m=m_cz)
        if m_cz is None:
            m_cz = self.poly_m

        if Lz is None:
            if n_rho_cz is not None:
                Lz = self._calculate_Lz_cz(n_rho_cz, m_cz)
            else:
                logger.error("Either Lz or n_rho must be set")
                raise
        if Lx is None:
            Lx = Lz*aspect_ratio

        super(Polytrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=Lz)
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        self.z0 = 1. + self.Lz
        
        self.constant_diffusivities = constant_diffusivities
        if constant_kappa:
            self.constant_diffusivities = False

        self._set_atmosphere()
        self._set_timescales()

    def _calculate_Lz_cz(self, n_rho_cz, m_cz):
        '''
        Calculate Lz based on the number of density scale heights and the initial polytrope.
        '''
        Lz_cz = np.exp(n_rho_cz/m_cz)-1
        return Lz_cz
    
    def _set_atmosphere_parameters(self, gamma=5/3, epsilon=0, poly_m=None, g=None):
        # polytropic atmosphere characteristics
        self.gamma = gamma
        self.epsilon = epsilon

        self.m_ad = 1/(self.gamma-1)

        # trap on poly_m/epsilon conflicts?
        if poly_m is None:
            self.poly_m = self.m_ad - self.epsilon
        else:
            self.poly_m = poly_m

        if g is None:
            self.g = self.poly_m + 1
        else:
            self.g = g

        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_m = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_m, self.epsilon, self.gamma))
    
    def _set_atmosphere(self):
        super(Polytrope, self)._set_atmosphere()

        self.del_ln_rho_factor = -self.poly_m
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)
        self.rho0['g'] = (self.z0 - self.z)**self.poly_m

        self.del_s0_factor = - self.epsilon/self.gamma
        self.delta_s = self.del_s0_factor*np.log(self.z0)
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)
 
        self.T0_zz['g'] = 0        
        self.T0_z['g'] = -1
        self.T0['g'] = self.z0 - self.z       

        self.P0['g'] = (self.z0 - self.z)**(self.poly_m+1)
        self.P0.differentiate('z', out=self.del_P0)
        self.del_P0.set_scales(1, keep_data=True)
        self.P0.set_scales(1, keep_data=True)
        
        if self.constant_diffusivities:
            self.scale['g'] = self.z0 - self.z
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g'] = self.z0 - self.z

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z0 - self.z)

        rho0_max, rho0_min = self.value_at_boundary(self.rho0)
        if rho0_max is not None:
            rho0_ratio = rho0_max/rho0_min
            logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))
            logger.info("   density scale heights = {:g} (measured)".format(np.log(rho0_ratio)))
            logger.info("   density scale heights = {:g} (target)".format(np.log((self.z0)**self.poly_m)))
            
        H_rho_top = (self.z0-self.Lz)/self.poly_m
        H_rho_bottom = (self.z0)/self.poly_m
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format(H_rho_top,H_rho_bottom))
        logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
            
        # min of global quantity
        atmosphere.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']))), op=MPI.MIN)
        atmosphere.freefall_time = np.sqrt(self.Lz/self.g)
        atmosphere.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(atmosphere.min_BV_time,
                                                                                               atmosphere.freefall_time,
                                                                                               atmosphere.buoyancy_time))
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
                    
        #Allows for atmosphere reuse
        self.chi.set_scales(1, keep_data=True)
        self.nu['g'] = nu
        self.chi['g'] = chi

        self.chi.differentiate('z', out=self.del_chi)
        self.chi.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/self.chi.interpolate(z=self.Lz/2)['g'][0][0]
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/self.nu.interpolate(z=self.Lz/2)['g'][0][0]
        self.top_viscous_time = 1/nu_top

        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))

class Polytrope_adiabatic(Polytrope):
    def __init__(self,
                 nx=256, nz=128,
                 Lx=None, aspect_ratio=4,
                 Lz=None, n_rho_cz = 3.5,
                 gamma=5/3,                 
                 **kwargs):

        logger.info("************* entering polytrope_adiabatic, setting full_atm")
        full_atm = Polytrope(Lz=Lz, n_rho_cz=n_rho_cz,
                             gamma=gamma,nx=nx,nz=nz, Lx=Lx, aspect_ratio=aspect_ratio,
                             **kwargs)
        
        self.atmosphere_name = 'single adiabatic polytrope'

        if Lz is None:
            # take Lz from full_atm so that n_rho_cz is consistent
            Lz = full_atm.Lz
        if Lx is None:
            Lx = Lz*aspect_ratio

        super(Polytrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=Lz)
        self.z0 = 1. + self.Lz

        logger.info("************* setting adiabatic atm")

        self.constant_diffusivities = full_atm.constant_diffusivities
        self._set_atmosphere_parameters(gamma=gamma, epsilon=0, poly_m=full_atm.m_ad)

        self._set_atmosphere()
        full_atm._set_timescales(atmosphere=self)

        self.T0_IC = self._new_ncc()
        self.rho0_IC = self._new_ncc()
        self.ln_rho0_IC = self._new_ncc()
        
        self.T0_IC['g'] = full_atm.T0['g'] - self.T0['g']
        self.rho0_IC['g'] = full_atm.rho0['g'] - self.rho0['g']
        self.ln_rho0_IC['g'] = np.log(full_atm.rho0['g']) - np.log(self.rho0['g'])

        self.delta_s = full_atm.delta_s
                      
class Multitrope(MultiLayerAtmosphere):
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
                 stable_bottom=True,
                 stable_top=False,
                 width=None,
                 overshoot_pad = None,
                 **kwargs):

        self.atmosphere_name = 'multitrope'
        
        if stable_top:
            stable_bottom = False
            
        self.stable_bottom = stable_bottom

        self._set_atmosphere_parameters(gamma=gamma, n_rho_cz=n_rho_cz,
                                        n_rho_rz=n_rho_rz, m_rz=m_rz, stiffness=stiffness)
        
        Lz_cz, Lz_rz, Lz = self._calculate_Lz(n_rho_cz, self.m_cz, n_rho_rz, self.m_rz)
        self.Lz_cz = Lz_cz
        self.Lz_rz = Lz_rz
        
        Lx = Lz_cz*aspect_ratio
        
        # guess at overshoot offset and tanh width
        # current guess for overshoot pad is based off of entropy equilibration
        # in a falling plume (location z in RZ where S(z) = S(z_top)), assuming
        # the plume starts with the entropy of the top of the CZ.
        # see page 139 of lab-book 15, 9/1/15 (Brown)
        T_bcz = Lz_cz+1
        L_ov = ((T_bcz)**((stiffness+1)/stiffness) - T_bcz)*(self.m_rz+1)/(self.m_cz+1)

        if overshoot_pad is None:
            overshoot_pad = 2*L_ov # add a safety factor of 2x for timestepping for now
            if overshoot_pad >= Lz_rz:
                # if we go past the bottom or top of the domain,
                # put the matching region in the middle of the stable layer.
                # should only be a major problem for stiffness ~ O(1)
                overshoot_pad = 0.5*Lz_rz

        if self.stable_bottom:
            self.match_center = self.Lz_rz
        else:
            self.match_center = self.Lz_cz
            
        # this is going to widen the tanh and move the location of del(s)=0 as n_rho_cz increases...
        # match_width = 2% of Lz_cz, somewhat analgous to Rogers & Glatzmaier 2005
        erf_v_tanh = 18.5/(np.sqrt(np.pi)*6.5/2)
        if width is None:
            width = 0.04*erf_v_tanh
        logger.info("erf width factor is {} of Lz_cz (total: {})".format(width, width*Lz_cz))
        self.match_width = width*Lz_cz # adjusted by ~3x for erf() vs tanh()
        
        logger.info("using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))
        if self.stable_bottom:
            Lz_bottom = Lz_rz - overshoot_pad
            Lz_top = Lz_cz + overshoot_pad
        else:
            Lz_bottom = Lz_cz + overshoot_pad
            Lz_top = Lz_rz - overshoot_pad

        if len(nz) == 1:
            #nz = nz[-1] # grab the last set of points as the full resolution; bit of a hack.
            # do this at the driving script level.
            super(Multitrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=[Lz_bottom + Lz_top], **kwargs)
        else:
            super(Multitrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=[Lz_bottom, Lz_top], **kwargs)

        logger.info("   Lx = {:g}, Lz = {:g} (Lz_cz = {:g}, Lz_rz = {:g})".format(self.Lx, self.Lz, self.Lz_cz, self.Lz_rz))

        self.z_cz =self.Lz_cz + 1
        self._set_atmosphere()
        logger.info("Done set_atmosphere")
        T0_max, T0_min = self.value_at_boundary(self.T0)
        P0_max, P0_min = self.value_at_boundary(self.P0)
        rho0_max, rho0_min = self.value_at_boundary(self.rho0)

        logger.info("   temperature: min {}  max {}".format(T0_min, T0_max))
        logger.info("   pressure: min {}  max {}".format(P0_min, P0_max))
        logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))

        if rho0_max is not None:
            rho0_ratio = rho0_max/rho0_min
            logger.info("   density scale heights = {:g}".format(np.log(rho0_ratio)))
            logger.info("   target n_rho_cz = {:g} n_rho_rz = {:g}".format(self.n_rho_cz, self.n_rho_rz))
            logger.info("   target n_rho_total = {:g}".format(self.n_rho_cz+self.n_rho_rz))
        H_rho_top = (self.z_cz-self.Lz_cz)/self.m_cz
        H_rho_bottom = (self.z_cz)/self.m_cz
        logger.info("   H_rho = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top,H_rho_bottom))
        logger.info("   H_rho/delta x = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))

        self._set_timescales()
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
        # min of global quantity
        BV_time = np.sqrt(np.abs(self.g*self.del_s0['g']))
        if BV_time.shape[-1] == 0:
            logger.debug("BV_time {}, shape {}".format(BV_time, BV_time.shape))
            BV_time = np.array([np.inf])
            
        self.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(BV_time), op=MPI.MIN)
        self.freefall_time = np.sqrt(self.Lz_cz/self.g)
        self.buoyancy_time = np.sqrt(self.Lz_cz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,
                                                                                               self.freefall_time,
                                                                                               self.buoyancy_time))
            
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

        if self.stable_bottom:
            Lz_cz = np.exp(n_rho_cz/m_cz)-1
        else:
            Lz_rz = np.exp(n_rho_rz/m_rz)-1
            
        del_T_rz = -(m_cz+1)/(m_rz+1)

        if self.stable_bottom:
            T_interface = (Lz_cz+1) # T at bottom of CZ
            Lz_rz = T_interface/(-del_T_rz)*(np.exp(n_rho_rz/m_rz)-1)
        else:
            T_interface = (Lz_rz+1) # T at bottom of CZ
            Lz_cz = T_interface/(-del_T_rz)*(np.exp(n_rho_cz/m_cz)-1)
            
        Lz = Lz_cz + Lz_rz
        logger.info("Calculating scales {}".format((Lz_cz, Lz_rz, Lz)))
        return (Lz_cz, Lz_rz, Lz)

    def match_Phi(self, z, f=scp.erf, center=None, width=None):
        if center is None:
            center = self.match_center
        if width is None:
            width = self.match_width
        return 1/2*(1-f((z-center)/width))
         
    def _compute_kappa_profile(self, kappa_ratio):
        # start with a simple profile, adjust amplitude later (in _set_diffusivities)
        kappa_top = 1
        Phi = self.match_Phi(self.z)
        inv_Phi = 1-Phi
        
        self.kappa = self._new_ncc()
        if self.stable_bottom:
            self.kappa['g'] = (Phi*kappa_ratio+inv_Phi)*kappa_top
        else:
            self.kappa['g'] = (Phi+inv_Phi*kappa_ratio)*kappa_top
        self.necessary_quantities['kappa'] = self.kappa


    def _set_atmosphere_parameters(self,
                                   gamma=5/3,
                                   n_rho_cz=3.5,
                                   n_rho_rz=2, m_rz=3, stiffness=100,
                                   g=None):
        
        # polytropic atmosphere characteristics

        # gamma = c_p/c_v
        # n_rho_cz = number of density scale heights in CZ
        # n_rho_rz = number of density scale heights in RZ
        # m_rz = polytropic index of radiative zone
        # stiffness = (m_rz - m_ad)/(m_ad - m_cz) = (m_rz - m_ad)/epsilon

        self.gamma = gamma

        self.m_ad = 1/(gamma-1)
        self.m_rz = m_rz
        self.stiffness = stiffness
        self.epsilon = (self.m_rz - self.m_ad)/self.stiffness
        self.m_cz = self.m_ad - self.epsilon

        self.n_rho_cz = n_rho_cz
        self.n_rho_rz = n_rho_rz
        
        if g is None:
            self.g = self.m_cz + 1
        else:
            self.g = g

        logger.info("multitrope atmosphere parameters:")
        logger.info("   m_cz = {:g}, epsilon = {:g}, gamma = {:g}".format(self.m_cz, self.epsilon, self.gamma))
        logger.info("   m_rz = {:g}, stiffness = {:g}".format(self.m_rz, self.stiffness))
    
    def _set_atmosphere(self, atmosphere_type2=False):
        super(MultiLayerAtmosphere, self)._set_atmosphere()
        
        kappa_ratio = (self.m_rz + 1)/(self.m_cz + 1)

        self.delta_s = self.epsilon*(self.gamma-1)/self.gamma*np.log(self.z_cz)
        logger.info("Atmosphere delta s is {}".format(self.delta_s))

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z_cz - self.z)

        # this doesn't work: numerical instability blowup, and doesn't reduce bandwidth much at all
        #self.scale['g'] = (self.z_cz - self.z)
        # this seems to work fine; bandwidth only a few terms worse.
        self.scale['g'] = 1.
        #self.scale['g'] = self.T0['g']
        #self.scale['g'] = (self.Lz+1 - self.z)/(self.Lz+1)

        if atmosphere_type2:
            # specify T0_z as smoothly matched profile
            # for now, hijack compute_kappa_profile (this could be cleaned up),
            # but grad T ratio has inverse relationship to kappa_ratio
            self._compute_kappa_profile(1/kappa_ratio)
            flux_top = -1
            # copy out kappa profile, which is really grad T
            self.T0_z['g'] = self.kappa['g']/flux_top
            # now invert grad T for kappa
            logger.info("Solving for kappa")
            self.kappa['g'] = flux_top/self.T0_z['g']
        else:
            # specify kappa as smoothly matched profile
            self._compute_kappa_profile(kappa_ratio)
            logger.info("Solving for T0")
            # start with an arbitrary -1 at the top, which will be rescaled after _set_diffusivites
            flux_top = -1
            self.T0_z['g'] = flux_top/self.kappa['g']
            
        self.T0_z.antidifferentiate('z',('right',0), out=self.T0)
        # need T0_zz in multitrope
        self.T0_z.differentiate('z', out=self.T0_zz)
        self.T0['g'] += 1
        self.T0.set_scales(1, keep_data=True)
        #self.scale['g'] = self.T0['g']

    
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


    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1):
        logger.info("problem parameters (multitrope):")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        Rayleigh_top = Rayleigh
        Prandtl_top = Prandtl
        # inputs:
        # Rayleigh_top = g dS L_cz**3/(chi_top**2 * Pr_top)
        # Prandtl_top = nu_top/chi_top
        self.chi_top = np.sqrt((self.g*self.delta_s*self.Lz_cz**3)/(Rayleigh_top*Prandtl_top))
            
        self.nu_top = self.chi_top*Prandtl_top

        if not self.stable_bottom:
            # try to rescale chi appropriately so that the
            # Rayleigh number is set at the top of the CZ
            # to the desired value by removing the density
            # scaling from the rz.  This is a guess.
            self.chi_top = np.exp(self.n_rho_rz)*self.chi_top

        self.constant_diffusivities = False

        logger.info("about to access nu")
        self.nu['g'] = self.nu_top
        # rescale kappa to correct values based on Rayleigh number derived chi
        logger.info("about to access kappa")

        self.kappa['g'] *= self.chi_top
        self.kappa.set_scales(self.domain.dealias, keep_data=True)
        self.rho0.set_scales(self.domain.dealias, keep_data=True)
        self.chi.set_scales(self.domain.dealias, keep_data=True)
        logger.info("about to access chi")
        if self.rho0['g'].shape[-1] != 0:
            self.chi['g'] = self.kappa['g']/self.rho0['g']
            self.chi.differentiate('z', out=self.del_chi)
            self.chi.set_scales(1, keep_data=True)
        logger.info("done chi")
        self.top_thermal_time = 1/self.chi_top
        self.thermal_time = self.Lz_cz**2/self.chi_top
        logger.info("done times")
        logger.info("   nu_top = {:g}, chi_top = {:g}".format(self.nu_top, self.chi_top))            
        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))

    def get_flux(self, rho, T):
        rho.set_scales(1,keep_data=True)
        T_z = self._new_ncc()
        T.differentiate('z', out=T_z)
        T_z.set_scales(1,keep_data=True)
        chi = self.chi
        chi.set_scales(1,keep_data=True)
        flux = self._new_ncc()
        flux['g'] = rho['g']*T_z['g']*chi['g']
        return flux


class MultitropeDense(Multitrope):
    '''
    Multiple joined polytropes.  Currently two are supported, unstable on top, stable below.  To be generalized.

    When specifying the z resolution, use a list, with the stable layer
    as the first entry and the unstable layer as the second list entry.
    e.g.,
    nz = [nz_rz, nz_cz]
    
    '''
    def __init__(self, nx=256, nz=[128, 32, 128],
                 aspect_ratio=4,
                 gamma=5/3,
                 n_rho_cz=3.5, n_rho_rz=2, 
                 m_rz=3, stiffness=100,
                 stable_bottom=True,
                 stable_top=False,
                 **kwargs):

        self.atmosphere_name = 'multitrope'
        
        if stable_top:
            stable_bottom = False
            
        self.stable_bottom = stable_bottom

        self._set_atmosphere_parameters(gamma=gamma, n_rho_cz=n_rho_cz,
                                        n_rho_rz=n_rho_rz, m_rz=m_rz, stiffness=stiffness)
        
        Lz_cz, Lz_rz, Lz = self._calculate_Lz(n_rho_cz, self.m_cz, n_rho_rz, self.m_rz)
        self.Lz_cz = Lz_cz
        self.Lz_rz = Lz_rz
        
        Lx = Lz_cz*aspect_ratio

        if self.stable_bottom:
            self.match_center = self.Lz_rz
        else:
            self.match_center = self.Lz_cz

        self.match_width = 0.02*Lz_cz # 2% of Lz_cz, somewhat analgous to Rogers & Glatzmaier 2005
        overshoot_pad = 3*self.match_width
        
        logger.info("using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))
        if self.stable_bottom:
            Lz_bottom = Lz_rz - overshoot_pad
            Lz_mid = 2*overshoot_pad
            Lz_top = Lz_cz + overshoot_pad
        else:
            Lz_bottom = Lz_cz + overshoot_pad
            Lz_mid = 2*overshoot_pad
            Lz_top = Lz_rz - overshoot_pad

        super(Multitrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=[Lz_bottom, Lz_mid, Lz_top], **kwargs)

        logger.info("   Lx = {:g}, Lz = {:g} (Lz_cz = {:g}, Lz_rz = {:g})".format(self.Lx, self.Lz, self.Lz_cz, self.Lz_rz))

        self.z_cz =self.Lz_cz + 1
        self._set_atmosphere()

        T0_max, T0_min = self.value_at_boundary(self.T0)
        P0_max, P0_min = self.value_at_boundary(self.P0)
        rho0_max, rho0_min = self.value_at_boundary(self.rho0)

        logger.info("   temperature: min {}  max {}".format(T0_min, T0_max))
        logger.info("   pressure: min {}  max {}".format(P0_min, P0_max))
        logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))

        rho0_ratio = rho0_max/rho0_min
        logger.info("   density scale heights = {:g}".format(np.log(rho0_ratio)))
        logger.info("   target n_rho_cz = {:g} n_rho_rz = {:g}".format(self.n_rho_cz, self.n_rho_rz))
        logger.info("   target n_rho_total = {:g}".format(self.n_rho_cz+self.n_rho_rz))
        H_rho_top = (self.z_cz-self.Lz_cz)/self.m_cz
        H_rho_bottom = (self.z_cz)/self.m_cz
        logger.info("   H_rho = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top,H_rho_bottom))
        logger.info("   H_rho/delta x = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))

        self._set_timescales()

        
# need to implement flux-based Rayleigh number here.
class PolytropeFlux(Polytrope):
    def __init__(self, *args, **kwargs):
        super(Polytrope, self).__init__(*args, **kwargs)
        self.atmosphere_name = 'single Polytrope'
        
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


class Equations():
    def __init__(self):
        pass
    
    def set_IVP_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem = EVP_homogeneous(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def at_boundary(self, f, z=0, tol=1e-12, derivative=False, dz=False, BC_text='BC'):        
        if derivative or dz:
            f_bc=self._new_ncc()
            f.differentiate('z', out=f_bc)
            BC_text += "_z"
        else:
            f_bc = f

        BC_field = f_bc.interpolate(z=z)

        try:
            BC = BC_field['g'][0][0]
        except:
            BC = BC_field['g']
            logger.error("shape of BC_field {}".format(BC_field['g'].shape))
            logger.error("BC = {}".format(BC))

        if np.abs(BC) < tol:
            BC = 0
        logger.info("Calculating boundary condition z={:7g}, {}={:g}".format(z, BC_text, BC))
        return BC

    def _set_subs(self):
        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dz(f_z))"
        self.problem.substitutions['Div(f, f_z)'] = "(dx(f) + f_z)"
        self.problem.substitutions['Div_u'] = "Div(u, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + w*(f_z))"
        
        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

        # output parameters        
        self.problem.substitutions['enstrophy'] = '(dx(w) - u_z)**2'
        self.problem.substitutions['vorticity'] = '(dx(w) - u_z)'        

    def global_noise(self, seed=42, **kwargs):            
        # Random perturbations, initialized globally for same results in parallel
        gshape = self.domain.dist.grid_layout.global_shape(scales=self.domain.dealias)
        slices = self.domain.dist.grid_layout.slices(scales=self.domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field = self._new_field()
        noise_field.set_scales(self.domain.dealias, keep_data=False)
        noise_field['g'] = noise
        self.filter_field(noise_field, **kwargs)

        return noise_field['g']
    
class FC_equations(Equations):
    def __init__(self):
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes'
        self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1']

        # possible boundary condition values for a normal system
        self.T1_left  = 0
        self.T1_right = 0
        self.T1_z_left  = 0
        self.T1_z_right = 0

    def set_eigenvalue_problem_type_2(self, Rayleigh, Prandtl, **kwargs):
        self.problem = EVP_homogeneous(self.domain, variables=self.variables, eigenvalue='nu')
        self.problem.substitutions['dt(f)'] = "(0*f)"
        self.set_equations(Rayleigh, Prandtl, EVP_2 = True, **kwargs)
        
    def _set_subs(self):
        super(FC_equations, self)._set_subs()
        
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'

        self.problem.parameters['delta_s_atm'] = self.delta_s
        self.problem.substitutions['s_fluc'] = '(1/Cv_inv*log(1+T1/T0) - 1/Cv_inv*(gamma-1)*ln_rho1)'
        self.problem.substitutions['s_mean'] = '(1/Cv_inv*log(T0) - 1/Cv_inv*(gamma-1)*ln_rho0)'

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)/(nu*chi)'
        
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
        self.problem.substitutions['kappa_flux_mean'] = '-rho0*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '-rho_full*chi*dz(T1) - rho_fluc*chi*dz(T0)'
        self.problem.substitutions['kappa_flux'] = '((kappa_flux_mean) + (kappa_flux_fluc))'
        self.problem.substitutions['KE_flux'] = 'w*KE'

    def set_equations(self, Rayleigh, Prandtl, EVP_2 = False, include_background_flux=True):
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu')
            self.problem.parameters.pop('chi')
 
        # thermal boundary conditions
        self.problem.parameters['T1_left']  = self.T1_left
        self.problem.parameters['T1_right'] = self.T1_right
        self.problem.parameters['T1_z_left']  = self.T1_z_left
        self.problem.parameters['T1_z_right'] = self.T1_z_right

        self._set_subs()
                
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
        self.linear_thermal_diff    = (" Cv_inv*(chi*(Lap(T1, T1_z)     + T1_z*del_ln_rho0 "
                                       "              + T0_z*dz(ln_rho1)) + del_chi*dz(T1)) ")
        self.nonlinear_thermal_diff =  " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                  " Cv_inv*(chi*(T0_zz             + T0_z*del_ln_rho0) + del_chi*T0_z)"
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff 
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
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

        self.problem.add_equation(("(scale)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 
        
        logger.info("using nonlinear EOS for entropy, via substitution")
        # non-linear EOS for s, where we've subtracted off
        # Cv_inv*∇s0 =  T0_z/(T0 + T1) - (gamma-1)*del_ln_rho0
        # move entropy to a substitution; no need to solve for it.
        #self.problem.add_equation(("(scale)*(Cv_inv*s - T1/T0 + (gamma-1)*ln_rho1) = "
        #                           "(scale)*(log(1+T1/T0) - T1/T0)"))

                
    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None):

        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True
        if not(stress_free) and not(no_slip):
            stress_free = True

        self.dirichlet_set = []
        
        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (T1_z)")
            self.problem.add_bc( "left(T1_z) = T1_z_left")
            self.problem.add_bc("right(T1_z) = T1_z_right")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = T1_left")
            self.problem.add_bc("right(T1) = T1_right")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: mixed flux/temperature (T1_z/T1)")
            self.problem.add_bc("left(T1_z) = T1_z_left")
            self.problem.add_bc("right(T1)  = T1_right")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: mixed temperature/flux (T1/T1_z)")
            self.problem.add_bc("left(T1)    = T1_left")
            self.problem.add_bc("right(T1_z) = T1_z_right")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc( "left(u_z) = 0")
            self.problem.add_bc("right(u_z) = 0")
            self.dirichlet_set.append('u_z')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")
        self.dirichlet_set.append('w')
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

    def set_IC(self, solver, A0=1e-6, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        self.T_IC['g'] = A0*np.sin(np.pi*z_dealias/self.Lz)*noise*self.T0['g']

        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))

    def get_full_T(self, solver):
        T1 = solver.state['T1']
        T_scales = T1.meta[:]['scale']
        T1.set_scales(self.domain.dealias, keep_data=True)
        T = self._new_field()
        T.set_scales(self.domain.dealias, keep_data=False)
        T['g'] = self.T0['g'] + T1['g']
        T.set_scales(T_scales, keep_data=True)
        T1.set_scales(T_scales, keep_data=True)
        return T

    def get_full_rho(self, solver):
        ln_rho1 = solver.state['ln_rho1']
        rho_scales = ln_rho1.meta[:]['scale']
        rho = self._new_field()
        rho['g'] = self.rho0['g']*np.exp(ln_rho1['g'])
        rho.set_scales(rho_scales, keep_data=True)
        ln_rho1.set_scales(rho_scales, keep_data=True)
        return rho

    def check_system(self, solver, **kwargs):
        T = self.get_full_T(solver)
        rho = self.get_full_rho(solver)

        self.check_atmosphere(T=T, rho=rho, **kwargs)

    def initialize_output(self, solver, data_dir, **kwargs):
        analysis_tasks = OrderedDict()
        self.analysis_tasks = analysis_tasks
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("s_fluc", name="s")
        analysis_slice.add_task("s_fluc - plane_avg(s_fluc)", name="s'")
        #analysis_slice.add_task("T1", name="T")
        #analysis_slice.add_task("ln_rho1", name="ln_rho")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("vorticity", name="vorticity")
        analysis_tasks['slice'] = analysis_slice

        analysis_coeff = solver.evaluator.add_file_handler(data_dir+"coeffs", max_writes=20, parallel=False, **kwargs)
        analysis_coeff.add_task("s_fluc", name="s", layout='c')
        analysis_coeff.add_task("s_fluc - plane_avg(s_fluc)", name="s'", layout='c')
        analysis_coeff.add_task("T1", name="T", layout='c')
        analysis_coeff.add_task("ln_rho1", name="ln_rho", layout='c')
        analysis_coeff.add_task("u", name="u", layout='c')
        analysis_coeff.add_task("w", name="w", layout='c')
        analysis_coeff.add_task("enstrophy", name="enstrophy", layout='c')
        analysis_coeff.add_task("vorticity", name="vorticity", layout='c')
        analysis_tasks['coeff'] = analysis_coeff
        
        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(T1)", name="T1")
        analysis_profile.add_task("plane_avg(ln_rho1)", name="ln_rho1")
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
        analysis_profile.add_task("plane_avg(kappa_flux_mean)", name="kappa_flux_mean_z")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)", name="Rayleigh_local")
        analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
        analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
        analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
        analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
        analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
        analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
        analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")
        analysis_profile.add_task("plane_avg(g*dz(s_fluc)*Cv_inv/gamma))", name="brunt_squared_fluc")        
        analysis_profile.add_task("plane_avg(g*dz(s_mean)*Cv_inv/gamma))", name="brunt_squared_mean")        
        analysis_profile.add_task("plane_avg(g*dz(s_fluc + s_mean)*Cv_inv/gamma))", name="brunt_squared_tot")        
        analysis_profile.add_task("plane_avg(Cv_inv*(chi*(T0_zz + T0_z*del_ln_rho0) + del_chi*T0_z))",
                                  name="T1_source_terms")
        
        analysis_tasks['profile'] = analysis_profile

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

        analysis_tasks['scalar'] = analysis_scalar

        # workaround for issue #29
        self.problem.namespace['enstrophy'].store_last = True

        return self.analysis_tasks
        
class FC_polytrope(FC_equations, Polytrope):
    def __init__(self, *args, **kwargs):
        super(FC_polytrope, self).__init__() 
        Polytrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_polytrope, self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)

class FC_polytrope_adiabatic(FC_equations, Polytrope_adiabatic):
    def __init__(self, *args, **kwargs):
        super(FC_polytrope_adiabatic, self).__init__() 
        Polytrope_adiabatic.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

        self.T1_left  = self.at_boundary(self.T0_IC, z=0,       BC_text='T1')
        self.T1_right = self.at_boundary(self.T0_IC, z=self.Lz, BC_text='T1')
        self.T1_z_left  = self.at_boundary(self.T0_IC, z=0,       dz=True, BC_text='T1')
        self.T1_z_right = self.at_boundary(self.T0_IC, z=self.Lz, dz=True, BC_text='T1')

    def set_equations(self, *args, **kwargs):
        super(FC_polytrope_adiabatic, self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
        
    def set_IC(self, *args, **kwargs):
        super(FC_polytrope_adiabatic, self).set_IC(*args, **kwargs)
        # update initial conditions to include super-adiabatic component
        logger.info("shapes: {} and {}".format(self.T_IC['g'].shape, self.T0_IC['g'].shape))
        self.T0_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC['g'] += self.T0_IC['g']
        
        self.ln_rho_IC['g'] = self.ln_rho0_IC['g']        
        logger.info("adding in nonadiabatic background to T1 and ln_rho1")

class FC_multitrope(FC_equations, Multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_multitrope, self).__init__() 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_multitrope,self).set_equations(*args, **kwargs)

        #self.problem.meta[:]['z']['dirichlet'] = True
        logger.info("skipping HS balance check")
        #self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)

    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
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
        self.T_IC['g'] = self.epsilon*A0*noise*self.T0['g']*taper
        self.filter_field(self.T_IC, **kwargs)
        self.ln_rho_IC['g'] = 0
        
        logger.info("Starting with tapered T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))

class FC_multitropedense(FC_equations, MultitropeDense):
    def __init__(self, *args, **kwargs):
        super(FC_multitropedense, self).__init__() 
        MultitropeDense.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_multitropedense,self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
    
    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        self.T_IC['g'] = self.epsilon*A0*noise*np.sin(np.pi*z_dealias/self.Lz)*self.T0['g']

        logger.info("Starting with T1 perturbations of amplitude A0*epsilon = {:g}".format(A0*self.epsilon))




class FC_MHD_equations(FC_equations):
    def __init__(self):
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes with MHD, 2.5D'
        self.variables = ['u','u_z','v','v_z','w','w_z','T1', 'T1_z', 'ln_rho1','Ax','Ay','Az','Bx','By','Phi']

        # possible boundary condition values for a normal system
        self.T1_left  = 0
        self.T1_right = 0
        self.T1_z_left  = 0
        self.T1_z_right = 0

    def _set_subs(self):
        super(FC_MHD_equations, self)._set_subs()

        self.problem.parameters['pi'] = np.pi
        self.problem.substitutions['Bz'] = '(dx(Ay) )'
        self.problem.substitutions['Jx'] = '( -dz(By))'
        self.problem.substitutions['Jy'] = '(dz(Bx) - dx(Bz))'
        self.problem.substitutions['Jz'] = '(dx(By) )'
        
        self.problem.substitutions['BdotGrad(f, f_z)'] = "(Bx*dx(f) + Bz*(f_z))"
        self.problem.substitutions['ME'] = '1/(8*pi)*(Bx**2+By**2+Bz**2)'

        self.problem.substitutions['J_squared'] = "(Jx**2 + Jy**2 + Jz**2)"

    def _set_diffusivities(self, Rayleigh, Prandtl, MagneticPrandtl, Q=None, B0=None, use_Q=False, **kwargs):
        logger.info("use_Q {}".format(use_Q))
        if not use_Q:
            super(FC_MHD_equations, self)._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, **kwargs)
        else:
            logger.info("using Q-based vs Ra-based diffusivities")
            self._set_diffusivities_Q(Q=Q, B0=B0,Prandtl=Prandtl)
        self.eta = self._new_ncc()
        self.eta.set_scales(self.domain.dealias, keep_data=False)
        self.necessary_quantities['eta'] = self.eta
        self.nu.set_scales(self.domain.dealias, keep_data=True)        
        self.eta['g'] = self.nu['g']/MagneticPrandtl

    def _set_diffusivities_Q(self, Q=1e6, B0=None, Prandtl=1):
        
        logger.info("problem parameters:")
        logger.info("   Q = {:g}, Pr = {:g}".format(Q, Prandtl))

        # set nu and chi at top based on Rayleigh number
        # wrong Prandtl number here (should be magnetic prandtl)
        nu_top = np.sqrt(Prandtl*(self.Lz**2*B0**2)/(4*np.pi*Q))
        chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
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
                    
        #Allows for atmosphere reuse
        self.chi.set_scales(1, keep_data=True)
        self.nu['g'] = nu
        self.chi['g'] = chi

        self.chi.differentiate('z', out=self.del_chi)
        self.chi.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/self.chi.interpolate(z=self.Lz/2)['g'][0][0]
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/self.nu.interpolate(z=self.Lz/2)['g'][0][0]
        self.top_viscous_time = 1/nu_top

        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))

    def _set_parameters(self):
        super(FC_MHD_equations, self)._set_parameters()
    
        self.problem.parameters['eta'] = self.eta
        
    def set_equations(self, Rayleigh, Prandtl, MagneticPrandtl, include_background_flux=True,  **kwargs):
        # DOES NOT YET INCLUDE Ohmic heating, variable eta.
        
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, MagneticPrandtl=MagneticPrandtl, **kwargs)
        self._set_parameters()
        # thermal boundary conditions
        self.problem.parameters['T1_left']  = self.T1_left
        self.problem.parameters['T1_right'] = self.T1_right
        self.problem.parameters['T1_z_left']  = self.T1_z_left
        self.problem.parameters['T1_z_right'] = self.T1_z_right

        self._set_subs()
                
        # here, nu and chi are constants        
        self.viscous_term_w = " nu*(Lap(w, w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*Div_u)"
        self.viscous_term_u = " nu*(Lap(u, u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*Div(dx(u), dx(w_z)))"
        self.viscous_term_v = " nu*(Lap(v, v_z) )" # work through this properly

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u
        self.problem.substitutions['L_visc_v'] = self.viscous_term_v
                
        self.nonlinear_viscous_w = " nu*(    u_z*dx(ln_rho1) + 2*w_z*dz(ln_rho1) + dx(ln_rho1)*dx(w) - 2/3*dz(ln_rho1)*Div_u)"
        self.nonlinear_viscous_u = " nu*(2*dx(u)*dx(ln_rho1) + dx(w)*dz(ln_rho1) + dz(ln_rho1)*u_z   - 2/3*dx(ln_rho1)*Div_u)"
        self.nonlinear_viscous_v = " 0 " # work through this properly

        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u
        self.problem.substitutions['NL_visc_v'] = self.nonlinear_viscous_v

        # double check implementation of variabile chi and background coupling term.
        self.problem.substitutions['Q_z'] = "(-T1_z)"
        self.linear_thermal_diff    = (" Cv_inv*(chi*(Lap(T1, T1_z)     + T1_z*del_ln_rho0 "
                                       "              + T0_z*dz(ln_rho1)) + del_chi*dz(T1)) ")
        self.nonlinear_thermal_diff =  " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                  " Cv_inv*(chi*(T0_zz             + T0_z*del_ln_rho0) + del_chi*T0_z)"
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff 
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source
        
        self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2/3*Div_u**2)"
        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
    
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(v) - v_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
        self.problem.add_equation("Bx + dz(Ay) = 0")
        self.problem.add_equation("By - dz(Ax) + dx(Az) = 0")

        self.problem.add_equation(("(scale)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w  + 1/(4*pi*rho_full)*(Jx*By - Jy*Bx))"))

        self.problem.add_equation(("(scale)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u  + 1/(4*pi*rho_full)*(Jy*Bz - Jz*By))"))

        self.problem.add_equation(("(scale)*( dt(v) +                                          - L_visc_v) = "
                                   "(scale)*(- UdotGrad(v, v_z) + NL_visc_v  + 1/(4*pi*rho_full)*(Jz*Bx - Jx*Bz))"))

        
        self.problem.add_equation(("(scale)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        self.problem.add_equation(("(scale)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 

        # assumes constant eta; no NCCs here to rescale.  Easy to modify.
        self.problem.add_equation("dt(Ax) + eta*Jx + dx(Phi)            =  v*Bz - w*By")
        self.problem.add_equation("dt(Ay) + eta*Jy                      =  w*Bx - u*Bz")
        self.problem.add_equation("dt(Az) + eta*Jz + dz(Phi)            =  u*By - v*Bx")
        self.problem.add_equation("dx(Ax) + dz(Az) = 0")
        
        logger.info("using nonlinear EOS for entropy, via substitution")

        # workaround for issue #29
        self.problem.namespace['Bz'].store_last = True
        self.problem.namespace['Jx'].store_last = True
        self.problem.namespace['Jy'].store_last = True
        self.problem.namespace['Jz'].store_last = True

    
    def set_BC(self, **kwargs):
        
        super(FC_MHD_equations, self).set_BC(**kwargs)

        self.problem.add_bc( "left(v_z) = 0")
        self.problem.add_bc("right(v_z) = 0")
 
        # perfectly conducting boundary conditions.
        self.problem.add_bc("left(Ax) = 0")
        self.problem.add_bc("left(Ay) = 0")
        self.problem.add_bc("left(Az) = 0")
        self.problem.add_bc("right(Ax) = 0")
        self.problem.add_bc("right(Ay) = 0")
        self.problem.add_bc("right(Az) = 0", condition="(nx != 0)")
        self.problem.add_bc("right(Phi) = 0", condition="(nx == 0)")

        self.dirichlet_set.append('Ax')
        self.dirichlet_set.append('Ay')
        self.dirichlet_set.append('Az')
        self.dirichlet_set.append('Phi')
        
    def set_IC(self, solver, A0=1e-6, **kwargs):
        super(FC_MHD_equations, self).set_IC(solver, A0=A0, **kwargs)
    
        self.Bx_IC = solver.state['Bx']
        self.Ay_IC = solver.state['Ay']

        # not in HS balance
        B0 = 1
        self.Bx_IC.set_scales(self.domain.dealias, keep_data=True)

        self.Bx_IC['g'] = A0*B0*np.cos(np.pi*self.z_dealias/self.Lz)
        self.Bx_IC.antidifferentiate('z',('left',0), out=self.Ay_IC)
        self.Ay_IC['g'] *= -1
        
    def initialize_output(self, solver, data_dir, **kwargs):
        super(FC_MHD_equations, self).initialize_output(solver, data_dir, **kwargs)

        # make analysis_tasks a dictionary!
        analysis_slice = self.analysis_tasks['slice']
        analysis_slice.add_task("Jy", name="Jy")
        analysis_slice.add_task("Bx", name="Bx")
        analysis_slice.add_task("Bz", name="Bz")
        analysis_slice.add_task("dx(Bx) + dz(Bz)", name="divB")
        analysis_slice.add_task("J_squared", name="J_squared")
            
        analysis_profile = self.analysis_tasks['profile']
        analysis_profile.add_task("plane_avg(ME)", name="ME")
        analysis_profile.add_task("plane_avg(J_squared)", name="J_squared")

        analysis_scalar = self.analysis_tasks['scalar']
        analysis_scalar.add_task("vol_avg(ME)", name="ME")
        analysis_scalar.add_task("vol_avg(J_squared)", name="J_squared")

        return self.analysis_tasks
    

class FC_MHD_polytrope(FC_MHD_equations, Polytrope):
    def __init__(self, *args, **kwargs):
        super(FC_MHD_polytrope, self).__init__() 
        Polytrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_MHD_polytrope, self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)

    def set_BC(self, *args, **kwargs):
        super(FC_MHD_polytrope, self).set_BC(*args, **kwargs)
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

class FC_MHD_multitrope(FC_MHD_equations, Multitrope):
    def __init__(self, *args, **kwargs):
        super(FC_MHD_multitrope, self).__init__() 
        Multitrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_MHD_multitrope, self).set_equations(*args, **kwargs)
        self.problem.meta[:]['z']['dirichlet'] = True
        logger.info("skipping HS balance check")
        #self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
    
    def set_IC(self, solver, A0=1e-3, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
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
        self.T_IC['g'] = self.epsilon*A0*noise*self.T0['g']*taper
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


    def set_BC(self, *args, **kwargs):
        super(FC_MHD_multitrope, self).set_BC(*args, **kwargs)
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True


# needs to be tested again and double-checked
class AN_equations(Equations):
    def __init__(self):
        self.equation_set = 'Energy-convserving anelastic equations (LBR)'
        self.variables = ['u','u_z','w','w_z','s', 'Q_z', 'pomega']

    def _set_subs(self):
        super(AN_equations, self)._set_subs()
        
        self.problem.substitutions['rho_full'] = 'rho0'
        self.problem.substitutions['rho_fluc'] = 'rho0'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'

        self.problem.parameters['delta_s_atm'] = self.delta_s
        self.problem.substitutions['s_fluc'] = 's'
        self.problem.substitutions['s_mean'] = '(1/Cv_inv*log(T0) - 1/Cv_inv*(gamma-1)*ln_rho0)'

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)/(nu*chi)'

        self.problem.substitutions['KE'] = 'rho_full*(u**2+w**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'gamma*Cv*T0*(s_fluc+s_mean)' #'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'gamma*Cv*T0*(s_fluc)' #'rho_full*Cv*T1+rho_fluc*Cv*T0'
        self.problem.substitutions['P'] = 'rho_full*pomega+T0*rho0'
        self.problem.substitutions['P_fluc'] = 'rho_full*pomega'
        self.problem.substitutions['h'] = 'IE + P'
        self.problem.substitutions['h_fluc'] = 'IE_fluc + P_fluc'
        self.problem.substitutions['u_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['w_rms'] = 'sqrt(w*w)'
        self.problem.substitutions['Re_rms'] = 'sqrt(u**2+w**2)*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'sqrt(u**2+w**2)*Lz/chi'

        self.problem.substitutions['h_flux'] = 'w*h'
        self.problem.substitutions['kappa_flux_mean'] = '-rho0*chi*Q_z'
        self.problem.substitutions['kappa_flux_fluc'] = '-rho_full*chi*Q_z'
        self.problem.substitutions['kappa_flux'] = '((kappa_flux_mean) + (kappa_flux_fluc))'
        self.problem.substitutions['KE_flux'] = 'w*KE'

    def set_equations(self, Rayleigh, Prandtl, include_background_flux=True):
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        self._set_subs()
        
        # Lecoanet et al 2014, ApJ, eqns D23-D31
        self.viscous_term_w = " (nu*(dx(dx(w)) + dz(w_z) + 2*del_ln_rho0*w_z + 1/3*(dx(u_z) + dz(w_z)) - 2/3*del_ln_rho0*(dx(u) + w_z)))"
        self.viscous_term_u = " (nu*(dx(dx(u)) + dz(u_z) + del_ln_rho0*(u_z+dx(w)) + 1/3*(dx(dx(u)) + dx(w_z))))"
        self.thermal_diff   = " chi*(dx(dx(s)) - 1/T0*dz(Q_z) - 1/T0*Q_z*del_ln_rho0)"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u
        self.problem.substitutions['L_thermal'] = self.thermal_diff

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")

        self.problem.add_equation("(scale)  * (dt(w)  - L_visc_w + dz(pomega) - s*g) = -(scale)  * UdotGrad(w, w_z)")
        self.problem.add_equation("(scale)  * (dt(u)  - L_visc_u + dx(pomega)      ) = -(scale)  * UdotGrad(u, u_z)")
        self.problem.add_equation("(scale)**2*(dt(s)  - L_thermal + w*del_s0       ) = -(scale)**2*UdotGrad(s, dz(s))")
        
        self.problem.add_equation("(scale)*(Div_u + w*del_ln_rho0) = 0")

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

    def set_IC(self, solver, A0=1e-6, **kwargs):
        # initial conditions
        self.s_IC = solver.state['s']
        noise = self.global_noise(**kwargs)
        self.s_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        self.s_IC['g'] = A0*np.sin(np.pi*z_dealias/self.Lz)*noise

        logger.info("Starting with s perturbations of amplitude A0 = {:g}".format(A0))

    def initialize_output(self, solver, data_dir, **kwargs):
        analysis_tasks = OrderedDict()
        self.analysis_tasks = analysis_tasks
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("s", name="s")
        analysis_slice.add_task("s - plane_avg(s)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("vorticity", name="vorticity")
        analysis_tasks['slice'] = analysis_slice
        
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
        analysis_profile.add_task("plane_avg(kappa_flux_mean)", name="kappa_flux_mean_z")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)", name="Rayleigh_local")
        analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
        analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
        analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
        analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
        analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
        analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
        analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")        
        analysis_profile.add_task("plane_avg(Cv_inv*(chi*(T0_zz + T0_z*del_ln_rho0) + del_chi*T0_z))",
                                  name="T1_source_terms")
        
        analysis_tasks['profile'] = analysis_profile

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

        analysis_tasks['scalar'] = analysis_scalar

        # workaround for issue #29
        self.problem.namespace['enstrophy'].store_last = True

        return self.analysis_tasks

class AN_polytrope(AN_equations, Polytrope):
    def __init__(self, *args, **kwargs):
        super(AN_polytrope, self).__init__() 
        Polytrope.__init__(self, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))
        
        
