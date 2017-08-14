import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tools import analysis
except:
    from sys import path
    path.insert(0, './tools')
    from ..tools import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Atmosphere:
    def __init__(self, verbose=False, fig_dir='./', dimensions=2, **kwargs):
        self._set_domain(**kwargs)
        
        self.make_plots = verbose
        self.fig_dir = fig_dir + '/'
        self.dimensions = dimensions
        
        if self.fig_dir[-1] != '/':
            self.fig_dir += '/'
        if self.domain.dist.comm_cart.rank == 0 and not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
            
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
        self.nu_l = self._new_ncc()
        self.chi_l = self._new_ncc()
        self.del_chi_l = self._new_ncc()
        self.del_nu_l = self._new_ncc()
        self.necessary_quantities['nu_l'] = self.nu_l
        self.necessary_quantities['chi_l'] = self.chi_l
        self.necessary_quantities['del_chi_l'] = self.del_chi_l
        self.necessary_quantities['del_nu_l'] = self.del_nu_l
        self.nu_r = self._new_ncc()
        self.chi_r = self._new_ncc()
        self.del_chi_r = self._new_ncc()
        self.del_nu_r = self._new_ncc()
        self.necessary_quantities['nu_r'] = self.nu_r
        self.necessary_quantities['chi_r'] = self.chi_r
        self.necessary_quantities['del_chi_r'] = self.del_chi_r
        self.necessary_quantities['del_nu_r'] = self.del_nu_r

        self.scale = self._new_ncc()
        self.scale_continuity = self._new_ncc()
        self.scale_energy = self._new_ncc()
        self.scale_momentum = self._new_ncc()
        self.necessary_quantities['scale'] = self.scale
        self.necessary_quantities['scale_continuity'] = self.scale_continuity
        self.necessary_quantities['scale_energy'] = self.scale_energy
        self.necessary_quantities['scale_momentum'] = self.scale_momentum

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
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(quantity['g'][0,:])-0.05*np.abs(np.min(quantity['g'][0,:])),
                        np.max(quantity['g'][0,:])+0.05*np.abs(np.max(quantity['g'][0,:])))
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

        for key in self.necessary_quantities:
            if key not in ['P0', 'rho0']:
                continue
            logger.debug("plotting atmosphereic quantity ln({})".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(1,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], np.log(quantity['g'][0,:]))
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(np.log(quantity['g'][0,:]))-0.05*np.abs(np.min(np.log(quantity['g'][0,:]))),
                        np.max(np.log(quantity['g'][0,:]))+0.05*np.abs(np.max(np.log(quantity['g'][0,:]))))
            ax.set_xlabel('z')
            ax.set_ylabel('ln_'+key)
            fig_q.savefig(self.fig_dir+"atmosphere_ln_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300, bbox_inches='tight')
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
                logger.info("WARNING: atmosphere {} is all zeros on process 0".format(key))
                
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
                    if rho_scales != 1:
                        rho.set_scales(1, keep_data=True)
                    if T_scales != 1:
                        T.set_scales(1, keep_data=True)
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
        if self.dimensions > 1:
            HS_average.integrate('x')
            HS_average['g'] /= self.Lx
        HS_average.set_scales(1, keep_data=True)

        relative_error_avg = self._new_field()
        relative_error_avg['g'] = relative_error
        if self.dimensions > 1:
            relative_error_avg.integrate('x')
            relative_error_avg['g'] /= self.Lx
        relative_error_avg.set_scales(1, keep_data=True)

        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            if self.dimensions > 1:
                ax1.plot(self.z[0,:], P_z['g'][0,:])
                ax1.plot(self.z[0,:], -self.g*rho['g'][0,:])
            else:
                ax1.plot(self.z[:], P_z['g'][:])
                ax1.plot(self.z[:], -self.g*rho['g'][:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            if self.dimensions > 1:
                ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
                ax2.semilogy(self.z[0,:], np.abs(relative_error_avg['g'][0,:]))
            else:
                ax2.semilogy(self.z[:], np.abs(relative_error[:]))
                ax2.semilogy(self.z[:], np.abs(relative_error_avg['g'][:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig(self.fig_dir+"atmosphere_HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)

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
        

class Polytrope(Atmosphere):
    '''
    Single polytrope, stable or unstable.
    '''
    def __init__(self,
                 nx=256, Lx=None,
                 ny=256, Ly=None,
                 nz=128, Lz=None,
                 aspect_ratio=4,
                 n_rho_cz = 3,
                 m_cz=None, epsilon=1e-4, gamma=5/3,
                 constant_kappa=True, constant_mu=True,
                 **kwargs):
        
        self.atmosphere_name = 'single polytrope'
        self.aspect_ratio    = aspect_ratio
        self.n_rho_cz        = n_rho_cz

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
        if Ly is None:
            Ly = Lx
            
        super(Polytrope, self).__init__(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, **kwargs)
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        self.z0 = 1. + self.Lz
       
        self.constant_kappa = constant_kappa
        self.constant_mu    = constant_mu
        if self.constant_kappa == False and self.constant_mu == False:
            self.constant_diffusivities = True
        else:
            self.constant_diffusivities = False

        self._set_atmosphere()
        self._set_timescales()

    def _calculate_Lz_cz(self, n_rho_cz, m_cz):
        '''
        Calculate Lz based on the number of density scale heights and the initial polytrope.
        '''
        #The absolute value allows for negative m_cz.
        Lz_cz = np.exp(n_rho_cz/np.abs(m_cz))-1
        return Lz_cz
    
    def _set_atmosphere_parameters(self, gamma=5/3, epsilon=0, poly_m=None, g=None):
        # polytropic atmosphere characteristics
        self.gamma = gamma
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv
        self.epsilon = epsilon

        self.m_ad = 1/(self.gamma-1)

        # trap on poly_m/epsilon conflicts?
        if poly_m is None:
            self.poly_m = self.m_ad - self.epsilon
        else:
            self.poly_m = poly_m

        self.m_cz = self.poly_m

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

        self.del_s0_factor = - self.epsilon 
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
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)
            self.scale_energy['g']     = (self.z0 - self.z)
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)# **np.ceil(self.m_cz)
            self.scale_energy['g']     = (self.z0 - self.z)# **np.ceil(self.m_cz)

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
        if self.delta_x != None:
            logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
            
        # min of global quantity
        atmosphere.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))), op=MPI.MIN)
        atmosphere.freefall_time = np.sqrt(self.Lz/self.g)
        atmosphere.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(atmosphere.min_BV_time,
                                                                                               atmosphere.freefall_time,
                                                                                               atmosphere.buoyancy_time))
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set nu and chi at top based on Rayleigh number
        self.nu_top = nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s/self.Cp)*self.g)/Rayleigh)
        self.chi_top = chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        else:
            if self.constant_kappa:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    chi_l = chi_top/(self.rho0['g'])
                    chi_r = 0
                else:
                    if self.poly_m < 1:
                        chi_l = np.exp(self.n_rho_cz)*chi_top/(self.z0 - self.z)
                    else:
                        chi_l = chi_top/(self.z0 - self.z)
                    chi_r = chi_top/(self.rho0['g']) - chi_l
                logger.info('using constant kappa')
            else:
                chi_l = chi_top
                chi_r = 0
                logger.info('using constant chi')
            if self.constant_mu:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    nu_l  = nu_top/(self.rho0['g'])
                    nu_r = 0
                else:
                    if self.poly_m < 1:
                        nu_l  = np.exp(self.n_rho_cz)*nu_top/(self.z0 - self.z)
                    else:
                        nu_l  = nu_top/(self.z0 - self.z)
                    nu_r  = nu_top/(self.rho0['g']) - nu_l
                logger.info('using constant mu')
            else:
                nu_l  = nu_top
                nu_r = 0
                logger.info('using constant nu')

      
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))
                    
        #Allows for atmosphere reuse
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu_l['g'] = nu_l
        self.chi_l['g'] = chi_l
        self.nu_r['g'] = nu_r
        self.chi_r['g'] = chi_r
        self.nu['g'] = nu_l + nu_r
        self.chi['g'] = chi_l + chi_r

        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/(self.chi.interpolate(z=self.Lz/2)['g'][0])
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/(self.nu.interpolate(z=self.Lz/2)['g'][0])
        self.top_viscous_time = 1/nu_top

        if self.dimensions > 1:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]
        if self.dimensions > 2:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time,
                                                                      self.top_thermal_time))
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)

                      
class Multitrope(Atmosphere):
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
                 constant_Prandtl=True,
                 **kwargs):

        self.atmosphere_name = 'multitrope'
        
        if stable_top:
            logger.debug("stable_top = True; CZ below, RZ above")
            stable_bottom = False
            
        self.stable_bottom = stable_bottom

        self._set_atmosphere_parameters(gamma=gamma, n_rho_cz=n_rho_cz,
                                        n_rho_rz=n_rho_rz, m_rz=m_rz, stiffness=stiffness)
        
        Lz_cz, Lz_rz, Lz = self._calculate_Lz(n_rho_cz, self.m_cz, n_rho_rz, self.m_rz)
        self.Lz_cz = Lz_cz
        self.Lz_rz = Lz_rz
        self.aspect_ratio = aspect_ratio
        
        Lx = Lz_cz*aspect_ratio
        
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
        
        if len(nz) == 3:
            overshoot_pad = self.match_width
            logger.info("dense multitrope: using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))
            logger.info("nz = {}".format(nz))
            if self.stable_bottom:
                Lz_bottom = Lz_rz - overshoot_pad
                Lz_mid = 2*overshoot_pad
                Lz_top = Lz_cz + overshoot_pad
            else:
                Lz_bottom = Lz_cz + overshoot_pad
                Lz_mid = 2*overshoot_pad
                Lz_top = Lz_rz - overshoot_pad
            Lz_set = [Lz_bottom, Lz_mid, Lz_top]
        elif len(nz) == 4:
            overshoot_pad = self.match_width
            logger.info("dense multitrope: using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))
            logger.info("nz = {}".format(nz))
            if self.stable_bottom:
                Lz_bottom = Lz_rz - overshoot_pad
                Lz_mid = overshoot_pad
                Lz_mid_2 = overshoot_pad
                Lz_top = Lz_cz + overshoot_pad
            else:
                Lz_bottom = Lz_cz + overshoot_pad
                Lz_mid = overshoot_pad
                Lz_mid_2 = overshoot_pad
                Lz_top = Lz_rz - overshoot_pad
            Lz_set = [Lz_bottom, Lz_mid, Lz_mid_2, Lz_top]
        else:
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
                    
            logger.info("using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))

            if self.stable_bottom:
                Lz_bottom = Lz_rz - overshoot_pad
                Lz_top = Lz_cz + overshoot_pad
            else:
                Lz_bottom = Lz_cz + overshoot_pad
                Lz_top = Lz_rz - overshoot_pad
                
            if len(nz) == 1:
                Lz_set = [Lz_bottom + Lz_top]            
            else:
                Lz_set = [Lz_bottom, Lz_top]            

        super(Multitrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=Lz_set, **kwargs)

        self.constant_Prandtl = constant_Prandtl
        self.constant_diffusivities = False
       
        if self.dimensions == 2:
            out_string = "   Lx = {:g}, Lz = {:g} (Lz_cz = {:g}, Lz_rz = {:g})".format(self.Lx, self.Lz, self.Lz_cz, self.Lz_rz)
        elif self.dimensions == 1:
            out_string = "   Lz = {:g} (Lz_cz = {:g}, Lz_rz = {:g})".format(self.Lz, self.Lz_cz, self.Lz_rz)
        logger.info(out_string)

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
        if self.dimensions > 1:
            logger.info("   H_rho/delta x = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top/self.delta_x,\
                                                                          H_rho_bottom/self.delta_x))

        self._set_timescales()
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
        # min of global quantity
        BV_time = np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))
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
         
    def _compute_step_profile(self, step_ratio, invert_profile=False):
        # a simple, smooth step function with amplitudes 1 and step_ratio
        # on either side of the matching region
        Phi = self.match_Phi(self.z)
        inv_Phi = 1-Phi
        
        self.profile = self._new_ncc()
        if invert_profile:
            self.profile['g'] = Phi + inv_Phi*step_ratio
        else:
            self.profile['g'] = Phi*step_ratio + inv_Phi
        self.necessary_quantities['profile'] = self.profile


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
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv

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
        super(Multitrope, self)._set_atmosphere()
        
        kappa_ratio = (self.m_rz + 1)/(self.m_cz + 1)

        self.delta_s = self.epsilon*np.log(self.z_cz)
        logger.info("Atmosphere delta s is {}".format(self.delta_s))

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z_cz - self.z)

        # this doesn't work: numerical instability blowup, and doesn't reduce bandwidth much at all
        #self.scale['g'] = (self.z_cz - self.z)
        # this seems to work fine; bandwidth only a few terms worse.
        self.scale['g'] = 1.
        self.scale_continuity['g'] = 1.
        self.scale_momentum['g'] = 1.
        self.scale_energy['g'] = 1.

        self.kappa = self._new_ncc()
        if atmosphere_type2:
            logger.info("ATMOSPHERE TYPE 2")
            # specify T0_z as smoothly matched profile
            # grad T ratio has inverse relationship to kappa_ratio
            self._compute_step_profile(1/kappa_ratio, invert_profile=not(self.stable_bottom))
            flux_top = -1
            self.T0_z['g'] = self.profile['g']/flux_top
            # now invert grad T for kappa
            logger.info("Solving for kappa")
            self.kappa['g'] = flux_top/self.T0_z['g']
        else:
            # specify kappa as smoothly matched profile
            self._compute_step_profile(kappa_ratio, invert_profile=not(self.stable_bottom))
            self.kappa['g'] = self.profile['g']
            logger.info("Solving for T0")
            # start with an arbitrary -1 at the top, which will be rescaled after _set_diffusivites
            flux_top = -1
            self.T0_z['g'] = flux_top/self.profile['g']
            
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

    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
        #TODO: Implement split_diffusivities
        logger.info("problem parameters (multitrope):")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh = Rayleigh_top = Rayleigh
        self.Prandtl = Prandtl_top = Prandtl

        self.constant_mu = False
        self.constant_kappa = False
        # inputs:
        # Rayleigh_top = g dS L_cz**3/(chi_top**2 * Pr_top)
        # Prandtl_top = nu_top/chi_top
        self.chi_top = np.sqrt((self.g*(self.delta_s/self.Cp)*self.Lz_cz**3)/(Rayleigh_top*Prandtl_top))
        if not self.stable_bottom:
            # try to rescale chi appropriately so that the
            # Rayleigh number is set at the top of the CZ
            # to the desired value by removing the density
            # scaling from the rz.  This is a guess.
            self.chi_top = np.exp(self.n_rho_rz)*self.chi_top

        #Set Prandtl number at same place as Ra.
        self.nu_top = self.chi_top*Prandtl_top
        
        #Reset kappa. Allows reuse of atmosphere.  But this is incorrect if we do type=2 atmospheres
        kappa_ratio = (self.m_rz + 1)/(self.m_cz + 1)
        self._compute_step_profile(kappa_ratio, invert_profile=not(self.stable_bottom))

        logger.info('setting chi')

        self.kappa['g'] *= self.chi_top
        self.kappa.set_scales(self.domain.dealias, keep_data=True)
        self.rho0.set_scales(self.domain.dealias, keep_data=True)
        self.chi_l.set_scales(self.domain.dealias, keep_data=True)
        if self.rho0['g'].shape[-1] != 0:
            self.chi_l['g'] = self.kappa['g']/self.rho0['g']
            self.chi_l.differentiate('z', out=self.del_chi_l)
            self.chi_l.set_scales(1, keep_data=True)
        
        logger.info("setting nu")
        if self.constant_Prandtl:
            self.kappa.set_scales(self.domain.dealias, keep_data=True)
            self.rho0.set_scales(self.domain.dealias, keep_data=True)
            self.nu_l.set_scales(self.domain.dealias, keep_data=True)
            if self.rho0['g'].shape[-1] != 0:
                self.nu_l['g'] = (self.nu_top/self.chi_top)*self.kappa['g']/self.rho0['g']
                self.nu_l.differentiate('z', out=self.del_nu_l)
                self.nu_l.set_scales(1, keep_data=True)
        else:
            self.nu_l['g'] = self.nu_top
            self.nu_l.differentiate('z', out=self.del_nu_l)

        # rescale kappa to correct values based on Rayleigh number derived chi

        self.nu_r['g'] = 0
        self.chi_r['g'] = 0
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu['g'] = self.nu_l['g'] + self.nu_r['g']
        self.chi['g'] = self.chi_l['g'] + self.chi_r['g']
        
        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)



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
        chi = self.chi_l
        chi_l.set_scales(1,keep_data=True)
        flux = self._new_ncc()
        flux['g'] = rho['g']*T_z['g']*chi_l['g']
        return flux


