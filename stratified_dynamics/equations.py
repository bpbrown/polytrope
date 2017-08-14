import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de


class Equations():
    def __init__(self, dimensions=2):
        self.dimensions=dimensions
        self.problem_type = ''
        pass

    def match_Phi(self, z, f=scp.erf, center=None, width=None):
        if width is None:
            width = 0.04 * 18.5/(np.sqrt(np.pi)*6.5/2)
        return 1/2*(1-f((z-center)/width))

    def _set_domain(self, nx=256, Lx=4,
                          ny=256, Ly=4,
                          nz=128, Lz=1,
                          grid_dtype=np.float64, comm=MPI.COMM_WORLD, mesh=None):
        self.mesh=mesh
        
        if not isinstance(nz, list):
            nz = [nz]
        if not isinstance(Lz, list):
            Lz = [Lz]   

        if len(nz)>1:
            logger.info("Setting compound basis in vertical (z) direction")
            z_basis_list = []
            Lz_interface = 0.
            for iz, nz_i in enumerate(nz):
                Lz_top = Lz[iz]+Lz_interface
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=3/2)
                z_basis_list.append(z_basis)
                Lz_interface = Lz_top
            self.compound = True
            z_basis = de.Compound('z', tuple(z_basis_list),  dealias=3/2)
        elif len(nz)==1:
            logger.info("Setting single chebyshev basis in vertical (z) direction")
            self.compound = False
            z_basis = de.Chebyshev('z', nz[0], interval=[0, Lz[0]], dealias=3/2)
        
        #z_basis = de.Chebyshev('z', nz, interval=[0., Lz], dealias=3/2)
        if self.dimensions > 1:
            x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        if self.dimensions > 2:
            y_basis = de.Fourier(  'y', ny, interval=[0., Ly], dealias=3/2)
        if self.dimensions == 1:
            bases = [z_basis]
        elif self.dimensions == 2:
            bases = [x_basis, z_basis]
        elif self.dimensions == 3:
            bases = [x_basis, y_basis, z_basis]
        else:
            logger.error('>3 dimensions not implemented')
        
        self.domain = de.Domain(bases, grid_dtype=grid_dtype, comm=comm, mesh=mesh)
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)

        if self.dimensions == 1:
            self.x, self.Lx, self.nx, self.delta_x = None, 0, None, None
            self.y, self.Ly, self.ny, self.delta_y = None, 0, None, None
        if self.dimensions > 1:
            self.x = self.domain.grid(0)
            self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.nx = self.domain.bases[0].coeff_size
            self.delta_x = self.Lx/self.nx
        if self.dimensions > 2:
            self.y = self.domain.grid(1)
            self.Ly = self.domain.bases[1].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.ny = self.domain.bases[1].coeff_size
            self.delta_y = self.Ly/self.ny
    
    def set_IVP_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem_type = 'EVP'
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=1)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def get_problem(self):
        return self.problem

    def _new_ncc(self):
        field = self.domain.new_field()
        if self.dimensions > 1:
            field.meta['x']['constant'] = True
        if self.dimensions > 2:
            field.meta['y']['constant'] = True            
        return field

    def _new_field(self):
        field = self.domain.new_field()
        return field

    def _set_subs(self):
        pass

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

        return noise_field

    def filter_field(self, field,frac=0.25, fancy_filter=False):
        dom = field.domain
        logger.info("filtering field with frac={}".format(frac))
        if fancy_filter:
            logger.debug("filtering using field_filter approach.  Please check.")
            local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
            coeff = []                
            for i in range(dom.dim)[::-1]:
                logger.info("building filter: i = {}".format(i))
                if i == np.max(dom.dim)-1 and self.compound:
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
                
            logger.info(coeff)
            cc = np.meshgrid(*coeff)
            field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

            for i in range(len(cc)):
                logger.info("cc {} shape {}".format(i, cc[i].shape))
                logger.info("local slice {}".format(local_slice[i]))
                logger.info("field_filter shape {}".format(field_filter.shape))

        
            for i in range(dom.dim):
                logger.info("trying i={}".format(i))
                field_filter = field_filter | (cc[i][local_slice[i]] > frac)
        
            # broken for 3-D right now; works for 2-D.  Nope, broken now in 2-D as well... what did I do?
            field['c'][field_filter] = 0j
        else:
            logger.debug("filtering using set_scales approach.  Please check.")
            orig_scale = field.meta[:]['scale']
            field.set_scales(frac, keep_data=True)
            field['c']
            field['g']
            field.set_scales(orig_scale, keep_data=True)
            
class FC_equations(Equations):
    def __init__(self, **kwargs):
        super(FC_equations, self).__init__(**kwargs)

    def set_eigenvalue_problem_type_2(self, Rayleigh, Prandtl, **kwargs):
        self.problem_type = 'EVP_2'
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='nu')
        self.problem.substitutions['dt(f)'] = "(0*f)"
        self.set_equations(Rayleigh, Prandtl, EVP_2 = True, **kwargs)

    def _set_subs(self):
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

        # output parameters        
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'
        self.problem.substitutions['T_full']            = '(T0 + T1)'
        self.problem.substitutions['ln_rho_full']       = '(ln_rho0 + ln_rho1)'

        self.problem.parameters['delta_s_atm'] = self.delta_s
        self.problem.substitutions['s_fluc'] = '((1/Cv_inv)*log(1+T1/T0) - ln_rho1)'
        self.problem.substitutions['s_mean'] = '((1/Cv_inv)*log(T0) - ln_rho0)'
        self.problem.substitutions['epsilon'] = 'plane_avg(log(T0**(1/(gamma-1))/rho0)/log(T0))'
        self.problem.substitutions['m_ad']    = '((gamma-1)**-1)'

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm*Cp_inv/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)*Cp_inv/(nu*chi)'
        
        self.problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['KE'] = 'rho_full*(vel_rms**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'rho_full*Cv*T1+rho_fluc*Cv*T0'
        self.problem.substitutions['P'] = 'rho_full*(T1+T0)'
        self.problem.substitutions['P_fluc'] = 'rho_full*T1+rho_fluc*T0'
        self.problem.substitutions['h'] = '(IE + P)'
        self.problem.substitutions['h_fluc'] = '(IE_fluc + P_fluc)'
        self.problem.substitutions['u_rms'] = 'sqrt(u**2)'
        self.problem.substitutions['v_rms'] = 'sqrt(v**2)'
        self.problem.substitutions['w_rms'] = 'sqrt(w**2)'
        self.problem.substitutions['Re_rms'] = 'vel_rms*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'vel_rms*Lz/chi'
        self.problem.substitutions['Ma_iso_rms'] = '(vel_rms/sqrt(T_full))'
        self.problem.substitutions['Ma_ad_rms'] = '(vel_rms/(sqrt(gamma*T_full)))'
        #self.problem.substitutions['lambda_microscale'] = 'sqrt(plane_avg(vel_rms)/plane_avg(enstrophy))'
        #self.problem.substitutions['Re_microscale'] = 'vel_rms*lambda_microscale/nu'
        #self.problem.substitutions['Pe_microscale'] = 'vel_rms*lambda_microscale/chi'
        
        self.problem.substitutions['h_flux_z'] = 'w*(h)'
        self.problem.substitutions['kappa_flux_mean'] = '-rho0*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '(-rho_full*chi*dz(T1) - rho_fluc*chi*dz(T0))'
        self.problem.substitutions['kappa_flux_z'] = '(kappa_flux_mean + kappa_flux_fluc)'
        self.problem.substitutions['KE_flux_z'] = 'w*(KE)'
        self.problem.substitutions['PE_flux_z'] = 'w*(PE)'
        self.problem.substitutions['viscous_flux_z'] = '- rho_full * nu * (u*σxz + w*σzz)'
        self.problem.substitutions['convective_flux_z'] = '(viscous_flux_z + KE_flux_z + PE_flux_z + h_flux_z)'
        
        self.problem.substitutions['evolved_avg_kappa'] = 'vol_avg(rho_full*chi)'
        self.problem.substitutions['kappa_adiabatic_flux_z_G75']  = '(rho0*chi*g/Cp)'
        self.problem.substitutions['kappa_adiabatic_flux_z_AB17'] = '(evolved_avg_kappa*g/Cp)'
        self.problem.substitutions['kappa_reference_flux_z_G75'] = '(-chi*rho0*(right(T1+T0)-left(T1+T0))/Lz)'
        self.problem.substitutions['Nusselt_norm_G75']   = '(kappa_reference_flux_z_G75 - kappa_adiabatic_flux_z_G75)'
        self.problem.substitutions['Nusselt_norm_AB17']   = 'vol_avg(kappa_flux_z - kappa_adiabatic_flux_z_AB17)'
        self.problem.substitutions['all_flux_minus_adiabatic_G75'] = '(convective_flux_z+kappa_flux_z-kappa_adiabatic_flux_z_G75)'
        self.problem.substitutions['all_flux_minus_adiabatic_AB17'] = '(convective_flux_z+kappa_flux_z-kappa_adiabatic_flux_z_AB17)'
        self.problem.substitutions['Nusselt_G75'] = '((all_flux_minus_adiabatic_G75)/(Nusselt_norm_G75))'
        self.problem.substitutions['Nusselt_AB17'] = '((all_flux_minus_adiabatic_AB17)/(Nusselt_norm_AB17))'
        
    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None, chemistry=False):

        self.dirichlet_set = []

        self.set_thermal_BC(fixed_flux=fixed_flux, fixed_temperature=fixed_temperature,
                            mixed_flux_temperature=mixed_flux_temperature, mixed_temperature_flux=mixed_temperature_flux)
        
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)

        if chemistry:
            self.set_chemistry_BC()
        
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True
     
        if 'EVP' in self.problem_type:
            l_flux_rhs_str = "0"
            r_flux_rhs_str = "0"
        else:
            l_flux_rhs_str = " left((exp(-ln_rho1)-1+ln_rho1)*T0_z)"
            r_flux_rhs_str = "right((exp(-ln_rho1)-1+ln_rho1)*T0_z)"
        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z + ln_rho1*T0_z) = {:s}".format(l_flux_rhs_str))
            self.problem.add_bc("right(T1_z + ln_rho1*T0_z) = {:s}".format(r_flux_rhs_str))
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('ln_rho1')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z + ln_rho1*T0_z) =  {:s}".format(l_flux_rhs_str))
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
            self.dirichlet_set.append('ln_rho1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z + ln_rho1*T0_z) = {:s}".format(r_flux_rhs_str))
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
            self.dirichlet_set.append('ln_rho1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        if not(stress_free) and not(no_slip):
            stress_free = True
            
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

    def set_chemistry_BC(self):
        logger.info("Chemistry BC: 0 flux out of box")
        self.problem.add_bc("left(f_z)=0")
        self.problem.add_bc("right(f_z)=0")
        self.problem.add_bc("left(C_z)=0")
        self.problem.add_bc("right(C_z)=0")
        self.problem.add_bc("left(G_z)=0")
        self.problem.add_bc("right(G_z)=0")
        self.dirichlet_set.append('f_z')
        self.dirichlet_set.append('C_z')
        self.dirichlet_set.append('G_z')
        
    def set_IC(self, solver, A0=1e-6, chemistry=False, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']
            
        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC['g'] = self.epsilon*A0*np.sin(np.pi*self.z_dealias/self.Lz)*noise['g']*self.T0['g']
        
        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))

        if chemistry:
            self.f_IC = solver.state['f']
            self.C_IC = solver.state['C']
            self.G_IC = solver.state['G']
            self.f_IC.set_scales(self.domain.dealias, keep_data=True)
            self.C_IC.set_scales(self.domain.dealias, keep_data=True)
            self.G_IC.set_scales(self.domain.dealias, keep_data=True)

            c0 = 1
            chem_taper = self.match_Phi(self.z_dealias, \
                                        width=0.04 * 18.5/(np.sqrt(np.pi)*6.5/2)*self.Lz,center=self.Lz/2)
            self.f_IC['g'] = c0 * chem_taper
            self.C_IC['g'] = self.f_IC['g']
            self.G_IC['g'] = c0 * (self.Lz - self.z_dealias) / self.Lz

    def get_full_T(self, solver):
        T1 = solver.state['T1']
        T_scales = T1.meta[:]['scale']
        T1.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T = self._new_field()
        T.set_scales(self.domain.dealias, keep_data=False)
        T['g'] = self.T0['g'] + T1['g']
        T.set_scales(T_scales, keep_data=True)
        T1.set_scales(T_scales, keep_data=True)
        return T

    def get_full_rho(self, solver):
        ln_rho1 = solver.state['ln_rho1']
        rho_scales = ln_rho1.meta[:]['scale']
        self.rho0.set_scales(rho_scales, keep_data=True)
        rho = self._new_field()
        rho['g'] = self.rho0['g']*np.exp(ln_rho1['g'])
        rho.set_scales(rho_scales, keep_data=True)
        ln_rho1.set_scales(rho_scales, keep_data=True)
        return rho

    def check_system(self, solver, **kwargs):
        T = self.get_full_T(solver)
        rho = self.get_full_rho(solver)

        self.check_atmosphere(T=T, rho=rho, **kwargs)

    def set_eigenvalue_problem_type_2(self, Rayleigh, Prandtl, **kwargs):
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='nu')
        self.problem.substitutions['dt(f)'] = "(0*f)"
        self.set_equations(Rayleigh, Prandtl, EVP_2 = True, **kwargs)
        
    def initialize_output(self, solver, data_dir, full_output=False, coeffs_output=False,chemistry=False,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = OrderedDict()
        self.analysis_tasks = analysis_tasks

        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=max_writes, parallel=False,
                                                             mode=mode, **kwargs)
        analysis_profile.add_task("plane_avg(T1)", name="T1")
        analysis_profile.add_task("plane_avg(T_full)", name="T_full")
        analysis_profile.add_task("plane_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_profile.add_task("plane_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_profile.add_task("plane_avg(ln_rho1)", name="ln_rho1")
        analysis_profile.add_task("plane_avg(rho_full)", name="rho_full")
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(IE)", name="IE")
        analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
        analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
        analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
        analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")

        analysis_profile.add_task("plane_avg(KE_flux_z)", name="KE_flux_z")
        analysis_profile.add_task("plane_avg(PE_flux_z)", name="PE_flux_z")
        analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
        analysis_profile.add_task("plane_avg(w*(P))",  name="P_flux_z")
        analysis_profile.add_task("plane_avg(h_flux_z)",  name="enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(viscous_flux_z)",  name="viscous_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z)", name="kappa_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z) - vol_avg(kappa_flux_z)", name="kappa_flux_fluc_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_G75)", name="kappa_flux_z_minus_ad_G75")
        analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_AB17)", name="kappa_flux_z_minus_ad_AB17")
        analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_G75)/vol_avg(Nusselt_norm_G75)", name="norm_kappa_flux_z_G75")
        analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_AB17)/vol_avg(Nusselt_norm_AB17)", name="norm_kappa_flux_z_AB17")
        analysis_profile.add_task("plane_avg(Nusselt_G75)", name="Nusselt_G75")
        analysis_profile.add_task("plane_avg(Nusselt_AB17)", name="Nusselt_AB17")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(vel_rms)", name="vel_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)",  name="Rayleigh_local")
        analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
        analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
        analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
        analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
        analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
        analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
        analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")
        analysis_profile.add_task("plane_avg(g*dz(s_fluc)*Cp_inv)", name="brunt_squared_fluc")        
        analysis_profile.add_task("plane_avg(g*dz(s_mean)*Cp_inv)", name="brunt_squared_mean")        
        analysis_profile.add_task("plane_avg(g*dz(s_fluc + s_mean)*Cp_inv)", name="brunt_squared_tot")

        if chemistry:
            analysis_profile.add_task("plane_avg(f)", name="f")
            analysis_profile.add_task("plane_avg(C)", name="C")
            analysis_profile.add_task("plane_avg(G)", name="G")
            analysis_profile.add_task("plane_avg(u*dx(f) + w*f_z)", name="Ugradf")
            analysis_profile.add_task("plane_avg(u*dx(C) + w*C_z)", name="UgradC")
            analysis_profile.add_task("plane_avg(u*dx(G) + w*G_z)", name="UgradG")
            analysis_profile.add_task("plane_avg(w*f_z)", name="Wdzf")
            analysis_profile.add_task("plane_avg(w*C_z)", name="WdzC")
            analysis_profile.add_task("plane_avg(w*G_z)", name="WdzG")

        analysis_tasks['profile'] = analysis_profile

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False,
                                                            mode=mode, **kwargs)
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
        analysis_scalar.add_task("vol_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_scalar.add_task("vol_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")
        analysis_scalar.add_task("vol_avg(Nusselt_G75)", name="Nusselt_G75")
        analysis_scalar.add_task("vol_avg(Nusselt_AB17)", name="Nusselt_AB17")
        analysis_scalar.add_task("vol_avg(Nusselt_norm_G75)", name="Nusselt_norm_G75")
        analysis_scalar.add_task("vol_avg(Nusselt_norm_AB17)", name="Nusselt_norm_AB17")
        analysis_scalar.add_task("log(left(plane_avg(rho_full))/right(plane_avg(rho_full)))", name="n_rho")
        analysis_scalar.add_task("integ(right(kappa_flux_z) - left(kappa_flux_z),'x')/Lx",name="flux_equilibration")
        analysis_scalar.add_task("integ((right(kappa_flux_z) - left(kappa_flux_z))/left(kappa_flux_z),'x')/Lx",name="flux_equilibration_pct")

        if chemistry:
            analysis_scalar.add_task("vol_avg(f)", name="f")
            analysis_scalar.add_task("vol_avg(C)", name="C")
            analysis_scalar.add_task("vol_avg(G)", name="G")
            analysis_scalar.add_task("vol_avg(-rho_full * f * log(f))", name="S_f")
            analysis_scalar.add_task("vol_avg(-rho_full * C * log(C))", name="S_C")
            analysis_scalar.add_task("vol_avg(-rho_full * G * log(G))", name="S_G")
            
        analysis_tasks['scalar'] = analysis_scalar

        return self.analysis_tasks
    
class FC_equations_2d(FC_equations):
    def __init__(self, chemistry=False, **kwargs):
        super(FC_equations_2d, self).__init__(**kwargs)

        if chemistry:
            self.equation_set = 'Fully Compressible (FC) Navier-Stokes with chemical reactions'
            self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1',\
                              'C','C_z','G','G_z','f','f_z']
        else:
            self.equation_set = 'Fully Compressible (FC) Navier-Stokes'
            self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1']

            
    def _set_subs(self):
        # 2-D specific subs
        self.problem.substitutions['ω_y']         = '( u_z  - dx(w))'        
        self.problem.substitutions['enstrophy']   = '(ω_y**2)'
        self.problem.substitutions['v']           = '(0)'
        self.problem.substitutions['v_z']         = '(0)'

        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dz(f_z))"
        self.problem.substitutions['Div(f, f_z)'] = "(dx(f) + f_z)"
        self.problem.substitutions['Div_u'] = "Div(u, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + w*(f_z))"
        # analysis operators
        if self.dimensions == 1:
            self.problem.substitutions['plane_avg(A)'] = '(A)'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        else:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

        self.problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["σxz"] = "(dx(w) +  u_z )"

        super(FC_equations_2d, self)._set_subs()

        
    def _set_diffusivities(self, Rayleigh, Prandtl, chemistry=False, ChemicalPrandtl=1, \
                           Qu_0=5e-8, phi_0=10, **kwargs):
        super(FC_equations_2d, self)._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, **kwargs)
        
        if chemistry:
            self.nu_chem_l = self._new_ncc()
            self.nu_chem_r = self._new_ncc()
            self.del_nu_chem_l = self._new_ncc()
            self.del_nu_chem_r = self._new_ncc()
            self.necessary_quantities['nu_chem_l'] = self.nu_chem_l
            self.necessary_quantities['nu_chem_r'] = self.nu_chem_r
            self.necessary_quantities['del_nu_chem_l'] = self.del_nu_chem_l
            self.necessary_quantities['del_nu_chem_r'] = self.del_nu_chem_r
            self.nu_chem_l.set_scales(self.domain.dealias, keep_data=True)
            self.nu_chem_r.set_scales(self.domain.dealias, keep_data=True)
            self.del_nu_chem_l.set_scales(self.domain.dealias, keep_data=True)
            self.del_nu_chem_r.set_scales(self.domain.dealias, keep_data=True)
            self.nu_l.set_scales(self.domain.dealias, keep_data=True)
            self.nu_r.set_scales(self.domain.dealias, keep_data=True)
            self.nu_chem_l['g'] = self.nu_l['g']/ChemicalPrandtl
            self.nu_chem_r['g'] = self.nu_r['g']/ChemicalPrandtl
            self.del_nu_chem_l['g'] = self.del_nu_l['g']/ChemicalPrandtl
            self.del_nu_chem_r['g'] = self.del_nu_r['g']/ChemicalPrandtl

            
            # Setting chemical rate coefficient
            self.k_chem = self._new_ncc()
            self.necessary_quantities['k_chem'] = self.k_chem
            self.k_chem.set_scales(self.domain.dealias, keep_data=False)

            # -- Recalculating to avoid parallelization issues --
            # -- Fixing Ra=1e4, Re=10 just as numbers to fix QP in atmosphere --
            nu_BOA = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s/self.Cp)*self.g)/1e4) \
                     / self.z0**self.poly_m 
            H_rho_BOA = (self.poly_m + 1) * self.z0  / self.poly_m / self.g
            tau_0 = Qu_0 * H_rho_BOA**2 / nu_BOA / 10 
            T_act = phi_0 * self.z0   # T0_BOA = z0

            kchem = 1 / tau_0 * np.exp(-T_act / (self.z0-self.z_dealias))
            self.t_chem_BOA = tau_0/self.z0**self.poly_m * np.exp(T_act / self.z0) 

            for i in range(self.k_chem['g'].shape[0]):
                self.k_chem['g'][i] = kchem

                
            # Setting up equilibrium profiles
            c0 = 1
            chem_taper = self.match_Phi(self.z_dealias, \
                                        width=0.04 * 18.5/(np.sqrt(np.pi)*6.5/2)*self.Lz,center=self.Lz/2)

            
            self.C_eq = self._new_ncc()
            self.necessary_quantities['C_eq'] = self.C_eq
            self.C_eq.set_scales(self.domain.dealias, keep_data=False)
            self.C_eq['g'] = c0 * chem_taper

            self.G_eq = self._new_ncc()
            self.necessary_quantities['G_eq'] = self.G_eq
            self.G_eq.set_scales(self.domain.dealias, keep_data=False)
            self.G_eq['g'] = c0 * (self.Lz - self.z_dealias) / self.Lz

            
    def _set_parameters(self,chemistry=False):
        super(FC_equations_2d, self)._set_parameters()

        if chemistry:    
            self.problem.parameters['nu_chem_l'] = self.nu_chem_l
            self.problem.parameters['nu_chem_r'] = self.nu_chem_r
            self.problem.parameters['del_nu_chem_l'] = self.del_nu_chem_l
            self.problem.parameters['del_nu_chem_r'] = self.del_nu_chem_r

            
            # Adding in equilibrium value to correct source term
            c0 = 1
            chem_taper = self.match_Phi(self.z_dealias, \
                                        width=0.04 * 18.5/(np.sqrt(np.pi)*6.5/2)*self.Lz,center=self.Lz/2)
            
            self.C_eq = self._new_ncc()
            self.necessary_quantities['C_eq'] = self.C_eq
            self.C_eq.set_scales(self.domain.dealias, keep_data=False)
            self.C_eq['g'] = c0 * chem_taper

            self.G_eq = self._new_ncc()
            self.necessary_quantities['G_eq'] = self.G_eq
            self.G_eq.set_scales(self.domain.dealias, keep_data=False)
            self.G_eq['g'] = c0 * (self.Lz - self.z_dealias) / self.Lz

            # Is this the right place to set these?
            self.problem.parameters['k_chem'] = self.k_chem
            self.problem.parameters['C_eq'] = self.C_eq
            self.problem.parameters['G_eq'] = self.G_eq
            
        
    def set_equations(self, Rayleigh, Prandtl, ChemicalPrandtl=1,
                      chemistry=False, Qu_0=5e-8, phi_0=10,
                      kx = 0, EVP_2 = False, 
                      split_diffusivities=False):

        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl,
                                chemistry=chemistry, ChemicalPrandtl=ChemicalPrandtl,
                                Qu_0=Qu_0, phi_0=phi_0,
                                split_diffusivities=split_diffusivities)
        self._set_parameters(chemistry=chemistry)
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu_l)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu_l')
            self.problem.parameters.pop('chi_l')

        # define nu and chi for output
        if split_diffusivities:
            self.problem.substitutions['nu']  = '(nu_l + nu_r)'
            self.problem.substitutions['del_nu']  = '(del_nu_l + del_nu_r)'
            if chemistry:
                self.problem.substitutions['nu_chem']  = '(nu_chem_l + nu_chem_r)'
                self.problem.substitutions['del_nu_chem']  = '(del_nu_chem_l + del_chem_nu_r)'
            self.problem.substitutions['chi'] = '(chi_l + chi_r)'
            self.problem.substitutions['del_chi'] = '(del_chi_l + del_chi_r)'
        else:
            self.problem.substitutions['nu']  = '(nu_l)'
            self.problem.substitutions['del_nu']  = '(del_nu_l)'
            if chemistry:
                self.problem.substitutions['nu_chem']  = '(nu_chem_l)'
                self.problem.substitutions['del_nu_chem']  = '(del_nu_chem_l)'
            self.problem.substitutions['chi'] = '(chi_l)'
            self.problem.substitutions['del_chi'] = '(del_chi_l)'
        self._set_subs()
        
        self.viscous_term_u_l = " nu_l*(Lap(u, u_z) + 1/3*Div(dx(u), dx(w_z)))"
        self.viscous_term_u_r = " nu_r*(Lap(u, u_z) + 1/3*Div(dx(u), dx(w_z)))"
        self.viscous_term_w_l = " nu_l*(Lap(w, w_z) + 1/3*Div(  u_z, dz(w_z)))"
        self.viscous_term_w_r = " nu_r*(Lap(w, w_z) + 1/3*Div(  u_z, dz(w_z)))"

        if chemistry:
            self.diffusion_term_f_l = " nu_chem_l*Lap(f,f_z) "
            self.diffusion_term_f_r = " nu_chem_r*Lap(f,f_z) "
            self.diffusion_term_C_l = " nu_chem_l*Lap(C,C_z) "
            self.diffusion_term_C_r = " nu_chem_r*Lap(C,C_z) "
            self.diffusion_term_G_l = " nu_chem_l*Lap(G,G_z) "
            self.diffusion_term_G_r = " nu_chem_r*Lap(G,G_z) "
            
        if not self.constant_mu:
            self.viscous_term_u_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σxz"
            self.viscous_term_u_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σxz"
            self.viscous_term_w_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σzz"
            self.viscous_term_w_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σzz"

            if chemistry:
                self.diffusion_term_f_l += " + nu_chem_l * f_z * del_ln_rho0 + f_z * del_nu_chem_l "
                self.diffusion_term_f_r += " + nu_chem_r * f_z * del_ln_rho0 + f_z * del_nu_chem_r "+\
                                           " + nu_chem_r * f_z * dz(ln_rho1) + nu_chem_r * dx(f) * dx(ln_rho1) "
                self.diffusion_term_C_l += " + nu_chem_l * C_z * del_ln_rho0 + C_z * del_nu_chem_l "
                self.diffusion_term_C_r += " + nu_chem_r * C_z * del_ln_rho0 + C_z * del_nu_chem_r "+\
                                           " + nu_chem_r * C_z * dz(ln_rho1) + nu_chem_r * dx(C) * dx(ln_rho1) "
                self.diffusion_term_G_l += " + nu_chem_l * G_z * del_ln_rho0 + G_z * del_nu_chem_l "
                self.diffusion_term_G_r += " + nu_chem_r * G_z * del_ln_rho0 + G_z * del_nu_chem_r "+\
                                           " + nu_chem_r * G_z * dz(ln_rho1) + nu_chem_r * dx(G) * dx(ln_rho1) "
                
        self.problem.substitutions['L_visc_w'] = self.viscous_term_w_l
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u_l
        if chemistry:
            self.problem.substitutions['L_diff_f'] = self.diffusion_term_f_l
            self.problem.substitutions['L_diff_C'] = self.diffusion_term_C_l
            self.problem.substitutions['L_diff_G'] = self.diffusion_term_G_l
        
        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*σxx + dz(ln_rho1)*σxz)"
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*σxz + dz(ln_rho1)*σzz)"
        if chemistry:
            self.NL_diff_term_f = " nu_chem_l * f_z * dz(ln_rho1) + nu_chem_l * dx(f) * dx(ln_rho1) "
            self.NL_diff_term_C = " nu_chem_l * C_z * dz(ln_rho1) + nu_chem_l * dx(C) * dx(ln_rho1) "
            self.NL_diff_term_G = " nu_chem_l * G_z * dz(ln_rho1) + nu_chem_l * dx(G) * dx(ln_rho1) "  
        if split_diffusivities:
            self.nonlinear_viscous_u += " + {}".format(self.viscous_term_u_r)
            self.nonlinear_viscous_w += " + {}".format(self.viscous_term_w_r)
            if chemistry:
                self.NL_diff_term_f += " + {}".format(self.diffusion_term_f_r)
                self.NL_diff_term_C += " + {}".format(self.diffusion_term_C_r)
                self.NL_diff_term_G += " + {}".format(self.diffusion_term_G_r)
            
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u
        if chemistry:
            self.problem.substitutions['NL_diff_f'] = self.NL_diff_term_f
            self.problem.substitutions['NL_diff_C'] = self.NL_diff_term_C
            self.problem.substitutions['NL_diff_G'] = self.NL_diff_term_G
            

        # double check implementation of variabile chi and background coupling term.
        self.linear_thermal_diff_l    = " Cv_inv*(chi_l*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.linear_thermal_diff_r    = " Cv_inv*(chi_r*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.source                   = " Cv_inv*(chi*(T0_zz))" 
        if not self.constant_kappa:
            self.linear_thermal_diff_l += '+ Cv_inv*(chi_l*del_ln_rho0 + del_chi_l)*T1_z'
            self.linear_thermal_diff_r += '+ Cv_inv*(chi_r*del_ln_rho0 + del_chi_r)*T1_z'
            self.source                += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'
        
        self.nonlinear_thermal_diff = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        if split_diffusivities:
            self.nonlinear_thermal_diff += " + {}".format(self.linear_thermal_diff_r)
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff_l
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source

        # double check this      
        self.viscous_heating = " Cv_inv*nu*(dx(u)*σxx + w_z*σzz + σxz**2)"

        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
        if chemistry:
            self.problem.add_equation("dz(f) - f_z = 0")
            self.problem.add_equation("dz(C) - C_z = 0")
            self.problem.add_equation("dz(G) - G_z = 0")
            
        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))


        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 
                
        if chemistry:
            logger.debug("Setting passive and reactive tracer equations")
            self.problem.add_equation(("(scale)*(dt(f) - L_diff_f)                 = (scale)*(-UdotGrad(f,f_z) + NL_diff_f)"))
            self.problem.add_equation(("(scale)*(dt(C) - L_diff_C + k_chem*rho0*C) = (scale)*(-UdotGrad(C,C_z) + NL_diff_C + k_chem*rho0*C_eq)"))
            self.problem.add_equation(("(scale)*(dt(G) - L_diff_G + k_chem*rho0*G) = (scale)*(-UdotGrad(G,G_z) + NL_diff_G + k_chem*rho0*G_eq)"))


    def initialize_output(self, solver, data_dir, chemistry=False,
                          full_output=False, coeffs_output=True,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = super(FC_equations_2d, self).initialize_output(solver, data_dir, full_output=full_output,
                                                                        max_writes=max_writes, mode=mode, chemistry=chemistry, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=max_writes, parallel=False,
                                                            mode=mode, **kwargs)
        analysis_slice.add_task("s_fluc", name="s")
        analysis_slice.add_task("s_fluc - plane_avg(s_fluc)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("ω_y", name="vorticity")
        if chemistry:
            analysis_slice.add_task("f", name="f")
            analysis_slice.add_task("C", name="C")
            analysis_slice.add_task("G", name="G")
        analysis_tasks['slice'] = analysis_slice

        if coeffs_output:
            analysis_coeff = solver.evaluator.add_file_handler(data_dir+"coeffs", max_writes=max_writes, parallel=False,
                                                               mode=mode, **kwargs)
            analysis_coeff.add_task("s_fluc", name="s", layout='c')
            analysis_coeff.add_task("s_fluc - plane_avg(s_fluc)", name="s'", layout='c')
            analysis_coeff.add_task("T1", name="T", layout='c')
            analysis_coeff.add_task("ln_rho1", name="ln_rho", layout='c')
            analysis_coeff.add_task("u", name="u", layout='c')
            analysis_coeff.add_task("w", name="w", layout='c')
            analysis_coeff.add_task("enstrophy", name="enstrophy", layout='c')
            analysis_coeff.add_task("ω_y", name="vorticity", layout='c')
            if chemistry:
                analysis_coeff.add_task("f", name="f", layout='c')
                analysis_coeff.add_task("C", name="C", layout='c')
                analysis_coeff.add_task("G", name="G", layout='c')
            analysis_tasks['coeff'] = analysis_coeff
        
        return self.analysis_tasks


class FC_equations_2d_kappa(FC_equations_2d):
                
    def set_equations(self, Rayleigh, Prandtl, split_diffusivities=None):
        
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()

        # define nu and chi for outputs
        self.problem.substitutions['nu']  = 'μ/rho0*exp(-ln_rho1)'
        self.problem.substitutions['chi'] = 'κ/rho0*exp(-ln_rho1)'        
        self._set_subs()

        self.problem.substitutions['L_visc_w'] = " μ/rho0*(Lap(w, w_z) + 1/3*Div(  u_z, dz(w_z)) + del_ln_μ*σzz)"                
        self.problem.substitutions['L_visc_u'] = " μ/rho0*(Lap(u, u_z) + 1/3*Div(dx(u), dx(w_z)) + del_ln_μ*σxz)"
        
        self.problem.substitutions['NL_visc_w'] = "L_visc_w*(exp(-ln_rho1)-1)"
        self.problem.substitutions['NL_visc_u'] = "L_visc_u*(exp(-ln_rho1)-1)"

        self.problem.substitutions['κT0'] = "(del_ln_κ*T0_z + T0_zz)"
        self.problem.substitutions['κT1'] = "(del_ln_κ*T1_z + Lap(T1, T1_z))"
        
        self.problem.substitutions['L_thermal']  =   " κ/rho0*Cv_inv*(κT0*-1*ln_rho1 + κT1)"
        self.problem.substitutions['NL_thermal'] =   " κ/rho0*Cv_inv*(κT0*(exp(-ln_rho1)+ln_rho1) + κT1*(exp(-ln_rho1)-1))"
        self.problem.substitutions['source_terms'] = " κ/rho_full*Cv_inv*(T0_zz + del_ln_κ*T0_z)"        
        self.problem.substitutions['NL_visc_heat'] = " μ/rho_full*Cv_inv*(dx(u)*σxx + w_z*σzz + σxz**2)"

        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z     + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-UdotGrad(w, w_z) - T1*dz(ln_rho1) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1)   + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-UdotGrad(u, u_z) - T1*dx(ln_rho1) + NL_visc_u)"))

        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z  + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z) - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 

    def _set_diffusivities(self, *args, **kwargs):
        super(FC_equations_2d_kappa, self)._set_diffusivities(*args, **kwargs)
        self.kappa = self._new_ncc()
        self.chi.set_scales(1, keep_data=True)
        self.rho0.set_scales(1, keep_data=True)
        self.kappa['g'] = self.chi['g']*self.rho0['g']
        self.problem.parameters['κ'] = self.kappa
        if self.constant_kappa:
            self.problem.substitutions['del_ln_κ'] = '0'
        else:
            self.del_ln_kappa = self._new_ncc()
            self.kappa.differentiate('z', out=self.del_ln_kappa)
            self.del_ln_kappa['g'] /= self.kappa['g']
            self.problem.parameters['del_ln_κ'] = self.del_ln_kappa
        self.mu = self._new_ncc()
        self.mu['g'] = self.nu['g']*self.rho0['g']
        self.problem.parameters['μ'] = self.mu
        if self.constant_mu:
            self.problem.substitutions['del_ln_μ'] = '0'
        else:
            self.del_ln_mu = self._new_ncc()
            self.mu.differentiate('z', out=self.del_ln_mu)
            self.del_ln_mu['g'] /= self.mu['g']
            self.problem.parameters['del_ln_μ'] = self.del_ln_mu
                    
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True
            
        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z) = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z) = 0")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise
            
class FC_equations_rxn(FC_equations_2d):
    def __init__(self, **kwargs):
        super(FC_equations_rxn, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes with chemical reactions'
        self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1','c','c_z','f','f_z']
                
    def _set_diffusivities(self, Rayleigh, Prandtl, ChemicalPrandtl, ChemicalReynolds, **kwargs):
        super(FC_equations_rxn, self)._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, **kwargs)

        self.nu_chem = self._new_ncc()
        self.necessary_quantities['nu_chem'] = self.nu_chem
        self.nu_chem.set_scales(self.domain.dealias, keep_data=True)
        self.nu.set_scales(self.domain.dealias, keep_data=True)
        self.nu_chem['g'] = self.nu['g']/ChemicalPrandtl

        # this should become a chemical Reynolds number control parameter, or something sensible
        self.k_chem = self._new_ncc()
        self.necessary_quantities['k_chem'] = self.k_chem
        self.k_chem.set_scales(self.domain.dealias, keep_data=False)
        self.k_chem['g'] = self.nu_chem['g']*self.Lz_cz*ChemicalReynolds
        
    def _set_parameters(self):
        super(FC_equations_rxn, self)._set_parameters()
    
        self.problem.parameters['nu_chem'] = self.nu_chem

        # need to set rate coefficient somewhere.  In init?  In diffusivities?
        self.problem.parameters['k_chem'] = self.k_chem

    def set_equations(self, Rayleigh, Prandtl, ChemicalPrandtl, ChemicalReynolds,
                      kx = 0, EVP_2 = False, 
                      easy_rho_momentum=False, easy_rho_energy=False,
                      **kwargs):

        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
            
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, ChemicalPrandtl=ChemicalPrandtl, ChemicalReynolds=ChemicalReynolds, **kwargs)
        self._set_parameters()
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu')
            self.problem.parameters.pop('chi')
 
        self._set_subs()
                
        # here, nu and chi are constants        
        self.viscous_term_w = " nu*(Lap(w, w_z) + 1/3*Div(u_z,   dz(w_z)))"
        self.viscous_term_u = " nu*(Lap(u, u_z) + 1/3*Div(dx(u), dx(w_z)))"
        if not easy_rho_momentum:
            self.viscous_term_w += " + (nu*del_ln_rho0 + del_nu) * (2*w_z - 2/3*Div_u)"
            self.viscous_term_u += " + (nu*del_ln_rho0 + del_nu) * (  u_z +     dx(w))"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u

        self.nonlinear_viscous_w = " nu*(    u_z*dx(ln_rho1) + 2*w_z*dz(ln_rho1) + dx(ln_rho1)*dx(w) - 2/3*dz(ln_rho1)*Div_u)"
        self.nonlinear_viscous_u = " nu*(2*dx(u)*dx(ln_rho1) + dx(w)*dz(ln_rho1) + dz(ln_rho1)*u_z   - 2/3*dx(ln_rho1)*Div_u)"
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u

        # double check implementation of variabile chi and background coupling term.
        self.problem.substitutions['Q_z'] = "(-T1_z)"
        self.linear_thermal_diff    = " Cv_inv*(chi*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.nonlinear_thermal_diff = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                 " Cv_inv*(chi*(T0_zz))"# - Qcool_z/rho_full)"
        if not easy_rho_energy:
            self.linear_thermal_diff += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T1_z'
            self.source              += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff 
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source
      
        self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2/3*Div_u**2)"
        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
        self.problem.add_equation("dz(f) - f_z = 0")
        self.problem.add_equation("dz(c) - c_z = 0")


        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))


        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 

        # no NCCs here to rescale.  Easy to modify.
        logger.debug("Setting passive tracer scalar equation")
        self.problem.add_equation("dt(f) - nu_chem*Lap(f, f_z) =  -UdotGrad(f,f_z) - f*Div_u")
        logger.debug("Setting passive reacting scalar equation")
        self.problem.add_equation("(scale)*(dt(c) - nu_chem*Lap(c, c_z) + k_chem*rho0*c) =  (scale)*(-UdotGrad(c,c_z) - c*Div_u)")

        logger.info("using nonlinear EOS for entropy, via substitution")

    def set_BC(self, **kwargs):
        super(FC_equations_rxn, self).set_BC(**kwargs)

        # perfectly conducting boundary conditions.
        self.problem.add_bc("left(f_z) = 0")
        self.problem.add_bc("left(c_z) = 0")
        self.problem.add_bc("right(f_z) = 0")
        self.problem.add_bc("right(c_z) = 0")

        self.dirichlet_set.append('c_z')
        self.dirichlet_set.append('f_z')

        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
        
    def set_IC(self, solver, A0=1e-6, **kwargs):
        super(FC_equations_rxn, self).set_IC(solver, A0=A0, **kwargs)
    
        self.f_IC = solver.state['f']
        self.c_IC = solver.state['c']

        # this is just a bump in z
        self.c_IC['g'] = A0*np.cos(np.pi*self.z_dealias/self.Lz)
        self.f_IC['g'] = self.c_IC['g']
        
    def initialize_output(self, solver, data_dir, **kwargs):
        super(FC_equations_rxn, self).initialize_output(solver, data_dir, **kwargs)

        # make analysis_tasks a dictionary!
        analysis_slice = self.analysis_tasks['slice']
        analysis_slice.add_task("f", name="f")
        analysis_slice.add_task("c", name="c")

        analysis_profile = self.analysis_tasks['profile']
        analysis_profile.add_task("plane_avg(f)", name="f")
        analysis_profile.add_task("plane_avg(c)", name="c")

        analysis_scalar = self.analysis_tasks['scalar']
        analysis_scalar.add_task("vol_avg(f)", name="f")
        analysis_scalar.add_task("vol_avg(c)", name="c")

        return self.analysis_tasks

class FC_equations_3d(FC_equations):
    def __init__(self, **kwargs):
        super(FC_equations_3d, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes in 3-D'
        self.variables = ['u','u_z','v','v_z','w','w_z','T1', 'T1_z', 'ln_rho1']
    
    def _set_subs(self, **kwargs):
        # 3-D specific subs
        self.problem.substitutions['ω_x'] = '(dy(w) - v_z)'        
        self.problem.substitutions['ω_y'] = '( u_z  - dx(w))'        
        self.problem.substitutions['ω_z'] = '(dx(v) - dy(u))'        
        self.problem.substitutions['enstrophy']   = '(ω_x**2 + ω_y**2 + ω_z**2)'

        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dy(dy(f)) + dz(f_z))"
        self.problem.substitutions['Div(fx, fy, fz_z)'] = "(dx(fx) + dy(fy) + fz_z)"
        self.problem.substitutions['Div_u'] = "Div(u, v, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + v*dy(f) + w*(f_z))"
                    
        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

        self.problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["σyy"] = "(2*dy(v) - 2/3*Div_u)"
        self.problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["σxy"] = "(dx(v) + dy(u))"
        self.problem.substitutions["σxz"] = "(dx(w) +  u_z )"
        self.problem.substitutions["σyz"] = "(dy(w) +  v_z )"
           
        super(FC_equations_3d, self)._set_subs(**kwargs)

                        
    def set_equations(self, Rayleigh, Prandtl, Taylor=None, theta=0,
                      kx = 0, EVP_2 = False, 
                      split_diffusivities=False):
        
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl, split_diffusivities=split_diffusivities)
        self._set_parameters()
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu_l)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu_l')
            self.problem.parameters.pop('chi_l')
                # define nu and chi for output
        if split_diffusivities:
            self.problem.substitutions['nu']  = '(nu_l + nu_r)'
            self.problem.substitutions['del_nu']  = '(del_nu_l + del_nu_r)'
            self.problem.substitutions['chi'] = '(chi_l + chi_r)'
            self.problem.substitutions['del_chi'] = '(del_chi_l + del_chi_r)'
        else:
            self.problem.substitutions['nu']  = '(nu_l)'
            self.problem.substitutions['del_nu']  = '(del_nu_l)'
            self.problem.substitutions['chi'] = '(chi_l)'
            self.problem.substitutions['del_chi'] = '(del_chi_l)'
        self._set_subs()
    
        if Taylor:
            self.rotating = True
            self.problem.parameters['θ'] = theta
            self.problem.parameters['Ω'] = omega = np.sqrt(Taylor*self.nu_top**2/(4*self.Lz**4))
            logger.info("Rotating f-plane with Ω = {} and θ = {} (Ta = {})".format(omega, theta, Taylor))
            self.problem.substitutions['Ωx'] = '0'
            self.problem.substitutions['Ωy'] = 'Ω*sin(θ)'
            self.problem.substitutions['Ωz'] = 'Ω*cos(θ)'
            self.problem.substitutions['Coriolis_x'] = '(2*Ωy*w - 2*Ωz*v)'
            self.problem.substitutions['Coriolis_y'] = '(2*Ωz*u - 2*Ωx*w)'
            self.problem.substitutions['Coriolis_z'] = '(2*Ωx*v - 2*Ωy*u)'
            self.problem.substitutions['Rossby'] = '(sqrt(enstrophy)/(2*Ω))'
        else:
            self.rotating = False
            self.problem.substitutions['Coriolis_x'] = '0'
            self.problem.substitutions['Coriolis_y'] = '0'
            self.problem.substitutions['Coriolis_z'] = '0'

        self.viscous_term_u_l = " nu_l*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z)))"
        self.viscous_term_v_l = " nu_l*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z)))"
        self.viscous_term_w_l = " nu_l*(Lap(w, w_z) + 1/3*Div(  u_z, v_z, dz(w_z)))"
        self.viscous_term_u_r = " nu_r*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z)))"
        self.viscous_term_v_r = " nu_r*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z)))"
        self.viscous_term_w_r = " nu_r*(Lap(w, w_z) + 1/3*Div(  u_z, v_z, dz(w_z)))"
        # here, nu and chi are constants        
        
        if not self.constant_mu:
            self.viscous_term_u_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σxz"
            self.viscous_term_w_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σzz"
            self.viscous_term_v_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σyz"
            self.viscous_term_u_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σxz"
            self.viscous_term_w_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σzz"
            self.viscous_term_v_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σyz"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w_l
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u_l
        self.problem.substitutions['L_visc_v'] = self.viscous_term_v_l

        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*σxx + dy(ln_rho1)*σxy + dz(ln_rho1)*σxz)"
        self.nonlinear_viscous_v = " nu*(dx(ln_rho1)*σxy + dy(ln_rho1)*σyy + dz(ln_rho1)*σyz)"
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*σxz + dy(ln_rho1)*σyz + dz(ln_rho1)*σzz)"
        if split_diffusivities:
            self.nonlinear_viscous_u += " + {}".format(self.viscous_term_u_r)
            self.nonlinear_viscous_v += " + {}".format(self.viscous_term_v_r)
            self.nonlinear_viscous_w += " + {}".format(self.viscous_term_w_r)
 
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u
        self.problem.substitutions['NL_visc_v'] = self.nonlinear_viscous_v
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w

        # double check implementation of variabile chi and background coupling term.
        self.linear_thermal_diff_l    = " Cv_inv*(chi_l*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.linear_thermal_diff_r    = " Cv_inv*(chi_r*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.nonlinear_thermal_diff   = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + dy(T1)*dy(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                   " Cv_inv*(chi*(T0_zz))" # - Qcool_z/rho_full)"
        if not self.constant_kappa:
            self.linear_thermal_diff_l += '+ Cv_inv*(chi_l*del_ln_rho0 + del_chi_l)*T1_z'
            self.linear_thermal_diff_r += '+ Cv_inv*(chi_r*del_ln_rho0 + del_chi_r)*T1_z'
            self.source                += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'

        if split_diffusivities:
            self.nonlinear_thermal_diff += " + {}".format(self.linear_thermal_diff_r)
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff_l
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source

        self.viscous_heating = " Cv_inv*nu*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2)"

        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(v) - v_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + Coriolis_z + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + Coriolis_x + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))

        logger.debug("Setting y-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(v) + Coriolis_y + dy(T1) + T0*dy(ln_rho1)                  - L_visc_v) = "
                                   "(scale_momentum)*(-T1*dy(ln_rho1) - UdotGrad(v, v_z) + NL_visc_v)"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)"))
        

    def set_BC(self, **kwargs):        
        super(FC_equations_3d, self).set_BC(**kwargs)
        # stress free boundary conditions.
        self.problem.add_bc("left(v_z) = 0")
        self.problem.add_bc("right(v_z) = 0")
        self.dirichlet_set.append('v_z')
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

        
    def initialize_output(self, solver, data_dir, full_output=False, coeffs_output=False,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = super(FC_equations_3d, self).initialize_output(solver, data_dir, full_output=full_output, coeffs_output=coeffs_output,
                                                                        max_writes=max_writes, mode=mode, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=max_writes, parallel=False,
                                                           mode=mode, **kwargs)
        analysis_slice.add_task("interp(s_fluc,                     y={})".format(self.Ly/2), name="s")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), y={})".format(self.Ly/2), name="s'")
        analysis_slice.add_task("interp(enstrophy,                  y={})".format(self.Ly/2), name="enstrophy")
        analysis_slice.add_task("interp(ω_y,                        y={})".format(self.Ly/2), name="vorticity")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.95*self.Lz), name="s near top")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.95*self.Lz), name="s' near top")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.95*self.Lz), name="enstrophy near top")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.95*self.Lz), name="vorticity_z near top")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.5*self.Lz),  name="s midplane")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.5*self.Lz),  name="s' midplane")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.5*self.Lz),  name="enstrophy midplane")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.5*self.Lz),  name="vorticity_z midplane")
        analysis_tasks['slice'] = analysis_slice

        analysis_volume = solver.evaluator.add_file_handler(data_dir+"volumes", max_writes=max_writes, parallel=False, 
                                                            mode=mode, **kwargs)
        analysis_volume.add_task("enstrophy", name="enstrophy")
        analysis_volume.add_task("s_fluc+s_mean", name="s_tot")
        analysis_tasks['volume'] = analysis_volume

        if self.rotating:
            analysis_scalar = self.analysis_tasks['scalar']
            analysis_scalar.add_task("vol_avg(Rossby)", name="Rossby")

            analysis_profile = self.analysis_tasks['profile']
            analysis_profile.add_task("plane_avg(Rossby)", name="Rossby")
        
        return self.analysis_tasks
        

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

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm*Cp_inv/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)*Cp_inv/(nu*chi)'

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

    def set_equations(self, Rayleigh, Prandtl):
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
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.s_IC.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=1, scales=self.domain.dealias)
        self.s_IC['g'] = A0*np.sin(np.pi*z_dealias/self.Lz)*noise['g']

        logger.info("Starting with s perturbations of amplitude A0 = {:g}".format(A0))

    def initialize_output(self, solver, data_dir, max_writes=20, **kwargs):
        analysis_tasks = OrderedDict()
        self.analysis_tasks = analysis_tasks
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=max_writes, parallel=False, **kwargs)
        analysis_slice.add_task("s", name="s")
        analysis_slice.add_task("s - plane_avg(s)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("vorticity", name="vorticity")
        analysis_tasks['slice'] = analysis_slice
        
        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=max_writes, parallel=False, **kwargs)
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
        
        analysis_tasks['profile'] = analysis_profile

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False, **kwargs)
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


