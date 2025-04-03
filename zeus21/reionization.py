"""

Models reionization using an analogy of a halo mass function to ionized bubbles 
See Sklansky et al. (in prep)

Authors: Yonatan Sklansky, Emilie Thelie
UT Austin - January 2025

"""

from . import z21_utilities
from . import cosmology
from . import constants
import numpy as np
import astropy.units as u
from scipy.integrate import cumulative_trapezoid
from tqdm import trange


class BMF:
    """
    Computes the bubble mass function (BMF). 

    
    """
    
    def __init__(self, CoeffStructure, HMFintclass, CosmoParams, AstroParams, R_linear_sigma_fit_input=10, FLAG_converge=True, max_iter=10, ZMAX_REION = 30):

        self.ZMAX_REION = ZMAX_REION #max redshift up to which we calculate reionization observables
        self.zlist = CoeffStructure.zintegral
        self.Rs = CoeffStructure.Rtabsmoo

        
        self.gamma = CoeffStructure.gamma_niondot_II_index2D
        self.gamma2 = CoeffStructure.gamma2_niondot_II_index2D
        self.sigma = CoeffStructure.sigmaofRtab

        self.Hz = cosmology.Hubinvyr(CosmoParams, self.zlist)
        self.trec0 = 1/(constants.alphaB * cosmology.n_H(CosmoParams,0) * AstroParams._clumping) #seconds
        self.trec = self.trec0/(1+self.zlist)**3/constants.yrTos #years
        self.niondot_avg = CoeffStructure.niondot_avg_II

        #self.xHI = CoeffStructure.xHI_avg ----> need to figure out what's going on with this
        self.ion_frac = np.fmin(1, [self.calc_Q(CosmoParams, i) for i in np.arange(len(self.zlist))])
        self.ion_frac_initial = np.copy(self.ion_frac)

        zr_mesh = np.meshgrid(np.arange(len(self.Rs)), np.arange(len(self.zlist)))
        self.nion_norm = self.nion_normalization(zr_mesh[1], zr_mesh[0])
        self.barrier = self.compute_barrier(CosmoParams, self.ion_frac)
        self.barrier_initial = np.copy(self.barrier)

        self.R_linear_sigma_fit_idx = z21_utilities.find_nearest_idx(self.Rs, R_linear_sigma_fit_input)
        self.R_linear_sigma_fit = self.Rs[self.R_linear_sigma_fit_idx]

        self.BMF = np.array([self.VRdn_dR(HMFintclass, np.arange(len(self.Rs)), i) for i in range(len(self.zlist))]) #initial bubble mass function
        self.BMF_initial = np.copy(self.BMF)
        self.ion_frac = np.nan_to_num([np.trapz(self.BMF[i], np.log(self.Rs)) for i in range(len(self.zlist))]) #ion_frac by integrating the BMF
        self.ion_frac[self.barrier[:, -1]<=0] = 1
        
        if FLAG_converge:
            self.converge_BMF(CosmoParams, HMFintclass, self.ion_frac, max_iter=max_iter)
        #two functions: compute BMF and iterate
        


    def compute_barrier(self, CosmoParams, ion_frac):
        """
        Computes the density barrier threshold for ionization.
        
        Using the analytic model from Sklansky et al. (in prep), if the total number of ionized photons produced in an overdensity exceeds the sum of the number of hydrogens present and total number of recombinations occurred, then the overdensity is ionized. The density required to ionized is recorded.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        ion_frac: 1D np.array
            The ionized fractions to be used to compute the number of recombinations. 

        Output
        ----------
        barrier: 2D np.array
            The resultant density threshold array. First dimension is each redshift, second dimension is each radius scale.
        """
        barrier = np.zeros((len(self.zlist), len(self.Rs)))
        ds_array = np.linspace(-1, 5, 101)
        
        for ir in range(len(self.Rs)):
            #Compute nion_values and nrec_values for this 'ir'
            nion_values = self.nion_delta_r_int(CosmoParams, ds_array, ir)  #Shape (nd, nz)
            nrec_values = self.nrec(CosmoParams, ds_array, ion_frac)             #Shape (nd, nz)
        
            total_values = np.log10(nion_values / (1 + nrec_values) + 1e-10)   #taking difference in logspace to find zero-crossing 
        
            #Loop over redshift indices
            for i in range(len(self.zlist)):
                y_values = total_values[:, i]  #Shape (nd,)
        
                #Find zero crossings
                sign_change = np.diff(np.sign(y_values))
                idx = np.where(sign_change)[0]
                if idx.size > 0:
                    #Linear interpolation to find zero crossings
                    x0 = ds_array[idx]
                    x1 = ds_array[idx + 1]
                    y0 = y_values[idx]
                    y1 = y_values[idx + 1]
                    x_intersect = x0 - y0 * (x1 - x0) / (y1 - y0)
                    barrier[i, ir] = x_intersect[0]  #Assuming we take the first crossing
                else:
                    barrier[i, ir] = np.nan #Never crosses
        barrier = barrier * (CosmoParams.growthint(self.zlist)/CosmoParams.growthint(self.zlist[0]))[:, np.newaxis] #scale barrier with growth factor
        barrier[self.zlist > self.ZMAX_REION] = 100 #sets density to an unreachable barrier, as if reionization isn't happening
        return barrier

    #normalizing the nion/sfrd model
    def nion_normalization(self, z, R):
        return 1/np.sqrt(1-2*self.gamma2[z, R]*self.sigma[z, R]**2)*np.exp(self.gamma[z, R]**2 * self.sigma[z, R]**2 / (2-4*self.gamma2[z, R]*self.sigma[z, R]**2))

    def nrec(self, CosmoParams, d_array, ion_frac):
        """
        Vectorized computation of nrec over an array of overdensities d_array.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate nrec over.
        ion_frac: 1D np.array
            The ionized fraction over all redshifts.

        Output
        ----------
        nrecs: 2D np.array
            The total number of recombinations at each overdensity for a certain ionized fraction history at each redshift. The first dimension is densities, the second dimension is redshifts.
        """
        
        #reverse the inputs to make the integral easier to compute
        z_rev = self.zlist[::-1]
        Hz_rev = self.Hz[::-1]
        trec_rev = self.trec[::-1]
        ion_frac_rev = ion_frac[::-1]
    
        denom = -1 / (1 + z_rev) / Hz_rev / trec_rev
        integrand_base = denom * ion_frac_rev 
        Dg = CosmoParams.growthint(z_rev) #growth factor

        nrecs = cumulative_trapezoid(integrand_base*(1+d_array[:, np.newaxis]*Dg/Dg[-1]), x=z_rev, initial=0) #(1+delta) rather than (1+delta)^2 because nrec and nion are per hydrogen atom 
        
        #TODO: nonlinear recombinations/higher order

        nrecs = nrecs[:, ::-1]  # Reverse to match self.zlist order
        return nrecs
    
    def niondot_delta_r(self, CosmoParams, d_array, ir):
        """
        Compute niondot over an array of overdensities d_array for a given ir.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate niondot over.
        ir: int
            Index corresponding to a certain radius value from self.Rs.

        Output
        ----------
        niondot: 2D np.array
            The rates of ionizing photon production. The first dimension is densities, the second dimension is redshifts.
        """
        
        d_array = d_array[:, np.newaxis] * CosmoParams.growthint(self.zlist)[np.newaxis, :] / CosmoParams.growthint(self.zlist[0])
    
        gamma_ir = self.gamma[:, ir]   
        gamma2_ir = self.gamma2[:, ir]  
        nion_norm_ir = self.nion_norm[:, ir] 
    
        exp_term = np.exp(gamma_ir[np.newaxis, :] * d_array + gamma2_ir[np.newaxis, :] * d_array**2)
        niondot = (self.niondot_avg[np.newaxis, :] / nion_norm_ir[np.newaxis, :]) * exp_term
        
        return niondot
    
    def nion_delta_r_int(self, CosmoParams, d_array, ir):
        """
        Vectorized computation of nion over an array of overdensities d_array for a given ir.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate niondot over.
        ir: int
            Index corresponding to a certain radius value from self.Rs.

        Output
        ----------
        nion: 2D np.array
            The total number of ionizing photons produced since z=50. The first dimension is densities, the second dimension is redshifts.
        """
        
        #reverse the inputs to make the integral easier to compute
        z_rev = self.zlist[::-1]
        Hz_rev = self.Hz[::-1]
    
        niondot_values = self.niondot_delta_r(CosmoParams, d_array, ir)
    
        integrand = -1 / (1 + z_rev) / Hz_rev * niondot_values[:, ::-1]
        #cumulative_integral = np.concatenate(
        #    (np.zeros((len(d_array), 1)), cumulative_trapezoid(integrand, x=z_rev, axis=1)), axis=1
        #)
        nion = cumulative_trapezoid(integrand, x=z_rev, initial=0)[:, ::-1] #reverse back to match self.zlist order
        #nion = cumulative_integral[:, ::-1]  # Reverse back to match self.zlist order
        return nion

    #calculating ionized fraction, put outside the class
    def calc_Q(self, CosmoParams, iz):
        z = self.zlist[iz]
        dtdz = 1/cosmology.Hubinvyr(CosmoParams, self.zlist[iz:])/(1 + self.zlist[iz:])
        tau0 = self.trec0 * np.sqrt(CosmoParams.OmegaM) * cosmology.Hubinvyr(CosmoParams,0) / constants.yrTos
        exp = np.exp(2/3/tau0 * (np.power(1 + z, 3/2) - np.power(1 + self.zlist[iz:], 3/2))) #switched order around to be correct
        
        integrand = dtdz * self.niondot_avg[iz:] * exp
    
        return np.trapz(integrand, x = self.zlist[iz:])

    #computing linear barrier
    def B_1(self, HMFintclass, ir, iz):
        sigmax = HMFintclass.sigmaR_int(self.Rs[self.R_linear_sigma_fit_idx+1], self.zlist[iz])
        sigmin = HMFintclass.sigmaR_int(self.Rs[self.R_linear_sigma_fit_idx-1], self.zlist[iz])
        barriermax = self.barrier[iz, self.R_linear_sigma_fit_idx+1]
        barriermin = self.barrier[iz, self.R_linear_sigma_fit_idx-1]
        return (barriermax - barriermin)/(sigmax**2 - sigmin**2)
        
    def B_0(self, HMFintclass, ir, iz):
        sigmin = HMFintclass.sigmaR_int(self.Rs[self.R_linear_sigma_fit_idx-1], self.zlist[iz])
        barriermin = self.barrier[iz, self.R_linear_sigma_fit_idx-1]
        return barriermin - sigmin**2 * self.B_1(HMFintclass, ir, iz)
    
    def B(self, HMFintclass, ir, iz): 
        sig = HMFintclass.sigmaR_int(self.Rs[ir], self.zlist[iz])
        return self.B_0(HMFintclass, ir, iz) + self.B_1(HMFintclass, ir, iz)*sig**2

    
    #computing other terms in the BMF
    def dsigma_dR(self, HMFintclass, ir, iz):
        R = self.Rs[ir]
        z = self.zlist[iz]
        sigma = HMFintclass.sigmaR_int(R, z)
        return sigma/R*np.gradient(np.log(sigma), np.log(R))
    
    def dlogsigma_dlogR(self, HMFintclass, ir, iz):
        R = self.Rs[ir]
        z = self.zlist[iz]
        return self.dsigma_dR(HMFintclass, ir, iz) * R/HMFintclass.sigmaR_int(R,z)
    
    def VRdn_dR(self, HMFintclass, ir, iz):
        R = self.Rs[ir]
        z = self.zlist[iz]
        sig = HMFintclass.sigmaR_int(self.Rs[ir], self.zlist[iz])
        return np.sqrt(2/np.pi) * np.abs(self.dlogsigma_dlogR(HMFintclass, ir, iz)) * np.abs(self.B_0(HMFintclass, ir, iz))/sig * np.exp(-self.B(HMFintclass, ir, iz)**2/2/sig**2)
    
    def Rdn_dR(self, ir, iz):
        R = self.Rs[ir]
        z = self.zlist[iz]
        return self.VRdn_dR(HMFintclass, ir, iz)*3/(4*np.pi*R**3)

    def converge_BMF(self, CosmoParams, HMFintclass, ion_frac_input, max_iter):
        self.ion_frac = ion_frac_input
        for j in trange(max_iter):
            ion_frac_prev = np.copy(self.ion_frac)
            
            self.barrier = self.compute_barrier(CosmoParams, self.ion_frac)
            self.BMF = np.array([self.VRdn_dR(HMFintclass, np.arange(len(self.Rs)), i) for i in range(len(self.zlist))])
            self.ion_frac = np.nan_to_num([np.trapz(self.BMF[i], np.log(self.Rs)) for i in range(len(self.zlist))])
            self.ion_frac[self.barrier[:, -1]<=0] = 1

            if np.allclose(ion_frac_prev, self.ion_frac):
                print(f'SUCCESS: BMF converged in {j} iterations.')
                return 
            
        print(f"WARNING: BMF didn't converge within {max_iter} iterations.")