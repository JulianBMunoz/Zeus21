"""

Make maps! For fun and science

Authors: Julian B. Muñoz, Yonatan Sklansky, Emilie Thelie
UT Austin - February 2025

"""

from . import cosmology
from . import constants
from . import z21_utilities

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from pyfftw import empty_aligned as empty
from tqdm import trange
import time


class CoevalMaps:
    "Class that calculates and keeps coeval maps, one z at a time."

    def __init__(self, T21_coefficients, Power_Spectrum, z, Lbox=600, Nbox=200, KIND=None, seed=1605):
        'the KIND flag determines the kind of map you make. Options are:'
        'KIND = 0, only T21 lognormal. OK approximation'
        'KIND = 1, density and T21 correlated. T21 has a gaussian and a lognormal component. Decent approximation'
        'KIND = 2, all maps'
        'KIND = 3, same as 2 but integrating over all R. Slow but most accurate'

        zlist = T21_coefficients.zintegral 
        _iz = min(range(len(zlist)), key=lambda i: np.abs(zlist[i]-z)) #pick closest z
        self.T21global = T21_coefficients.T21avg[_iz]
        self.Nbox = Nbox
        self.Lbox = Lbox
        self.seed = seed
        self.z = zlist[_iz] #will be slightly different from z input

        klist = Power_Spectrum.klist_PS
        k3over2pi2 = klist**3/(2*np.pi**2)


        if (KIND == 0): #just T21, ~gaussian
                
            P21 = Power_Spectrum.Deltasq_T21_lin[_iz]/k3over2pi2
            P21norminterp = interp1d(klist,P21/self.T21global**2,fill_value=0.0,bounds_error=False)


            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: P21norminterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed                
            )

            self.T21map = self.T21global * (1 + pb.delta_x() )
            self.deltamap = None


            
        elif (KIND == 1):
            Pd = Power_Spectrum.Deltasq_d_lin[_iz,:]/k3over2pi2
            Pdinterp = interp1d(klist,Pd,fill_value=0.0,bounds_error=False)

            pb = pbox.PowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: Pdinterp(k), 
                boxlength = self.Lbox,           
                seed = self.seed               
            )

            self.deltamap = pb.delta_x() #density map, basis of this KIND of approach

            #then we make a map of the linear T21 fluctuation, better to use the cross to keep sign, at linear level same 
            PdT21 = Power_Spectrum.Deltasq_dT21[_iz]/k3over2pi2

            powerratioint = interp1d(klist,PdT21/Pd,fill_value=0.0,bounds_error=False)


            deltak = pb.delta_k()

            powerratio = powerratioint(pb.k())
            T21lin_k = powerratio * deltak
            self.T21maplin= self.T21global + z21_utilities.powerboxCtoR(pb,mapkin = T21lin_k)

            #now make a nonlinear correction, built as \sum_R [e^(gR dR) - gR dR]. Uncorrelatd with all dR so just a separate field!
            #NOTE: its not guaranteed to work, excess power can be negative in some cases! Not for each component xa, Tk, but yes for T21
            excesspower21 = (Power_Spectrum.Deltasq_T21[_iz,:]-Power_Spectrum.Deltasq_T21_lin[_iz,:])/k3over2pi2

            lognormpower = interp1d(klist,excesspower21/self.T21global**2,fill_value=0.0,bounds_error=False)
            #G or logG? TODO revisit
            pbe = pbox.LogNormalPowerBox(
                N=self.Nbox,                     
                dim=3,                     
                pk = lambda k: lognormpower(k), 
                boxlength = self.Lbox,           
                seed = self.seed+1                # uncorrelated
            )

            self.T21mapNL = self.T21global*pbe.delta_x()

            #and finally, just add them together!
            self.T21map = self.T21maplin +  self.T21mapNL



        else:
            print('ERROR, KIND not implemented yet!')


class reionization_maps:
    """
    Generates 3D maps of the reionization fields.
    
    Uses a density threshold barrier determined from a converged bubble mass function. With default parameters, the code takes about 20 minutes on laptop to run.

    Parameters
    ----------
    CosmoParams: zeus21.Cosmo_Parameters class
        Stores cosmology.
    CoeffStructure: zeus21.get_T21_coefficients class
        Stores sfrd and 21cm coefficients.
    ClassyCosmo: zeus21.runclass class
        Sets up Class cosmology.
    CorrFClass: zeus21.Correlations class
        Calculates correlation functions.
    BMF: zeus21.reionization class
        Computes bubble mass functions and barriers.
    input_z: 1D np.array
        The redshifts at which to compute output maps. Narrowed down later to select available redshifts from CoeffStructure.zintegral.
    input_boxlength: float
        Comoving physical side length of the box. Default is 300 cMpc.
    ncells: int
        Number of cells on a side. Default is 300 cells.
    seed: int
        Sets the predetermined generation of maps. Default is 1234.
    r_precision: float
        Allows to change the steps of the radii for faster computation. Default (and max) is 1, lower values make the computation faster at the cost of accuracy.
    barrier: function
        Input density barrier to be used as the threshold for map generation. Takes z index //\\Change//\\ (relative to CoeffStructure.zintegral) as input and returns np.array of shape. Default is None.
    PRINT_TIMER: bool
        Whether to print the time elapsed along the process. Default is True.
    LOGNORMAL_DENSITY: bool
        Whether to use lognormal (True) or Gaussian (False) density fields. Default is False.of radii. Default is False.
    COMPUTE_DENSITY_AT_ALLZ: bool
        Whether to output the density field at all redshifts. If False, only the density at the lower input redshift is computed. If True, the computation time and memory usage dramatically increases. Default is False.
    SPHERIZE: bool
        Whether to flag spheres around ionized cells (True) instead of only central pixel flagging (False). Default is False. Central pixel flagging is generally more consistent with the bubble mass function than spherizing. Default is False.
    COMPUTE_MASSWEIGHTED_IONFRAC: bool
        Whether to compute the mass weighted ionized fraction. If True, COMPUTE_DENSITY_AT_ALLZ will be forced to True, thus increasing dramatically computation time. Default is False.
    lowres_massweighting: int
        Compute the mass-weighted ionized fraction more efficiently by using lower resolution density and ionized fields. Has to be >=1 and an integer. Default is 1.

    Attributes
    ----------
    dx: float
        Cell resolution of a side of the boxes.
    z: 1D np.array
        Redshifts at which the output maps are computed. Selected to be the closest to the input redshifts from the available ones in zeus21.
    r: 1D np.array
        Radii at which the density field is smoothed. Selected using r_precision from the available ones in zeus21.
    z_of_density: float
        Redshift at which the density is computed.
    density: 3D np.array
        Overdensity field at the lowest redshift asked by the user.
    density_allz: 4D np.array
        Overdensity field at all the redshifts asked by the user. First dimension correponds to redshifts. Only computed if COMPUTE_DENSITY_AT_ALLZ is True.
    ion_field_allz: 4D np.array
        Ionized fraction field at all the redshifts asked by the user. First dimension correponds to redshifts.
    ion_frac: 1D np.array
        Volume weighted ionized fraction at all the redshifts asked by the user.
    ion_frac_massweighted: 1D np.array
        Mass weighted ionized fraction at all the redshifts asked by the user. Only computed if COMPUTE_MASSWEIGHTED_IONFRAC is True.
    """
    
    def __init__(self, CosmoParams, ClassyCosmo, CorrFClass, CoeffStructure, BMF, input_z, 
                 input_boxlength=300., ncells=300, seed=1234, r_precision=1., barrier=None, 
                 PRINT_TIMER=True, ENFORCE_BMF_SCALE=True, 
                 LOGNORMAL_DENSITY=False, COMPUTE_DENSITY_AT_ALLZ=False, SPHERIZE=False, 
                 COMPUTE_MASSWEIGHTED_IONFRAC=False, lowres_massweighting=1):
        #Measure time elapsed from start
        self._start_time = time.time()
        
        ### boxes parameters
        self.input_z = input_z
        self.ENFORCE_BMF_SCALE = ENFORCE_BMF_SCALE
        self.ncells = ncells
        self.boxlength = input_boxlength
        self.dx = self.boxlength/self.ncells
        if self.ENFORCE_BMF_SCALE:
            self.dx = CoeffStructure.Rtabsmoo[0] * (4*np.pi/3)**(1/3) #change to constant
            self.boxlength = self.dx * self.ncells
            print(f"WARNING: with ENFORCE_BMF_SCALE=True, boxlength is changed to match BMF scale. Current value: {self.boxlength:.2f} cMpc.")
        self.seed = seed

        ### FLAGS
        self.PRINT_TIMER = PRINT_TIMER
        self.LOGNORMAL_DENSITY = LOGNORMAL_DENSITY
        self.COMPUTE_DENSITY_AT_ALLZ = COMPUTE_DENSITY_AT_ALLZ
        self.SPHERIZE = SPHERIZE
        self.COMPUTE_MASSWEIGHTED_IONFRAC = COMPUTE_MASSWEIGHTED_IONFRAC
        if self.COMPUTE_MASSWEIGHTED_IONFRAC:
            self.COMPUTE_DENSITY_AT_ALLZ = True

        ### selecting redshifts and radii from available redshifts
        # redshifts
        self._z_idx = z21_utilities.find_nearest_idx(CoeffStructure.zintegral, self.input_z)
        self.z = CoeffStructure.zintegral[self._z_idx]
        # radii
        self.r_precision = r_precision
        self._r_idx = np.linspace(0, len(CoeffStructure.Rtabsmoo)-1, int(len(CoeffStructure.Rtabsmoo)*self.r_precision), dtype=int)
        self.r = CoeffStructure.Rtabsmoo[self._r_idx]

        ### generating the density field at the closest redshift to the lower one inputed
        self.z_of_density = self.z[0]
        self.density = self.generate_density(ClassyCosmo, CorrFClass)
        self.density_allz = np.empty((len(self.z), self.ncells, self.ncells, self.ncells))
        if self.COMPUTE_DENSITY_AT_ALLZ:
            self.generate_density_allz(CosmoParams)

        ### smoothing the density field
        self._k = self.compute_k()
        self.density_smoothed_allr = self.smooth_density()

        ### generating the ionized field, and computing the ionized fraction
        self.barrier = barrier
        if self.barrier is None:
            self.barrier = BMF.barrier
        self.ion_field_allz, self.ion_frac = self.generate_xHII(CosmoParams, CoeffStructure, BMF)

        ### computing the mass weighted ionized fraction
        self.lowres_massweighting = lowres_massweighting
        self.ion_frac_massweighted = np.empty(len(self.z))
        if self.COMPUTE_MASSWEIGHTED_IONFRAC:
            self.compute_massweighted_xHII(CosmoParams, self.lowres_massweighting)
        
        if self.PRINT_TIMER:
            print_timer(self._start_time, text_before="Total computation time: ")
        

    def generate_density(self, ClassyCosmo, CorrFClass):
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Generating density field...")
        #Generating matter power spectrum at the lowest redshift
        klist = CorrFClass._klistCF
        pk_matter = np.zeros_like(klist)
        for i, k in enumerate(klist):
            pk_matter[i] = ClassyCosmo.pk(k, self.z_of_density)
        pk_spl = spline(np.log(klist), np.log(pk_matter))
    
        #generating density map
        if self.LOGNORMAL_DENSITY:
            pb = pbox.LogNormalPowerBox(N=self.ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=self.boxlength, seed=self.seed)
        else:
            pb = pbox.PowerBox(N=self.ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=self.boxlength, seed=self.seed)
        density_field = pb.delta_x()
        if self.PRINT_TIMER:
            print_timer(start_time, text_before="    done in ")
        return density_field

    def generate_density_allz(self, CosmoParams):
        if self.PRINT_TIMER:
            start_time = time.time()
            print('Evolving density field...')
        Dg = CosmoParams.growthint(self.z)
        growthfactor_ratio = (Dg/Dg[0])[:, np.newaxis, np.newaxis, np.newaxis]
        density_lastz = np.copy(self.density)
        self.density_allz = density_lastz[np.newaxis]*growthfactor_ratio
        if self.PRINT_TIMER:
            print_timer(start_time, text_before="    done in ")
        return self.density_allz

    def compute_k(self):
        klistfftx = np.fft.fftfreq(self.ncells,self.dx)*2*np.pi
        k = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        return k

    def smooth_density(self):
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Smoothing density field...")
        density_fft = np.fft.fftn(self.density)
        density_smoothed_allr = np.array([z21_utilities.tophat_smooth(rr, self._k, density_fft) for rr in self.r])
        if self.PRINT_TIMER:
            print_timer(start_time, text_before="    done in ")
        return density_smoothed_allr

    def generate_xHII(self, CosmoParams, CoeffStructure, BMF):
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Generating ionized field...")
        ion_field_allz = np.zeros((len(self.z),self.ncells,self.ncells,self.ncells))
        ion_frac = np.zeros(len(self.z))
        for i in trange(len(self.z)):
            curr_z_idx = self._z_idx[i]
            ion_field = self.ionize(CosmoParams, CoeffStructure, curr_z_idx)
            ion_field_allz[i] = ion_field
            ion_frac[i] = np.sum(ion_field)/(self.ncells**3)
        if self.PRINT_TIMER:
            print_timer(start_time, text_before="    done in ")
        return ion_field_allz, ion_frac

    def ionize(self,CosmoParams, CoeffStructure, curr_z_idx):
        zlist = CoeffStructure.zintegral
        Rs = CoeffStructure.Rtabsmoo
        
        Dg0 = CosmoParams.growthint(zlist[0])
        Dg = CosmoParams.growthint(zlist[curr_z_idx])
        if not self.SPHERIZE:
            ion_field = np.any(self.density_smoothed_allr > (Dg0/Dg)*self.barrier[curr_z_idx, self._r_idx][:, None, None, None], axis=0)
        else:
            ion_field_Rs = np.zeros((len(self._r_idx),self.ncells,self.ncells,self.ncells))
            for j in range(len(self._r_idx)):
                curr_R = self._r_idx[j]
                ion_field_oneR = self.density_smoothed_allr[j] > (Dg0/Dg)*self.barrier[curr_z_idx, curr_R]
                ion_field_oneR_fft = np.fft.fftn(ion_field_oneR)
                cutoff = 1/(4/3*np.pi*Rs[curr_R]**3)/2*(1+self.barrier[curr_z_idx, curr_R]) #comment
                ion_spheres = z21_utilities.tophat_smooth(Rs[curr_R]/(1+self.barrier[curr_z_idx, curr_R])**(1/3), self._k, ion_field_oneR_fft) > cutoff
                ion_field_Rs[j] = ion_spheres
                    
            ion_field = np.any(ion_field_Rs, axis=0)
        return ion_field
    
    def compute_massweighted_xHII(self, CosmoParams, lowres_massweighting=1):
        if np.sum(self.density_allz[0, 0, 0]) == 0.:
            self.generate_density_allz(CosmoParams)
        self.lowres_massweighting = lowres_massweighting
        if self.lowres_massweighting < 1:
            raise Exeption('lowres_massweighting should be >=1.')
        if not isinstance(self.lowres_massweighting, (int, np.int32, np.int64)):
            raise Exeption('lowres_massweighting should be an integer.')
        d_allz = self.density_allz[:, ::self.lowres_massweighting, ::self.lowres_massweighting, ::self.lowres_massweighting]
        ion_allz = self.ion_field_allz[:, ::self.lowres_massweighting, ::self.lowres_massweighting, ::self.lowres_massweighting]
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing mass weighted ionized fraction...")
        self.ion_frac_massweighted = np.average((1+d_allz) * ion_allz, axis=(1, 2, 3))
        if self.PRINT_TIMER:
            print_timer(start_time, text_before="    done in ")
        return self.ion_frac_massweighted

