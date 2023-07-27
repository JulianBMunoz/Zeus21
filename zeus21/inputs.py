"""

Takes inputs and stores them in useful classes

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

from . import constants

import numpy as np
from classy import Class
from scipy.interpolate import interp1d


class Cosmo_Parameters_Input:
    "Class to pass the 6 LCDM parameters as input"

    def __init__(self, omegab= 0.0223828, omegac = 0.1201075, h_fid = 0.67810, As = 2.100549e-09, ns = 0.9660499, tau_fid = 0.05430842, kmax_CLASS = 500., zmax_CLASS = 50.,zmin_CLASS = 5., Flag_emulate_21cmfast = False):

        self.omegab = omegab
        self.omegac = omegac
        self.h_fid = h_fid
        self.As = As
        self.ns = ns
        self.tau_fid = tau_fid

        #other params for CLASS
        self.kmax_CLASS = kmax_CLASS
        self.zmax_CLASS = zmax_CLASS
        self.zmin_CLASS = zmin_CLASS
        #and whether to emulate 21cmFAST
        self.Flag_emulate_21cmfast = Flag_emulate_21cmfast #whether to emulate 21cmFAST in HMF, LyA, and X-ray opacity calculations



class Cosmo_Parameters:
    "Class that will keep the cosmo parameters throughout"

    def __init__(self, CosmoParams_input, ClassCosmo):

        self.omegab = CosmoParams_input.omegab
        self.omegac = CosmoParams_input.omegac
        self.h_fid = CosmoParams_input.h_fid
        self.As = CosmoParams_input.As
        self.ns = CosmoParams_input.ns
        self.tau_fid = CosmoParams_input.tau_fid

        #other params in the input
        self.kmax_CLASS = CosmoParams_input.kmax_CLASS
        self.zmax_CLASS = CosmoParams_input.zmax_CLASS
        self.zmin_CLASS = CosmoParams_input.zmin_CLASS #when to start the HMF calcs., not an input strictly
        self.Flag_emulate_21cmfast = CosmoParams_input.Flag_emulate_21cmfast #whether to emulate 21cmFAST in HMF, LyA, and X-ray opacity calculations

        #derived params
        self.omegam = self.omegab + self.omegac
        self.OmegaM = ClassCosmo.Omega_m()
        self.rhocrit = 2.78e11*self.h_fid**2 #Msun/Mpc^3
        self.OmegaR = ClassCosmo.Omega_r()
        self.OmegaL = ClassCosmo.Omega_Lambda()
        self.OmegaB = ClassCosmo.Omega_b()

        self.Y_He = ClassCosmo.get_current_derived_parameters(['YHe'])['YHe']
        self.f_He = self.Y_He/4.0/(1.0 - 3.0/4.0 * self.Y_He) #=nHe/nb
        self.f_H = (1.0 - self.Y_He)/(1.0 - 3.0/4.0 * self.Y_He) #=nH/nb

        self.mu_baryon = (self.f_H + self.f_He * 4.) * 0.94 #mproton ~ 0.94 GeV
        self.mu_baryon_Msun = self.mu_baryon/constants.MsuntoGeV


        #for R->M conversions for HMF. Used for CLASS input so assumes tophat.
        self.constRM = self.OmegaM*self.rhocrit * 4.0 * np.pi/3.0

        self.rho_M0 = self.OmegaM*self.rhocrit



        self._ztabinchi = np.linspace(0.0, 100. , 10000) #cheap so do a lot
        # self._chitab = ClassCosmo.z_of_r(self._ztabinchi)[0]
        # self.zfofRint = interp1d(self._chitab, self._ztabinchi)
        self._chitab, self._Hztab = ClassCosmo.z_of_r(self._ztabinchi) #chi and dchi/dz
        self.zfofRint = interp1d(self._chitab, self._ztabinchi)
        self.chiofzint = interp1d(self._ztabinchi,self._chitab)
        self.Hofzint = interp1d(self._ztabinchi,self._Hztab)

        _thermo = ClassCosmo.get_thermodynamics()
        self.Tadiabaticint = interp1d(_thermo['z'], _thermo['Tb [K]'])
        self.xetanhint = interp1d(_thermo['z'], _thermo['x_e'])

        _ztabingrowth = np.linspace(0., 100. , 2000)
        _growthtabint = np.array([ClassCosmo.scale_independent_growth_factor(zz) for zz in _ztabingrowth])

        self.growthint = interp1d(_ztabingrowth,_growthtabint)


        #and define the shells that we integrate over at each z.
        self.Rsmmin = 0.5
        self.Rsmmax = 2000.

        if(self.Flag_emulate_21cmfast==True):
            self.Rsmmin = 0.62*1.5 #same as minmum R in 21cmFAST for their standard 1.5 Mpc cell resolution. 0.62 is their 'L_FACTOR'
            self.Rsmmax = 500. #same as R_XLy_MAX in 21cmFAST. Too low?

        self.NRs = np.floor(45*constants.precisionboost).astype(int)
        self._Rtabsmoo = np.logspace(np.log10(self.Rsmmin), np.log10(self.Rsmmax), self.NRs) # Smoothing Radii in Mpc com
        self._dlogRR = np.log(self.Rsmmax/self.Rsmmin)/(self.NRs-1.0)

        self.indexminNL = (np.log(constants.MIN_R_NONLINEAR/self.Rsmmin)/self._dlogRR).astype(int)
        self.indexmaxNL = (np.log(constants.MAX_R_NONLINEAR/self.Rsmmin)/self._dlogRR).astype(int) + 1 #to ensure it captures MAX_R


        #HMF-related constants
        if(self.Flag_emulate_21cmfast == False): #standard, best fit ST from Schneider+
            self.a_ST = 0.707 #OG ST fit, or 0.85 to fit 1805.00021
            self.p_ST = 0.3
            self.Amp_ST = 0.3222
            self.delta_crit_ST = 1.686
            self.a_corr_EPS = self.a_ST #correction to the eps relation between nu and nu' when doing extended PS. Follows hi-z simulation results from Schneider+
        elif(self.Flag_emulate_21cmfast == True): #emulate 21cmFAST, including HMF from Jenkins 2001
            self.a_ST = 0.73
            self.p_ST = 0.175
            self.Amp_ST = 0.353
            self.delta_crit_ST = 1.68
            self.a_corr_EPS = 1.0
        else:
            print("Error! Have to set either Flag_emulate_21cmfast = True or False")



class Astro_Parameters:
    "Class to pass the astro parameters as input"

    def __init__(self, Cosmo_Parameters, astromodel = 0, epsstar = 0.1, alphastar = 0.5, betastar = -0.5, Mc = 3e11, fesc10 = 0.1, alphaesc = 0.0, \
                 L40_xray = 3.0, E0_xray = 500., alpha_xray = -1.0, Emax_xray_norm=2000, Nalpha_lyA = 9690, Mturn_fixed = None, FLAG_MTURN_SHARP= False, \
                    accretion_model = 0, sigmaUV=0.5, C0dust = 4.43, C1dust = 1.99):


        if(Cosmo_Parameters.Flag_emulate_21cmfast==True and astromodel == 0):
            print('ERROR, picked astromodel = 0 but tried to emulate 21cmFAST. They use astromodel = 1. Changing it!')
            self.astromodel = 1
        else:
            self.astromodel = astromodel # which SFR model we use. 0=Gallumi-like, 1=21cmfast-like


        #SFR(Mh) parameters:
        self.epsstar = epsstar #epsilon_* = f* at Mc
        self.alphastar = alphastar #powerlaw index for lower masses
        self.betastar = betastar #powerlaw index for higher masses, only for model 0
        self.Mc = Mc # mass at which the power law cuts, only for model 0
        self.sigmaUV = sigmaUV #stochasticity (gaussian rms) in the halo-galaxy connection P(MUV | Mh) - TODO: only used in UVLF not sfrd

        if self.astromodel == 0: #GALUMI-like
            self.accretion_model = accretion_model #0 = exponential, 1= EPS #choose the accretion model. Default = EPS
        elif self.astromodel == 1: #21cmfast-like, ignores Mc and beta and has a t* later in SFR()
            self.tstar = 0.5
            self.fstar10 = self.epsstar
            self.fstarmax = 1.0 #where we cap it
        else:
            print('ERROR, need to pick astromodel')

        #fesc(M) parameter. Power law normalized (fesc10) at M=1e10 Msun with index alphaesc
        self.fesc10 = fesc10
        self.alphaesc = alphaesc
        self._clumping = 3.0 #clumping factor, z-independent and fixed for now
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self._clumping = 2.0 #this is the 21cmFAST value



        #xray parameters here, assumed power-law for now
        self.L40_xray = L40_xray # soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr)
        self.E0_xray = E0_xray #minimum energy in eV
        self.Emax_xray_norm = Emax_xray_norm #max energy in eV to normalize SED. Keep at 2000 eV normally
        self.Emax_xray_integral = 10000. #max energy in eV that we integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z
        self.alpha_xray = alpha_xray #Xray SED power-law index

        if(self.E0_xray < constants.EN_ION_HI):
            print('What the heck? How can E0_XRAY < EN_ION_HI ?')



        #table with how many energies we integrate over
        self.Nen_xray = 30
        self._log10EMIN_INTEGRATE = np.log10(self.E0_xray/2.0) # to account for photons coming from higher z that redshift
        self._log10EMAX_INTEGRATE = np.log10(self.Emax_xray_integral)
        self.Energylist = np.logspace(self._log10EMIN_INTEGRATE,self._log10EMAX_INTEGRATE,self.Nen_xray) #in eV
        self.dlogEnergy = (self._log10EMAX_INTEGRATE - self._log10EMIN_INTEGRATE)/(self.Nen_xray-1.0)*np.log(10.) #to get dlog instead of dlog10


        self.N_alpha_perbaryon=Nalpha_lyA #number of photons between LyA and Ly Cont. per baryon (from LB05)
        self.N_ion_perbaryon = 5000 #fixed for PopII-type (Salpeter)


        if(Mturn_fixed == None):
            self.FLAG_MTURN_FIXED = False #whether to fix Mturn or use Matom(z) at each z
        else:
            self.FLAG_MTURN_FIXED = True #whether to fix Mturn or use Matom(z) at each z
            self.Mturn_fixed = Mturn_fixed
            self.FLAG_MTURN_SHARP = FLAG_MTURN_SHARP #whether to do sharp cut at Mturn_fixed or regular exponential cutoff. Only active if FLAG_MTURN_FIXED and turned on by hand.

        #dust parameters for UVLFs:
        self.C0dust, self.C1dust = C0dust, C1dust #4.43, 1.99 is Meurer99; 4.54, 2.07 is Overzier01
        self._kappaUV = 1.15e-28 #SFR/LUV, value from Madau+Dickinson14, fully degenerate with epsilon




    def SED_XRAY(self, En):
        "SED of our Xray sources, normalized to integrate to 1 from E0_xray to Emax_xray (int dE E * SED(E), and E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). Takes energy En in eV"
        if np.abs(self.alpha_xray+1.0) < 0.01: #log
            norm = 1.0/np.log(self.Emax_xray_norm/self.E0_xray) / self.E0_xray
        else:
            norm = (1.0 + self.alpha_xray)/((self.Emax_xray_norm/self.E0_xray)**(1 + self.alpha_xray) - 1.0) / self.E0_xray

        return np.power(En/self.E0_xray,self.alpha_xray)/En * norm * np.heaviside(En - self.E0_xray, 0.5)
        #do not cut at higher energies since they redshift into <2 keV band


    def SED_LyA(self, nu_in):
        "SED of our Lyman-alpha-continuum sources, normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays) "

        nucut = constants.freqLyB #above and below this freq different power laws
        amps = np.array([0.68,0.32]) #Approx following the stellar spectra of BL05. Normalized to unity

        indexbelow = 0.14 #if one of them zero worry about normalization
        normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
        indexabove = -8.0
        normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]

        nulist = np.asarray([nu_in]) if np.isscalar(nu_in) else np.asarray(nu_in)

        result = np.zeros_like(nulist)
        for inu, currnu in enumerate(nulist):
            if (currnu<constants.freqLyA or currnu>=constants.freqLyCont):
                result[inu] = 0.0
            elif (currnu < nucut): #between LyA and LyB
                result[inu] = normbelow * (currnu/nucut)**indexbelow
            elif (currnu >= nucut):  #between LyB and Continuum
                result[inu] = normabove * (currnu/nucut)**indexabove
            else:
                print("Error in SED_LyA, whats the frequency Kenneth?")


        return result/nucut #extra 1/nucut because dnu, normalizes the integral
