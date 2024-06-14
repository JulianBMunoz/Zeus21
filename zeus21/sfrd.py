"""

Bulk of the Zeus21 calculation. Compute sSFRD from cosmology, determines Lyman-alpha and X-ray fluxes, and evolves the cosmic-dawn IGM state (WF coupling and heating). From that we get the 21-cm global signal and the effective biases gammaR to determine the 21-cm power spectrum.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

from . import cosmology
from .xrays import Xray_class, sigma_HI, sigma_HeI
from . import constants

import numpy as np




class get_T21_coefficients:
    "Loops through SFRD integrals and obtains avg T21 and the coefficients for its power spectrum. Takes input zmin, which minimum z we integrate down to. It accounts for: \
    -Xray heating \
    -LyA coupling. \
    TODO: reionization/EoR"

    def __init__(self, Cosmo_Parameters, ClassCosmo, Astro_Parameters, HMF_interpolator, zmin = 5.0):

        self.Rtabsmoo = Cosmo_Parameters._Rtabsmoo
        self.dlogRR = Cosmo_Parameters._dlogRR


        #define the integration redshifts, goes as log(z) (1+ doesn't change sampling much)
        self.zmax_integral = constants.ZMAX_INTEGRAL
        self.zmin = zmin
        self._dlogzint_target = 0.02/constants.precisionboost
        self.Nzintegral = np.ceil(1.0 + np.log(self.zmax_integral/self.zmin)/self._dlogzint_target).astype(int)
        self.dlogzint = np.log(self.zmax_integral/self.zmin)/(self.Nzintegral-1.0) #exact value rather than input target above
        self.zintegral = np.logspace(np.log10(self.zmin), np.log10(self.zmax_integral), self.Nzintegral) #note these are also the z at which we "observe", to share computational load

        self.SFRD_avg = np.zeros_like(self.zintegral)
        self.SFRDbar2D = np.zeros((self.Nzintegral, Cosmo_Parameters.NRs)) #SFR at z=zprime when averaged over a radius R (so up to a higher z)
        self.gamma_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta)
        self.gamma2_index2D = np.zeros_like(self.SFRDbar2D) #quadratic fit gamma like ~ exp(\gamma \delta + \gamma_2 \delta^2)
        
        self.niondot_avg = np.zeros_like(self.zintegral) #\dot nion at each z (int d(SFRD)/dM *fesc(M) dM)/rhobaryon
        self.gamma_Niondot_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma \delta)
        self.gamma2_Niondot_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma \delta + \gamma_2 \delta^2)

        self.ztabRsmoo = np.zeros_like(self.SFRDbar2D) #z's that correspond to each Radius R around each zp

        #EoR coeffs
        self.sigmaofRtab = np.array([HMF_interpolator.sigmaR_int(self.Rtabsmoo, zz) for zz in self.zintegral]) #to be used in correlations.py, in get_bubbles()

        fesctab = fesc(Astro_Parameters, HMF_interpolator.Mhtab) #prepare fesc(M) table -- z independent for now so only once


        #Xray coeffs
        self.coeff1Xzp = np.zeros_like(self.zintegral) #zp-dependent coeff in Xray calculation
        self.coeff2XzpRR = np.zeros_like(self.SFRDbar2D) #zp and R-dependent coeff in Xray calculation
        self.Tk_avg = np.zeros_like(self.zintegral) #average kinetic temperature

        #LyA coeffs
        self.coeff1LyAzp = np.zeros_like(self.zintegral) # Same but for LyA
        self.coeff2LyAzpRR = np.zeros_like(self.SFRDbar2D)# Same but for LyA
        self.Jalpha_avg = np.zeros_like(self.zintegral) #avg Jalpha (we compute xa at the end)

        _Jalpha_coeffs = np.zeros([constants.n_max_recycle-1,Cosmo_Parameters.NRs]) #the line recycled coeffs


        #and EPS factors
        Nsigmad = .2 #how many sigmas we explore (.2 originally)
        Nds = 3 #how many deltas (2 originally)
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)


        #initialize Xrays
        Xrays = Xray_class(Cosmo_Parameters)
        _Energylist = Astro_Parameters.Energylist


        #here goes the bulk of the work
        for izp, zp in enumerate(self.zintegral):
            dzp = self.dlogzint*zp


            self.coeff1Xzp[izp] = -2./3. * dzp/cosmology.Hubinvyr(Cosmo_Parameters,zp)/(1+zp) * (1+zp)**2.0 * constants.yrTos
            #Units of coeff1 are s^1. Does not include fheat or fion yet
            self.coeff1Xzp[izp] /= (1.0+zp)**2
            #this accounts for adiabatic cooling. compensated by the inverse at the end

            #self.coeff1LyAzp[izp] = (1+zp)**2/(4*np.pi)/cosmology.rho_baryon(Cosmo_Parameters,zp) * constants.Mpctoyr
            self.coeff1LyAzp[izp] = (1+zp)**2/(4*np.pi)
            #dimless


            chizp = ClassCosmo.comoving_distance(zp)
            chitosolve = chizp + self.Rtabsmoo
            chimax = Cosmo_Parameters._chitab[-1]
            chitosolve[chitosolve > chimax] = chimax #make sure we don't get chis for z outside of interpolation range. Either way nothing for z>zmax_CLASS~50 will be used.


            ztabedgeRsmoo =  cosmology.redshift_of_chi(Cosmo_Parameters, chitosolve)



            self.ztabRsmoo[izp]  = ztabedgeRsmoo
            if(Cosmo_Parameters.Flag_emulate_21cmfast == True): #21cmFAST takes it at the midpoint of the two shells. In dr really.
                self.ztabRsmoo[izp]  = np.append( (zp+ztabedgeRsmoo[0])/2.0, (ztabedgeRsmoo[1:]+ztabedgeRsmoo[:-1])/2.0 )


            indexkeepRsmoo = np.array(self.ztabRsmoo[izp] < Cosmo_Parameters.zmax_CLASS) #to ensure we dont integrate to z higher than our interpolation tables

            #for LyA we can calculate the recycle factors outside the inner loop:
            for n_rec in range(constants.n_max_recycle-1):
                n_line = n_rec + 2 #we start at LyA
                zmax_line = (1+zp)*(1 - pow(1+n_line,-2.0))/(1-pow(n_line,-2.0) ) - 1.0
                nu_linezp = constants.freqLyCont * (1 - (1.0/n_line)**2)
                nu_lineRR = nu_linezp * (1.+self.ztabRsmoo[izp])/(1+zp)
                eps_alphaRR = Astro_Parameters.SED_LyA(nu_lineRR) #two power laws



                zedgeLyAtab = self.ztabRsmoo[izp]

                weights_rec = np.heaviside(zmax_line - zedgeLyAtab, 0.0)


                #the last nonzero index of the array is overestimated since only part of the spherical shell is within zmax_line. Correct by by dz/Delta z
                _index_firstzero_weights = np.count_nonzero(weights_rec)-1
                if(_index_firstzero_weights+1>=len(zedgeLyAtab) and n_rec==0): #only warn on LyA
                    print('WARNING: Your Rsmmax may not be large enough (widen Rtabsmoo). The LyA calculation may not be converged at the largest zmax.')
                elif(_index_firstzero_weights>=0):
                    weights_rec[_index_firstzero_weights] *= (zmax_line - zedgeLyAtab[_index_firstzero_weights])/(zedgeLyAtab[_index_firstzero_weights+1] - zedgeLyAtab[_index_firstzero_weights])
                _Jalpha_coeffs[n_rec] = constants.fractions_recycle[n_rec] * weights_rec*eps_alphaRR
                #print(n_line, zmax_line, _Jalpha_coeffs[n_rec])
            LyAintegral = Astro_Parameters.N_alpha_perbaryon/Cosmo_Parameters.mu_baryon_Msun  * np.sum(_Jalpha_coeffs,axis=0)
            #print(LyAintegral.shape)



            for iR,RR in enumerate(self.Rtabsmoo[indexkeepRsmoo]):

                zRR = self.ztabRsmoo[izp,iR]

                sigmaR = HMF_interpolator.sigmaR_int(RR,zRR) #sigma(R)

                sigmacurr = HMF_interpolator.sigma_int(HMF_interpolator.Mhtab,zRR) #sigma(M)
                modsigmasq = sigmacurr**2 - sigmaR**2


                indextoobig = (modsigmasq <= 0.0)
                modsigmasq[indextoobig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
                modsigma = np.sqrt(modsigmasq)

                dsigmadMcurr = HMF_interpolator.dsigmadM_int(HMF_interpolator.Mhtab,zRR)
                dlogSdMcurr = (dsigmadMcurr*sigmacurr*2.0)/(modsigmasq) #d(log(S))/dM = dsigma^2/dM * 1/S,  S=sigmaM^2 - sigmaR^2 = modsigmasq. Input to EPS HMF (used in 21cmFAST mode). Algebra written out for clarity


                #calculate avg SFRD
                HMF_curr = HMF_interpolator.HMF_int(HMF_interpolator.Mhtab,zRR)
                SFRtab_curr = SFR(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, zRR)

                integrand = HMF_curr * SFRtab_curr * HMF_interpolator.Mhtab
                self.SFRDbar2D[izp,iR] = np.trapz(integrand, HMF_interpolator.logtabMh)


                if(iR==0):  #only the zp term (R->0)
                    self.niondot_avg[izp] = np.trapz(integrand * fesctab, HMF_interpolator.logtabMh) #multiply by Nion/baryon outside the loop


                #First do the average part
                #For Xrays:
                currEnergylist = _Energylist * (1+zRR)/(1+zp)
                _SEDtab = Astro_Parameters.SED_XRAY(currEnergylist)


                weights_X_z = Xrays.opacity_Xray(Cosmo_Parameters, _Energylist,zp, zRR) #input energy at zp, since it'll be redshifted inside the integral
                JX_coeffs = _SEDtab * weights_X_z



                sigma_times_en = Xrays.atomfractions[0] * sigma_HI(_Energylist) * (_Energylist - Xrays.atomEnIon[0])
                sigma_times_en += Xrays.atomfractions[1] * sigma_HeI(_Energylist) * (_Energylist - Xrays.atomEnIon[1])
                sigma_times_en /= np.sum(Xrays.atomfractions)#to normalize per baryon, instead of per Hydrogen nucleus
                #HI and HeII separate. Notice Energy (and not Energy'), since they get absorbed at the zp frame


                XrayEnergyintegral = np.sum(JX_coeffs* sigma_times_en * _Energylist * Astro_Parameters.dlogEnergy,axis=0)
                #Note that dEnergy = Energylist * dlogEnergy , since the table is logspaced

                self.coeff2XzpRR[izp,iR] = RR * self.dlogRR * self.SFRDbar2D[izp,iR] * XrayEnergyintegral * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST
                #Units of coeff2XzpRR are erg/s, since normLX*SFR = erg/s (after sigma goes from cm^2 to Mpc^2)


                #(for LyA is outside loop for vectorization)

                #And here calculate the density-dependent stuff -> gamma indices
                if (Nsigmad * sigmaR > 1.0): #to make sure we dont go to delta < -1 or >1.69
                    smax = 0.3
                    deltatab = deltatab_norm * smax
                else:
                    deltatab = deltatab_norm * sigmaR

                SFRD_delta = np.zeros_like(deltatab)
                Niondot_delta = np.zeros_like(deltatab)

                nu0 = Cosmo_Parameters.delta_crit_ST/sigmacurr #the EPS delta = 0 result. Note 1/sigmacurr and not sigmamod
                nu0[indextoobig]=1.0 #set to 1 to avoid under/overflows, we don't sum over those masses since they're too big


                for idelta,dd in enumerate (deltatab): ##for now only +1 and -1 sigma. Assume ~exp(gamma delta), find gamma
                    modd = Cosmo_Parameters.delta_crit_ST - dd
                    nu = modd/modsigma

                    #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
                    if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
                    #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
                        EPS_HMF_corr = (nu/nu0) * (sigmacurr/modsigma)**2.0 * np.exp(-Cosmo_Parameters.a_corr_EPS * (nu**2-nu0**2)/2.0 )
                        integrand_EPS = integrand * EPS_HMF_corr

                    elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): #as 21cmFAST, use PS HMF, integrate and normalize at the end
                        PS_HMF = cosmology.PS_HMF_unnorm(Cosmo_Parameters, HMF_interpolator.Mhtab,nu,dlogSdMcurr)
                        integrand_EPS = PS_HMF * SFRtab_curr * HMF_interpolator.Mhtab
                    else:
                        print("ERROR: Need to set FLAG_EMULATE_21CMFAST at True or False in the gamma_index2D calculation.")

                    integrand_EPS *= (1.0 + dd) #to convert from Lagrangian to Eulerian
                    SFRD_delta[idelta] = np.trapz(integrand_EPS, HMF_interpolator.logtabMh)
                    Niondot_delta[idelta] = np.trapz(integrand_EPS * fesctab, HMF_interpolator.logtabMh) #ionizing emissivity (Niondot) is basically SFRD but with an escape fraction that varies with halo mass


                midpoint = len(SFRD_delta)//2

                self.gamma_index2D[izp,iR] = np.log(SFRD_delta[midpoint+1]/SFRD_delta[midpoint-1])/(deltatab[midpoint+1] - deltatab[midpoint-1]) #notice its der wrt delta, not delta/sigma or growth
                
                self.gamma_Niondot_index2D[izp, iR] = np.log(Niondot_delta[midpoint+1]/Niondot_delta[midpoint-1])/(deltatab[midpoint+1] - deltatab[midpoint-1])
                
                
                '''if (izp == 35)*(iR == 16): #z~10 and R~10Mpc 
                    print(izp)
                    print(iR)
                    print(zp)
                    print(RR)
                    self.testSFRD = SFRD_delta
                    self.testdeltatab = deltatab
                    self.testNiondot = Niondot_delta
                    self.testfesctab = fesctab'''
                    
                
                
                der1 =  np.log(SFRD_delta[midpoint]/SFRD_delta[midpoint-1])/(deltatab[midpoint] - deltatab[midpoint-1]) #ln(y2/y1)/(x2-x1)
                der2 =  np.log(SFRD_delta[midpoint+1]/SFRD_delta[midpoint])/(deltatab[midpoint+1] - deltatab[midpoint]) #ln(y3/y2)/(x3-x2)
                
                self.gamma2_index2D[izp,iR] = (der2 - der1)/(deltatab[midpoint+1] - deltatab[midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)
                
                der1_N = np.log(Niondot_delta[midpoint]/Niondot_delta[midpoint-1])/(deltatab[midpoint] - deltatab[midpoint-1])
                der2_N =  np.log(Niondot_delta[midpoint+1]/Niondot_delta[midpoint])/(deltatab[midpoint+1] - deltatab[midpoint])
                
                self.gamma2_Niondot_index2D[izp, iR] = (der2_N - der1_N)/(deltatab[midpoint+1] - deltatab[midpoint-1])
                


            self.coeff2LyAzpRR[izp] = self.Rtabsmoo * self.dlogRR * self.SFRDbar2D[izp,:] * LyAintegral/ constants.yrTos/constants.Mpctocm**2
            #units of 1/cm^2 * (SFR/Msun=1/s)* 1/Hz


        #correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).
        if(constants.C2_RENORMALIZATION_FLAG==True):
            self.coeff2LyAzpRR*= 1.0 + (self.gamma_index2D-1.0)*self.sigmaofRtab**2
            self.coeff2XzpRR*= 1.0 + (self.gamma_index2D-1.0)*self.sigmaofRtab**2
            _corrfactorEulerian = 1.0 + (self.gamma_index2D-1.0)*self.sigmaofRtab**2
            _corrfactorEulerian=_corrfactorEulerian.T
            _corrfactorEulerian[0:Cosmo_Parameters.indexminNL] = _corrfactorEulerian[Cosmo_Parameters.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately
            self.coeff2LyAzpRR*= _corrfactorEulerian.T
            self.coeff2XzpRR*= _corrfactorEulerian.T
            ## alternative expression below: if you take (1+d)~exp(d) throughout.
            #self.coeff2LyAzpRR *= np.exp(self.sigmaofRtab**2/2.0 * (2.0 *self.gamma_index2D-1.0) )
            #self.coeff2XzpRR *= np.exp(self.sigmaofRtab**2/2.0 * (2.0 *self.gamma_index2D-1.0) )


        self._GammaXray = self.coeff1Xzp*np.sum(self.coeff2XzpRR,axis=1) #notice units are modified (eg 1/H) so it's simplest to sum

        #partial ionization from Xrays. Fit to Furlanetto&Stoever
        fion = 0.4 * np.exp(-cosmology.xefid(Cosmo_Parameters, self.zintegral)/0.2)

        atomEnIonavg = (Xrays.atomfractions[0] *  Xrays.atomEnIon[0] + Xrays.atomfractions[1] *  Xrays.atomEnIon[1])/(Xrays.atomfractions[0] + Xrays.atomfractions[1])
        # atomEnIonavg = Xrays.atomfractions[0] *  Xrays.atomEnIon[0] * sigma_HI(1000.) + Xrays.atomfractions[1] *  Xrays.atomEnIon[1]* sigma_HeI(1000.)
        # atomEnIonavg/=( sigma_HI(1000.) + sigma_HeI(1000.) )

        self.coeff_Gammah_Tx = -Astro_Parameters.L40_xray * constants.ergToK * (1.0+self.zintegral)**2 #convert from one to the other, last factors accounts for adiabatic cooling. compensated by the inverse at zp in coeff1Xzp. Minus because integral goes from low to high z, but we'll be summing from high to low everywhere.


        self.Gammaion = self.coeff_Gammah_Tx *constants.KtoeV * self._GammaXray * fion/atomEnIonavg * 3/2 #atomEnIonavg makes it approximate. No adiabatic cooling (or recombinations) so no 1+z factors. Extra 3/2 bc temperature has a 2/3
        #TODO: Improve model for xe


        self.xe_avg_ad = cosmology.xefid(Cosmo_Parameters, self.zintegral)
        self.xe_avg = self.xe_avg_ad + np.cumsum(self.Gammaion[::-1])[::-1]


        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self.xe_avg = 2e-4 * np.ones_like(self.Gammaion) #we force this when we emualte 21cmdast to compare both codes on the same footing


        #and heat from Xrays
        self._fheat = pow(self.xe_avg,0.225)
        self.coeff1Xzp*=self._fheat #since this is what we use for the power spectrum (and not Gammaheat) we need to upate it
        self.Gammaheat = self._GammaXray * self._fheat


        self.Tk_xray = self.coeff_Gammah_Tx * np.cumsum(self.Gammaheat[::-1])[::-1] #in K, cumsum reversed because integral goes from high to low z. Only heating part

        self.Tk_ad = cosmology.Tadiabatic(Cosmo_Parameters, self.zintegral)
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self.Tk_ad*=0.95 #they use recfast, so their 'cosmo' temperature is slightly off

        self.Tk_avg = self.Tk_ad + self.Tk_xray



        self.Jalpha_avg = self.coeff1LyAzp*np.sum(self.coeff2LyAzpRR,axis=1) #units of 1/(cm^2 s Hz sr)

        self.T_CMB = cosmology.Tcmb(ClassCosmo, self.zintegral)
        

        _tau_GP = 3./2.*cosmology.n_H(Cosmo_Parameters,self.zintegral)*constants.Mpctocm/cosmology.HubinvMpc(Cosmo_Parameters,self.zintegral) * (constants.wavelengthLyA/1e7)**3 * constants.widthLyAcm * (1.0 - self.xe_avg)  #~3e5 at z=6

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            _tau_GP/=Cosmo_Parameters.f_H #for some reason they multiuply by N0 (all baryons) and not NH0.


        _xiHirata = pow(_tau_GP*1e-7,1/3.)*pow(self.Tk_avg,-2./3)

        _factorxi = (1.0 + 2.98394*_xiHirata + 1.53583 * _xiHirata**2 + 3.8528 * _xiHirata**3)



        #prefactor without the Salpha correction from Hirata2006
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self._coeff_Ja_xa_0 = 1.66e11/(1+self.zintegral) #They use a fixed (and slightly ~10% off) value.
        else:
            self._coeff_Ja_xa_0 = 8.0*np.pi*(constants.wavelengthLyA/1e7)**2 * constants.widthLyA * constants.Tstar_21/(9.0*constants.A10_21*self.T_CMB) #units of (cm^2 s Hz sr), convert from Ja to xa. should give 1.81e11/(1+self.zintegral) for Tcmb_0=2.725 K
            
        #collision coefficient fh(1-x_e)
        self.xc_HH = Cosmo_Parameters.f_H * (1.0 - self.xe_avg) * cosmology.n_baryon(Cosmo_Parameters, self.zintegral) * np.interp(self.Tk_avg, constants.Tk_HH, constants.k_HH) / constants.A10_21 * constants.Tstar_21 / cosmology.Tcmb(ClassCosmo, self.zintegral)
        self.xc_He = self.xe_avg * cosmology.n_baryon(Cosmo_Parameters, self.zintegral) * np.interp(self.Tk_avg, constants.Tk_He, constants.k_He) / constants.A10_21 * constants.Tstar_21 / cosmology.Tcmb(ClassCosmo, self.zintegral) #xe
        self.xc_avg = self.xc_HH + self.xc_He


        if(constants.FLAG_WF_ITERATIVE==True): #iteratively find Tcolor and Ts. Could initialize one to zero, but this should converge faster
            _invTs_tryfirst = 1.0/self.T_CMB
            self._invTs_avg = 1.0/self.Tk_avg
        else: #no correction (ie Tcolor=Tk, Salpha= exp(...))
            self.invTcol_avg = 1.0 / self.Tk_avg
            self.coeff_Ja_xa = self._coeff_Ja_xa_0 * Salpha_exp(self.zintegral, self.Tk_avg, self.xe_avg)
            self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg
            self._invTs_avg = (1.0/self.T_CMB+self.xa_avg*self.invTcol_avg+self.xc_avg*self.invTk_avg)/(1+self.xa_avg+self.xc_avg)

            _invTs_tryfirst = self._invTs_avg #so the loop below does not trigger


        while(np.sum(np.fabs(_invTs_tryfirst/self._invTs_avg - 1.0))>0.01): #no more than 1% error total
            _invTs_tryfirst = self._invTs_avg


            #update xalpha
            _Salphatilde = (1.0 - 0.0632/self.Tk_avg + 0.116/self.Tk_avg**2 - 0.401/self.Tk_avg*self._invTs_avg + 0.336*self._invTs_avg/self.Tk_avg**2)/_factorxi
            self.coeff_Ja_xa = self._coeff_Ja_xa_0 * _Salphatilde
            self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg

            #and Tcolor^-1
            self.invTcol_avg = 1.0/self.Tk_avg + constants.gcolorfactorHirata * 1.0/self.Tk_avg * (_invTs_tryfirst - 1.0/self.Tk_avg)

            #and finally Ts^-1
            self._invTs_avg = (1.0/self.T_CMB+self.xa_avg * self.invTcol_avg + self.xc_avg * 1.0/self.Tk_avg)/(1+self.xa_avg+self.xc_avg)



        self.SFRD_avg = self.SFRDbar2D[:,0] #avg SFRD in Msun/yr/Mpc^3 at zp. Approximated by the R->0 case



        #EoR part here:
        _trec0 = 1.0/(constants.alphaB * cosmology.n_H(Cosmo_Parameters,0) *(1 + Cosmo_Parameters.x_He) * Astro_Parameters._clumping)#t_recombination  at z=0, in sec
        _recexp = 1.0/(_trec0 * np.sqrt(Cosmo_Parameters.OmegaM) * cosmology.Hubinvyr(Cosmo_Parameters,0) / constants.yrTos)# = 1/(_trec0 * H0 * sqrt(OmegaM) ), dimless. Assumes matter domination and constant clumping. Can be modified to power-law clumping changing the powerlaw below from 3/2

        self.coeffQzp = self.dlogzint*self.zintegral/cosmology.Hubinvyr(Cosmo_Parameters,self.zintegral)/(1+self.zintegral) #Deltaz * dt/dz. Units of 1/yr, inverse of niondot


        self.niondot_avg *= Astro_Parameters.N_ion_perbaryon/cosmology.rho_baryon(Cosmo_Parameters,0.) #note the rhob(z=0) because it's comoving volume

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #regular calculation, integrating over time and accounting for recombinations in the exponent

            self.Qfactrecomb = np.exp(-2/3 * _recexp * pow(1+self.zintegral,3/2))
            self.Qion_avg = 1/self.Qfactrecomb*np.cumsum(self.coeffQzp[::-1] * self.Qfactrecomb[::-1] * self.niondot_avg[::-1])[::-1]

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True): #21cmfast instead uses nion (rather than niondot and integrating). We can emulate that here. there nion = niondot * t_star/H(z) [see Park+19]. In that case we can iteratively solve for Q=nion - nrecom(Q), where nrecom = int dt Q/t_recom to correct for recombinations. Easier than ODE.

            #self._nion = np.cumsum(self.coeffQzp[::-1] * self.niondot_avg[::-1])[::-1]
            self._nion = self.niondot_avg * Astro_Parameters.tstar/cosmology.Hubinvyr(Cosmo_Parameters,self.zintegral)
            self._Q0iteration = self._nion #0th iteration has no recombinations
            self.trec = _trec0/(1+self.zintegral)**3/constants.yrTos #in yr at each time t
            self._Q1iteration = 0.0
            while(np.sum(np.abs(self._Q1iteration-self._Q0iteration))>0.001):
                self._Q1iteration = self._Q0iteration
                self._nrecombinations = np.cumsum(self.coeffQzp[::-1] * (self._Q1iteration/self.trec)[::-1])[::-1] #coeffQzp = dt/dz as before
                self._Q0iteration = self._nion - self._nrecombinations
            self.Qion_avg = self._Q0iteration


        #common to both methods.
        self.Qion_avg = np.fmin(1.0, self.Qion_avg)
        self._xHII_avg = self.Qion_avg + (1.0 - self.Qion_avg) * self.xe_avg #accounts for partial ionization, small effect
        self.xHI_avg = (1.0 - self._xHII_avg)
        self.xHI_avg = np.fmin(1.0, self.xHI_avg)


        #and finally, get the signal
        self.T21avg = cosmology.T021(Cosmo_Parameters,self.zintegral) * (self.xa_avg + self.xc_avg)/(1.0 + self.xa_avg + self.xc_avg) * (1.0 - self.T_CMB * self.invTcol_avg) * self.xHI_avg



def tau_reio(Cosmo_Parameters, T21_coefficients):
    "Returns the optical depth to reionization given a model. It assumes xHI=1 for z<zmin."
    #we separate into a low- and hi-z parts (z< or > zmini)


    _zlistlowz = np.linspace(0,T21_coefficients.zmin,100)
    _nelistlowz = cosmology.n_H(Cosmo_Parameters,_zlistlowz)*(1.0 + Cosmo_Parameters.x_He + Cosmo_Parameters.x_He * np.heaviside(constants.zHeIIreio - _zlistlowz,0.5))
    _distlistlowz = 1.0/cosmology.HubinvMpc(Cosmo_Parameters,_zlistlowz)/(1+_zlistlowz)

    _lowzint = constants.sigmaT * np.trapz(_nelistlowz*_distlistlowz,_zlistlowz) * constants.Mpctocm



    _zlisthiz = T21_coefficients.zintegral

    _nelistlhiz = cosmology.n_H(Cosmo_Parameters,_zlisthiz) * (1 + Cosmo_Parameters.x_He) * (1.0 - T21_coefficients.xHI_avg)
    _distlisthiz = 1.0/cosmology.HubinvMpc(Cosmo_Parameters,_zlisthiz)/(1+_zlisthiz)

    _hizint = constants.sigmaT * np.trapz(_nelistlhiz*_distlisthiz,_zlisthiz) * constants.Mpctocm

    return(_lowzint + _hizint)


def Matom(z):
    "Returns Matom as a function of z"
    return 3.3e7 * pow((1.+z)/(21.),-3./2)

#fstar = Mstardot/Mhdot, parametrizes as you wish
def fstarofz(Astro_Parameters, Cosmo_Parameters, z, Mhlist):
    epsstar_ofz = Astro_Parameters.epsstar * 10**(Astro_Parameters.dlog10epsstardz * (z-Astro_Parameters._zpivot) )
    return 2.0 * Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM * epsstar_ofz\
        /(pow(Mhlist/Astro_Parameters.Mc,- Astro_Parameters.alphastar) + pow(Mhlist/Astro_Parameters.Mc,- Astro_Parameters.betastar) )


#Only PopII for now (TODO, add PopIII)
def SFR(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, z):
    "SFR in Msun/yr at redshift z. Evaluated at the halo masses Mh [Msun] of the HMF_interpolator, given Astro_Parameters"
    Mh = HMF_interpolator.Mhtab
    if(Astro_Parameters.FLAG_MTURN_FIXED == False):
        fduty = np.exp(-Matom(z)/Mh)
    else:
        if(Astro_Parameters.FLAG_MTURN_SHARP == False): #whether to do regular exponential turn off or a sharp one at Mturn
            fduty = np.exp(-Astro_Parameters.Mturn_fixed/Mh)
        else:
            fduty = np.heaviside(Mh - Astro_Parameters.Mturn_fixed, 0.5)

    if(Astro_Parameters.astromodel == 0): #GALLUMI-like
        fstarM = fstarofz(Astro_Parameters, Cosmo_Parameters, z, Mh)

        if(Astro_Parameters.accretion_model == 0): #exponential accretion
            dMhdz = Mh * constants.ALPHA_accretion_exponential
        elif(Astro_Parameters.accretion_model == 1): #EPS accretion
            Mh2 = Mh * constants.EPSQ_accretion
            indexMh2low = (Mh2 < Mh[0])
            Mh2[indexMh2low] = Mh[0] #to avoid extrapolation to lower M. Those will have
            sigmaMh = HMF_interpolator.sigma_int(Mh,z)
            sigmaMh2 = HMF_interpolator.sigma_int(Mh2,z)
            sigmaMh2[indexMh2low] = 1e99

            growth = cosmology.growth(Cosmo_Parameters,z)
            dzgrow = z*0.01
            dgrowthdz = (cosmology.growth(Cosmo_Parameters,z+dzgrow) - cosmology.growth(Cosmo_Parameters,z-dzgrow))/(2.0 * dzgrow)
            dMhdz = - Mh * np.sqrt(2/np.pi)/np.sqrt(sigmaMh2**2 - sigmaMh**2) *dgrowthdz/growth * Cosmo_Parameters.delta_crit_ST
        else:
            print("ERROR! Have to choose an accretion model in Astro_Parameters (accretion_model)")

        Mhdot = dMhdz*cosmology.Hubinvyr(Cosmo_Parameters,z)*(1.0+z) #Msun/yr

        return Mhdot * fstarM * fduty

    elif(Astro_Parameters.astromodel == 1): #21cmfast-like
        fstarM = Astro_Parameters.fstar10*pow(Mh/1e10,Astro_Parameters.alphastar)
        fstarM = Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM *np.fmin(fstarM,Astro_Parameters.fstarmax)
        return Mh*fstarM/Astro_Parameters.tstar*cosmology.Hubinvyr(Cosmo_Parameters,z) * fduty
    else:
        print('ERROR, MODEL is not defined')

def fesc(Astro_Parameters, Mh):
    "f_escape for a halo of mass Mh [Msun] given Astro_Parameters"
    return np.fmin(1.0, Astro_Parameters.fesc10 * pow(Mh/1e10,Astro_Parameters.alphaesc) )


#Kept for reference purposes. Does not correct x_alpha as a function of Ts iteratively, but some old works don't either so this allows for comparison. Only used if FLAG_WF_ITERATIVE == False
def Salpha_exp(z, T, xe):
    "correction from Eq 55 in astro-ph/0608032, Tk in K evaluated for the IGM where there is small reionization (xHI~1 and xe<<1) during LyA coupling era"
    tau_GP_noreio = 3e5*pow((1+z)/7,3./2.)*(1-xe)
    gamma_Sobolev = 1.0/tau_GP_noreio
    return np.exp( - 0.803 * pow(T,-2./3.) * pow(1e-6/gamma_Sobolev,-1.0/3.0))
