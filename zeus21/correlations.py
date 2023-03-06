"""

Code to compute correlation functions from power spectra and functions of them. Holds two classes: Correlations (with matter correlation functions smoothed over different R), and Power_Spectra (which will compute and hold the 21-cm power spectrum and power for derived quantities like xa, Tk, etc.)

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import mcfit
from scipy.special import gammaincc #actually very fast, no need to approximate

from . import constants
from . import cosmology



class Correlations:
    "Class that calculates and keeps the correlation functions."

    def __init__(self, Cosmo_Parameters, ClassCosmo):


        #we choose the k to match exactly the log FFT of input Rtabsmoo.

        self._klistCF, _dummy_ = mcfit.xi2P(Cosmo_Parameters._Rtabsmoo, l=0, lowring=True)(0*Cosmo_Parameters._Rtabsmoo, extrap=False)
        self.NkCF = len(self._klistCF)

        self._PklinCF = np.zeros(self.NkCF) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = ClassCosmo.pk(kk, 0.0) # function .pk(k,z)



        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)

        self.WINDOWTYPE = 'TOPHAT'
        #options are 'TOPHAT', 'TOPHAT1D' and 'GAUSS' (for now). TOPHAT is calibrated for EPS, but GAUSS has less ringing

        self.xi_RR_CF = self.get_xi_R1R2_z0(Cosmo_Parameters)


    def _WinTH(self,k,R):
        x = k * R
        return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

    def _WinTH1D(self,k,R):
        x = k * R
        return  np.sin(x)/x

    def _WinG(self,k,R):
        x = k * R * constants.RGauss_factor
        return np.exp(-x**2/2.0)

    def Window(self,R):
        if self.WINDOWTYPE == 'TOPHAT':
            return self._WinTH(self._klistCF,R)
        elif self.WINDOWTYPE == 'GAUSS':
            return self._WinG(self._klistCF,R)
        elif self.WINDOWTYPE == 'TOPHAT1D':
            return self._WinTH1D(self._klistCF,R)
        else:
            print('ERROR in Window. Wrong type')


    def get_xi_z0_lin(self):
        "Get correlation function of density, linearly extrapolated to z=0"
        ##Warning: definitely check if beyond LCDM!
        #currenetly unused, just for refernce and plots

        rslinCF, xilinCF = self._xif(self._PklinCF, extrap=False)

        return rslinCF, xilinCF

    def get_xi_R1R2_z0 (self, Cosmo_Parameters):
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) \
        same separations rs as get_xi_z0_lin so it does not output them."

        xi_RR_CF = np.zeros((Cosmo_Parameters.NRs, Cosmo_Parameters.NRs, self.NkCF))

        for iR1, RR1 in enumerate(Cosmo_Parameters._Rtabsmoo):
            for iR2, RR2 in enumerate(Cosmo_Parameters._Rtabsmoo):
                if(iR1 > iR2):
                    xi_RR_CF[iR1,iR2] = xi_RR_CF[iR2,iR1]
                else:
                    _PkRR = self._PklinCF * self.Window(RR1) * self.Window(RR2)
                    self.rlist_CF, xi_mcfit = self._xif(_PkRR, extrap=False)

                    xi_RR_CF[iR1,iR2] = xi_mcfit

        return xi_RR_CF



class Power_Spectra:
    "Get power spetrum from correlation functions and coefficients"

    def __init__(self, Cosmo_Parameters, ClassCosmo, Correlations, T21_coefficients, RSD_MODE=1):


        #set up some variables
        self._rs_input_mcfit = Correlations.rlist_CF #just to make notation simpler
        self.klist_PS = Correlations._klistCF

        self.RSD_MODE = RSD_MODE #redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space #TODO: mode to save at different mu



        #first get the linear window functions -- note it already has growth factor in it, so it multiplies Pmatter(z=0)
        self.kwindow, self.windowalpha = self.get_xa_window(Cosmo_Parameters, Correlations, T21_coefficients)
        self._kwindowX, self.windowxray = self.get_Tx_window(Cosmo_Parameters, Correlations, T21_coefficients)


        #calculate some growth etc, and the bubble biases for the xHI linear window function:
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, T21_coefficients.zintegral)

        if(constants.FLAG_DO_BUBBLES):
            self.calculate_barrier(Cosmo_Parameters, T21_coefficients)
            self.get_bubbles(Cosmo_Parameters, Correlations, T21_coefficients)
            self.windowxion = np.array([Correlations.Window(self.Rbub_star[iz]) for iz in range(T21_coefficients.Nzintegral)]) #Window returns a k-array. Smooths at the peak of the BMF

            self.windowxion = (self.windowxion.T*T21_coefficients.Qstar * self.bias_bub_avg * self._lingrowthd * np.exp(-T21_coefficients.Qstar) ).T #normalize
        else:
            self.windowxion = np.zeros_like(self.windowalpha)




        #finally, get all the nonlinear correlation functions:
        self.get_all_corrs(Cosmo_Parameters, Correlations, T21_coefficients) #get them

        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)


        #and now define power spectra:
        #for xalpha, first linear
        self._Pk_xa_lin = self.windowalpha**2 * Correlations._PklinCF
        self.Deltasq_xa_lin = self._Pk_xa_lin * self._k3over2pi2 #note that it still has units of xa_avg

        #nonlinear corrections too:
        #self.deltaxi_xa = self.get_xa_deltaCF(Cosmo_Parameters, Correlations, T21_coefficients) #get them
        self._d_Pk_xa_nl = self.get_list_PS(self._deltaxi_xa, T21_coefficients.zintegral)
        self.Deltasq_xa = self.Deltasq_xa_lin + self._d_Pk_xa_nl * self._k3over2pi2 #note that it still has units of xa_avg


        # #and same for xray
        self._Pk_Tx_lin = self.windowxray**2 * Correlations._PklinCF
        self.Deltasq_Tx_lin = self._Pk_Tx_lin * self._k3over2pi2

        #self.deltaxi_Tx = self.get_Tx_deltaCF(Cosmo_Parameters, Correlations, T21_coefficients) #get them
        self._d_Pk_Tx_nl = self.get_list_PS(self._deltaxi_Tx, T21_coefficients.zintegral)
        self.Deltasq_Tx = self.Deltasq_Tx_lin + self._d_Pk_Tx_nl * self._k3over2pi2


        #and their cross correlation
        self._Pk_xaTx_lin = self.windowalpha * self.windowxray * Correlations._PklinCF
        self.Deltasq_xaTx_lin = self._Pk_xaTx_lin * self._k3over2pi2

        self._d_Pk_xaTx_nl = self.get_list_PS(self._deltaxi_xaTx, T21_coefficients.zintegral)
        self.Deltasq_xaTx = self.Deltasq_xaTx_lin + self._d_Pk_xaTx_nl * self._k3over2pi2 #note that it still has units of xa_avg


        #and the same for deltaNL and its cross terms:
        self._Pk_d_lin = np.outer(self._lingrowthd**2, Correlations._PklinCF)
        self.Deltasq_d_lin = self._Pk_d_lin * self._k3over2pi2 #note that it still has units of xa_avg

        self._Pk_dxa_lin = (self.windowalpha.T * self._lingrowthd).T * Correlations._PklinCF
        self._Pk_dTx_lin = (self.windowxray.T * self._lingrowthd).T * Correlations._PklinCF
        self.Deltasq_dxa_lin = self._Pk_dxa_lin * self._k3over2pi2
        self.Deltasq_dTx_lin = self._Pk_dTx_lin * self._k3over2pi2

        self._Pk_d =  self._Pk_d_lin
        self._Pk_dxa =  self._Pk_dxa_lin
        self._Pk_dTx =  self._Pk_dTx_lin

        if(constants.FLAG_DO_DENS_NL): #note that the nonlinear terms (cross and auto) below here have the growth already accounted for
            self._d_Pk_d_nl = self.get_list_PS(self._deltaxi_d, T21_coefficients.zintegral)
            self._Pk_d += self._d_Pk_d_nl
            self._d_Pk_dxa_nl = self.get_list_PS(self._deltaxi_dxa, T21_coefficients.zintegral)
            self._Pk_dxa += self._d_Pk_dxa_nl
            self._d_Pk_dTx_nl = self.get_list_PS(self._deltaxi_dTx, T21_coefficients.zintegral)
            self._Pk_dTx += self._d_Pk_dTx_nl

        self.Deltasq_d = self._Pk_d * self._k3over2pi2
        self.Deltasq_dxa = self._Pk_dxa * self._k3over2pi2
        self.Deltasq_dTx = self._Pk_dTx * self._k3over2pi2



        #and xHI too. Linear part does not have bubbles, only delta part
        if(constants.FLAG_DO_BUBBLES):
            #auto
            self._Pk_xion_lin = self.windowxion**2 * Correlations._PklinCF
            self.Deltasq_xion_lin = self._Pk_xion_lin * self._k3over2pi2

            self._d_Pk_xion_nl = self.get_list_PS(self._deltaxi_xi, T21_coefficients.zintegral)
            self.Deltasq_xion = self.Deltasq_xion_lin + self._d_Pk_xion_nl * self._k3over2pi2

            #cross with density
            self._Pk_dxion_lin = (self.windowxion.T * self._lingrowthd).T  * Correlations._PklinCF
            self.Deltasq_dxion_lin = self._Pk_dxion_lin * self._k3over2pi2

            self._d_Pk_dxion_nl = self.get_list_PS(self._deltaxi_dxi, T21_coefficients.zintegral)
            self.Deltasq_dxion = self.Deltasq_dxion_lin + self._d_Pk_dxion_nl * self._k3over2pi2

            #cross with xa
            self._Pk_xaxion_lin = self.windowxion * self.windowalpha  * Correlations._PklinCF
            self.Deltasq_xaxion_lin = self._Pk_xaxion_lin * self._k3over2pi2

            self._d_Pk_xaxion_nl = self.get_list_PS(self._deltaxi_xaxi, T21_coefficients.zintegral)
            self.Deltasq_xaxion = self.Deltasq_xaxion_lin + self._d_Pk_xaxion_nl * self._k3over2pi2

            #and cross with Tx
            self._Pk_Txxion_lin = self.windowxion * self.windowxray  * Correlations._PklinCF
            self.Deltasq_Txxion_lin = self._Pk_Txxion_lin * self._k3over2pi2

            self._d_Pk_Txxion_nl = self.get_list_PS(self._deltaxi_Txxi, T21_coefficients.zintegral)
            self.Deltasq_Txxion = self.Deltasq_Txxion_lin + self._d_Pk_Txxion_nl * self._k3over2pi2
        else:
            self.Deltasq_xion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_xion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_dxion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_dxion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_xaxion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_xaxion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_Txxion = np.zeros_like(self.Deltasq_d)
            self.Deltasq_Txxion_lin = np.zeros_like(self.Deltasq_d)
            #These have to be defined even if no EoR bubbles



        #and get the PS of T21 too.
        self._betaT = T21_coefficients.T_CMB*T21_coefficients.invTcol_avg/(T21_coefficients.Tk_avg - T21_coefficients.T_CMB) #multiplies \delta T_x and \delta T_ad [both dimensionful, not \deltaT/T]
        self._betaxa = 1./(1. + T21_coefficients.xa_avg)/T21_coefficients.xa_avg #multiplies \delta x_a [again not \delta xa/xa]

        #calculate beta_adiabatic
        self._dlingrowthd_dz = cosmology.dgrowth_dz(Cosmo_Parameters, T21_coefficients.zintegral)

        _factor_adi_ = (1+T21_coefficients.zintegral)**2
        _integrand_adi = T21_coefficients.Tk_avg*self._dlingrowthd_dz/_factor_adi_ * T21_coefficients.dlogzint*T21_coefficients.zintegral

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            _hizintegral = 0.0 #they do not account for the adiabatic history prior to starting their evolution. It misses ~half of the adiabatic flucts.
        else:
            #the z>zmax part of the integral we do aside. Assume Tk=Tadiabatic from CLASS.
            _zlisthighz_ = np.linspace(T21_coefficients.zintegral[-1], 99., 100) #beyond z=100 need to explictly tell CLASS to save growth
            _dgrowthhighz_ = cosmology.dgrowth_dz(Cosmo_Parameters, _zlisthighz_)
            _hizintegral = np.trapz(cosmology.Tadiabatic(Cosmo_Parameters,_zlisthighz_)
            /(1+_zlisthighz_)**2 * _dgrowthhighz_, _zlisthighz_)

        self._betaTad_ = -2./3. * _factor_adi_/self._lingrowthd * (np.cumsum(_integrand_adi[::-1])[::-1] + _hizintegral) #units of Tk_avg. Internal sum goes from high to low z (backwards), minus sign accounts for it properly so it's positive.
        self._betaTad_ *= self._betaT #now it's dimensionless, since it multiplies \delta_m(k,z)



        self._betad = (1.0 + self._betaTad_)# this includes both the usual (1+d) and the adiabatic Tk contribution. Now we add RSD
        if(self.RSD_MODE==0): #no RSD (real space)
            pass #nothing to change
        elif(self.RSD_MODE==1): #spherically avg'd RSD
            self._betad += constants.MU_AVG ** 2
        elif(self.RSD_MODE==2): #LoS RSD (mu=1)
            self._betad += constants.MU_LoS ** 2
        else:
            print('Error, have to choose an RSD mode! RSD_MODE')

        if(constants.FLAG_DO_BUBBLES):
            self._betaxion = - 1.0/T21_coefficients.xHI_avg * np.heaviside(constants.ZMAX_Bubbles - T21_coefficients.zintegral, 0.5) # xion = 1 - xHI, only for z<ZMAX_Bubbles. 1/xHI_avg since P_xHI has units of xHI
        else:
            self._betaxion = np.zeros_like(T21_coefficients.xHI_avg) # do not do EoR bubbles at all



        #To first order: dT21/T0 = (1+cT * betaTad) * delta_m + betaT * deltaTX + betaxa * delta xa + betaxion * delta xion

        self._allbetas = np.array([self._betad, self._betaxa, self._betaT, self._betaxion])
        self._allbetamatrix = np.einsum('ij,kj->ikj', self._allbetas, self._allbetas)

        self._allcorrs = np.array( [[self.Deltasq_d, self.Deltasq_dxa, self.Deltasq_dTx, self.Deltasq_dxion], \
                                    [self.Deltasq_dxa, self.Deltasq_xa, self.Deltasq_xaTx, self.Deltasq_xaxion], \
                                    [self.Deltasq_dTx, self.Deltasq_xaTx, self.Deltasq_Tx, self.Deltasq_Txxion], \
                                    [self.Deltasq_dxion, self.Deltasq_xaxion, self.Deltasq_Txxion, self.Deltasq_xion]]\
                                        )

        self.Deltasq_T21 = np.einsum('ijk...,ijkl...->kl...', self._allbetamatrix, self._allcorrs)
        #self.Deltasq_T21 = (self.Deltasq_T21.T*self._T021list**2).T
        self.Deltasq_T21 = (self.Deltasq_T21.T*T21_coefficients.T21avg**2).T



        self._allcorrs_lin = np.array( [[self.Deltasq_d_lin, self.Deltasq_dxa_lin, self.Deltasq_dTx_lin, self.Deltasq_dxion_lin], \
                                    [self.Deltasq_dxa_lin, self.Deltasq_xa_lin, self.Deltasq_xaTx_lin, self.Deltasq_xaxion_lin], \
                                    [self.Deltasq_dTx_lin, self.Deltasq_xaTx_lin, self.Deltasq_Tx_lin, self.Deltasq_Txxion_lin], \
                                    [self.Deltasq_dxion_lin, self.Deltasq_xaxion_lin, self.Deltasq_Txxion_lin, self.Deltasq_xion_lin]]\
                                        )

        self.Deltasq_T21_lin = np.einsum('ijk...,ijkl...->kl...', self._allbetamatrix, self._allcorrs_lin)
        self.Deltasq_T21_lin = (self.Deltasq_T21_lin.T*T21_coefficients.T21avg**2).T





    def get_xa_window(self, Cosmo_Parameters, Correlations, T21_coefficients):
        "Returns the xa window function for all z in zintegral"

        _Nkwin = len(T21_coefficients.Rtabsmoo)
        _win_alpha = np.zeros((T21_coefficients.Nzintegral,_Nkwin))

        for izp,zp in enumerate(T21_coefficients.zintegral):

            coeffzp = T21_coefficients.coeff1LyAzp[izp]
            coeffJaxa = T21_coefficients.coeff_Ja_xa[izp]

            zpRlist = T21_coefficients.ztabRsmoo[izp]
            growthRlist = cosmology.growth(Cosmo_Parameters,zpRlist)

            coeffR = T21_coefficients.coeff2LyAzpRR[izp]
            gammaR = T21_coefficients.gamma_index2D[izp] * growthRlist

            _wincoeffs = coeffR * gammaR #array in logR space

            if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
                _wincoeffs /=(4*np.pi * T21_coefficients.Rtabsmoo**2) * (T21_coefficients.Rtabsmoo * T21_coefficients.dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2 * Delta r)
                _kwinalpha, _win_alpha[izp] = self.get_Pk_from_xi(T21_coefficients.Rtabsmoo, _wincoeffs)

                #for reference only, here's how it is done with a sine Transform
                # _wincoeffs_alt = coeffR * gammaR/(Coeffs.Rtabsmoo)/(Coeffs.Rtabsmoo* Coeffs.dlogRR)/np.sqrt(2/np.pi) #array in logR space
                # kPf_alt, Pf_alt = mcfit.transforms.FourierSine(Coeffs.Rtabsmoo, lowring=True)(_wincoeffs_alt, extrap=False)
                # Pf_alt/=kPf_alt
            else:
                _kwinalpha = self.klist_PS
                _win_alpha[izp] = np.array([np.sum(coeffR * gammaR*Correlations._WinTH(T21_coefficients.Rtabsmoo,kk)) for kk in _kwinalpha]) #here we do it ``brute force" rather than FFT since the 3D window function makes the fourier transform blow up at low k because cancellations between sinc(x) and cos(x). Rather than brute force in resolution it's cheaper to just do a sum, and more accurate

            _win_alpha[izp] *= coeffzp*coeffJaxa


        return _kwinalpha, _win_alpha


    def get_Tx_window(self, Cosmo_Parameters,  Correlations, T21_coefficients):
        "Returns the Tx window function for all z in zintegral"

        _Nkwin = len(T21_coefficients.Rtabsmoo)
        _win_Tx = np.zeros((T21_coefficients.Nzintegral,_Nkwin))



        for izp,zp in reversed(list(enumerate(T21_coefficients.zintegral))): #reversed order sum to go from high to low z

            if (izp < len(T21_coefficients.zintegral)-1):#start by summing over the previous one. izp1 decreases between steps
                _win_Tx[izp] = _win_Tx[izp+1]

            coeffzp = T21_coefficients.coeff1Xzp[izp]

            zpRlist = T21_coefficients.ztabRsmoo[izp]
            growthRlist = cosmology.growth(Cosmo_Parameters,zpRlist)

            coeffR = T21_coefficients.coeff2XzpRR[izp]
            gammaR = T21_coefficients.gamma_index2D[izp] * growthRlist

            if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
                _wincoeffs = coeffR * gammaR #array in logR space
                _wincoeffs /=(4*np.pi * T21_coefficients.Rtabsmoo**2) * (T21_coefficients.Rtabsmoo * T21_coefficients.dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2) * Delta r
                _kwinTx, _win_Tx_curr = self.get_Pk_from_xi(T21_coefficients.Rtabsmoo, _wincoeffs)
                #minus because we're summing from hi to low z
                _win_Tx[izp] += _win_Tx_curr * coeffzp

            else: #3D, check the lya case above for more info
                _kwinTx = self.klist_PS
                _win_Tx[izp] += coeffzp * np.array([np.sum(coeffR * gammaR*Correlations._WinTH(T21_coefficients.Rtabsmoo,kk)) for kk in _kwinTx])



        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx #z-dependent, includes 10^40 erg/s/SFR normalizaiton and erg/K conversion factor, and the 1/(1+z)^2 factor to compensate the adiabatic cooling of the Tx olny part
        _win_Tx =(_win_Tx.T*_coeffTx_units).T


        return _kwinTx, _win_Tx




    def get_all_corrs(self, Cosmo_Parameters, Correlations, T21_coefficients):
        "Returns the correlation function of all observable at each z in zintegral"

        #_iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)
        _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL)
        corrdNL = Correlations.xi_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]

        #for R<RNL fix at RNL, avoids corelations blowing up at low R
        corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]

        corr_deltaR1R2z0 = np.transpose(corrdNL, (2,0,1)) #so we can broadcast it to the sum below, first index is zp now

        self._deltaxi_Tx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_xa = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_xaTx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))

        self._deltaxi_d = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_dxa = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_dTx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))


        self._deltaxi_xi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_dxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_xaxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
        self._deltaxi_Txxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))



        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        for izp1,zp1 in reversed(list(enumerate(T21_coefficients.zintegral))): #reversed order sum to go from high to low z
            if (izp1 < len(T21_coefficients.zintegral)-1):#start by summing over the previous one. izp1 decreases between steps
                self._deltaxi_Tx[izp1] = self._deltaxi_Tx[izp1+1]

            zpRlist1 = T21_coefficients.ztabRsmoo[izp1,_iRnonlinear]
            growthRlist1 = cosmology.growth(Cosmo_Parameters,zpRlist1)
            gammaR1 = T21_coefficients.gamma_index2D[izp1,_iRnonlinear] * growthRlist1

            coeffzp1xa = T21_coefficients.coeff1LyAzp[izp1] * T21_coefficients.coeff_Ja_xa[izp1]
            coeffzp1Tx = T21_coefficients.coeff1Xzp[izp1] #we can't multiply at the end because each object is summed over previous z too.

            coeffR1xa = T21_coefficients.coeff2LyAzpRR[izp1,_iRnonlinear]
            coeffR1Tx = T21_coefficients.coeff2XzpRR[izp1,_iRnonlinear]



            #for xa auto corr and delta we don't need the second zp2 sum:
            gammamatrixR1R1 = np.outer(gammaR1,gammaR1)
            coeffmatrixxa = np.outer(coeffR1xa,coeffR1xa)
            self._deltaxi_xa[izp1] = np.sum(coeffmatrixxa * ((np.exp(gammamatrixR1R1 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R1 * corr_deltaR1R2z0) , axis=(1,2))
            self._deltaxi_xa[izp1] *= (coeffzp1xa)**2 #brings it to xa units



            if(constants.FLAG_DO_DENS_NL):
                self._deltaxi_dxa[izp1] = np.sum(coeffR1xa * ((np.exp(gammaR1 * growthRlist1[0] * corr_deltaR1R2z0[:,0])-1.0) - gammaR1 * growthRlist1[0] * corr_deltaR1R2z0[:,0]) , axis=(1))
                self._deltaxi_dxa[izp1] *= coeffzp1xa #brings it to xa units
                #for d-d autocorrelation only keep the most local term
                self._deltaxi_d[izp1] = (np.exp(growthRlist1[0]**2 * corr_deltaR1R2z0[:,0,0])-1.0) - growthRlist1[0]**2 * corr_deltaR1R2z0[:,0,0]


            if(constants.FLAG_DO_BUBBLES):
                _indexRbub = self._Rbub_star_index[izp1] - Cosmo_Parameters.indexminNL #to get the correct array element, if too large/small for nonlinearities just skip the nonlinear part, outside of k range
            else:
                _indexRbub = -1

            _flag_doEoRNL = constants.FLAG_DO_BUBBLES and zp1 < constants.ZMAX_Bubbles and T21_coefficients.Qion_avg[izp1] < 1.0 and _indexRbub >= 0 and _indexRbub < len(_iRnonlinear)
            #all these things have to be true for us to run the nonlinear+bubble part

            if(_flag_doEoRNL):
                _eminusQstar = np.exp(-T21_coefficients.Qstar[izp1])
                gammaeffxHI = -T21_coefficients.Qstar[izp1] * self.bias_bub_avg[izp1] * growthRlist1[0] #effective bias of the xion term. includes growth

                self._deltaxi_xaxi[izp1] = np.sum(coeffR1xa * ((np.exp(gammaR1 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub])-1.0) - gammaR1 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub]) , axis=(1))
                self._deltaxi_xaxi[izp1] *= coeffzp1xa * _eminusQstar #brings it to xa units

                self._deltaxi_dxi[izp1] =  (1.0 - np.exp(gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,0,_indexRbub]) ) -  gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,0,_indexRbub]
                self._deltaxi_dxi[izp1] *= _eminusQstar

                #for autocorrelation we have a density and a bubble/random term. first density
                self._deltaxi_xi[izp1] =  (np.exp(-2.0 * gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,_indexRbub,_indexRbub]) -1.0) -  (-2.0) * gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,_indexRbub,_indexRbub]
                #plus the bubble part, fully nonlinear, no "correction wrt linear"
                self._deltaxi_xi[izp1] += (np.exp(self.Qo_tab[izp1]) - 1.0)

                self._deltaxi_xi[izp1] *= _eminusQstar**2


            for izp2,zp2 in reversed(list(enumerate(T21_coefficients.zintegral))): #double loop because nonlocal in time sum.

                _factorzp1equalzp2 = 2.0 #factor for 2 or 1 depending on whether they are the same for the sum below
                if (izp2 < izp1): #sum only for z >= zp1, not below
                    continue
                elif (izp2 == izp1):
                    _factorzp1equalzp2 = 1.0


                coeffzp2Tx = T21_coefficients.coeff1Xzp[izp2] #inside zp2 it's always Tx since it's the nonlocal-in-time one
                zpRlist2 = T21_coefficients.ztabRsmoo[izp2,_iRnonlinear]
                growthRlist2 = cosmology.growth(Cosmo_Parameters,zpRlist2)

                gammaR2 = T21_coefficients.gamma_index2D[izp2,_iRnonlinear] * growthRlist2
                gammamatrixR1R2 = np.outer(gammaR1,gammaR2)


                coeffR2Tx = T21_coefficients.coeff2XzpRR[izp2,_iRnonlinear]
                coeffmatrixTxTx = np.outer(coeffR1Tx,coeffR2Tx)
                coeffmatrixxaTx = np.outer(coeffR1xa,coeffR2Tx)

                self._deltaxi_Tx[izp1] += _factorzp1equalzp2 * coeffzp1Tx * coeffzp2Tx * np.sum(coeffmatrixTxTx * ((np.exp(gammamatrixR1R2 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R2 * corr_deltaR1R2z0) , axis=(1,2))

                self._deltaxi_xaTx[izp1] += coeffzp2Tx * np.sum(coeffmatrixxaTx * ((np.exp(gammamatrixR1R2 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R2 * corr_deltaR1R2z0) , axis=(1,2))

                if(constants.FLAG_DO_DENS_NL):
                    self._deltaxi_dTx[izp1] += coeffzp2Tx * np.sum(coeffR2Tx * ((np.exp(gammaR2* growthRlist1[0] * corr_deltaR1R2z0[:,0])-1.0) - gammaR2* growthRlist1[0] * corr_deltaR1R2z0[:,0]) , axis=(1))

                if(_flag_doEoRNL):
                    self._deltaxi_Txxi[izp1] += coeffzp2Tx * np.sum(coeffR2Tx * ((np.exp(gammaR2 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub])-1.0) - gammaR2 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub]) , axis=(1))


            self._deltaxi_xaTx[izp1]*= coeffzp1xa
            self._deltaxi_xaTx[izp1]*=_coeffTx_units[izp1]

            if(constants.FLAG_DO_DENS_NL):
                self._deltaxi_dTx[izp1]*=_coeffTx_units[izp1]

            if(_flag_doEoRNL):
                self._deltaxi_Txxi[izp1]*=_coeffTx_units[izp1] * _eminusQstar



        self._deltaxi_Tx=(self._deltaxi_Tx.T*_coeffTx_units**2).T #we cannot easily do this in the loop because it sums over previous ones

        return 1


    def get_list_PS(self, xi_list, zlisttoconvert):
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []

        for izp,zp in enumerate(zlisttoconvert):

            _kzp, _Pkzp = self.get_Pk_from_xi(self._rs_input_mcfit,xi_list[izp])
            _Pk_list.append(_Pkzp)
            #can ignore _kzp, it's the same as klist_PS above by construction


        return np.array(_Pk_list)


    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf


#
#     def calculate_barrier(self, Cosmo_Parameters, T21_coefficients):
#         "Caclulate the barrier B(z, sigmaR) that the density \delta has to cross to ionize"
#
#         self.Barrier0list = np.zeros_like(T21_coefficients.zintegral)
#         self.Barrier1list = np.zeros_like(T21_coefficients.zintegral)
#
#         sigmaminsqlist = (T21_coefficients.sigmaMatom * self._lingrowthd/self._lingrowthd[0])**2
#         sigmapivotsqlist = (T21_coefficients.sigmaMpivot * self._lingrowthd/self._lingrowthd[0])**2
#         #notice sigmaMatom depends on z and sigmaMpivot doesn't. For now at least. Code doesn't care since _lingrowthd does depend on z anyway
#
#
#         sigmaRref = np.sqrt(sigmaminsqlist/20.)
#         #pick this one for reference to take d/dsigmaR^2
#
#         alphaeff = T21_coefficients._alphaeff #note that if alpha_eff = 0 you recover erfc. For negative it can behave weird so beware (for instance voids reionize first. Not physical)
#
#         plindex = -T21_coefficients.dlogMdlogsigma
#         #M~sigma^-plindex
#
#         totalindex = plindex * alphaeff
#         sindex = 1./2. + totalindex
#
#         for izp, zp in enumerate(T21_coefficients.zintegral):
#
#             if zp>constants.ZMAX_Bubbles:
#                 continue
#
#             _invQbar = 1.0/T21_coefficients.Qion_avg[izp] #we need Nion/<Nion> > 1/invQbar to ionize the region. larger delta at higher z
#
#
#
#             dtab = np.linspace(-3.0 * sigmaRref[izp] , 3.0 * sigmaRref[izp] , 99)
#             dtabhi = np.linspace(3.3 * sigmaRref[izp], 1.5, 30)
#             dtab = np.append(dtab, dtabhi)
#
#             dtildetabsq = (constants.delta_crit_ST - dtab)**2
#
#
#             tabsigmasqit = [0.8*sigmaRref[izp]**2, 1.4*sigmaRref[izp]**2] #to get derivatives wrt sigma^2
#
#             barrier = np.zeros_like(tabsigmasqit)
#
#             for isigma, sigmaRRsq in enumerate(tabsigmasqit):
#
#                 mumintildesq = dtildetabsq/(sigmaminsqlist[izp] - sigmaRRsq)
#                 mupivottildesq = dtildetabsq/sigmapivotsqlist[izp]
#
#
#                 NionEPS = pow(dtildetabsq, - totalindex) * (gammaincc(sindex,mumintildesq/2.0) - gammaincc(sindex,mupivottildesq/2.0))
#
#                 Probdtab = np.exp(-dtab**2/sigmaRRsq/2.0)
#
#                 norm = np.trapz(NionEPS * Probdtab, dtab)
#                 NionEPS/=norm
#
#                 bindex = min(range(len(NionEPS)), key=lambda i: abs(NionEPS[i]-_invQbar))
#
#                 barrier[isigma] = dtab[bindex]
#
#             self.Barrier0list[izp] = np.sum(barrier)/len(barrier) #sigma-indep
#             self.Barrier1list[izp] = (barrier[-1] - barrier[0])/(tabsigmasqit[-1] - tabsigmasqit[0]) #linear in sigmaR^2
#
#     def get_bubbles(self, Cosmo_Parameters, Correlations, T21_coefficients):
#         "Returns the Bubble mass function for EoR"
#
#
#         _Rtab = T21_coefficients.Rtabsmoo
#         _rhob0 = cosmology.rho_baryon(Cosmo_Parameters, 0.)
#         _Mtab = _rhob0 * 4.0 * np.pi * _Rtab**3/3.0  #at z=0 because comoving
#         _dMdR = _rhob0 * 4.0 * np.pi * _Rtab**2 #at z=0 because comoving
#         _dlogMdlogR = 3.0
#         _dlog_Mtab = _dlogMdlogR * T21_coefficients.dlogRR
#
#
#         self.BMF_array = np.zeros_like(T21_coefficients.gamma_Niondot_index2D)
#         #bubble mass function, dn/dm in 1/cMpc^3/Msun
#
#         self.Qo_tab = np.zeros_like(self.BMF_array)
#         #Q_overlap, integral of [BMF * Voverlap(r)] at _Rtab
#         _Voverlap = np.array([[Voverlap(Rbb, rr) for Rbb in _Rtab] for rr in _Rtab])
#         #index is [ir, iRb]
#
#
#         self.Q_infer_BMF = np.zeros(T21_coefficients.Nzintegral)
#
#
#         self.Rbub_star = np.zeros(T21_coefficients.Nzintegral) #peak of BMF
#         self._Rbub_star_index = np.zeros(T21_coefficients.Nzintegral, dtype=int) #its index in Rsmoo
#         self.bias_bub_avg = np.zeros(T21_coefficients.Nzintegral) #avg (mass-weighted) bias
#
#
#         for izp, zp in enumerate(T21_coefficients.zintegral):
#
#             if (zp > constants.ZMAX_Bubbles or T21_coefficients.Qion_avg[izp] >= 1.0): #only do below a threshold and before EoR is complete to avoid numerical noise
#                 continue
#
#             sigmaofRtab = T21_coefficients.sigmaofRtab[izp]
#             logsigmaoflogR_f = UnivariateSpline(np.log(_Rtab),np.log(sigmaofRtab) )
#             dlogsigmadlogR_f = logsigmaoflogR_f.derivative()
#             dlogsigmadlogRtab = dlogsigmadlogR_f(np.log(_Rtab) )
#
#
#
#             B0 = self.Barrier0list[izp] #Fit is Barrier = B0 + B1 sigma^2
#             B1 = self.Barrier1list[izp]
#             Btab = B0 + B1 * sigmaofRtab**2
#
#             dlogsigmadlogMtab = dlogsigmadlogRtab / _dlogMdlogR
#
#             self.BMF_array[izp] = np.sqrt(2.0/np.pi) * _rhob0/(_Mtab**2) * np.abs(dlogsigmadlogMtab) * B0/sigmaofRtab * np.exp(-Btab**2/(2.0 * sigmaofRtab**2))
#
#
#             self.Q_infer_BMF[izp] = np.sum(self.BMF_array[izp] * _Mtab/_rhob0 * _Mtab)*_dlog_Mtab
#
#
#             self.BMF_array[izp] *= T21_coefficients.Qstar[izp]/self.Q_infer_BMF[izp] #renormalized now
#
#             self._bias_bubbles_zp = 1.0 + B0**2/(Btab * sigmaofRtab**2) #Eulerian bias of a bubble of some mass/radius at zp
#
#             self.bias_bub_avg[izp] = np.sum(self.BMF_array[izp] * self._bias_bubbles_zp * _Mtab/_rhob0 * _Mtab)*_dlog_Mtab/T21_coefficients.Qstar[izp]
#             #average bias
#
#
#             _dimlessBMF = _Mtab**2 * self.BMF_array[izp]
#             self._Rbub_star_index[izp] = max(range(len(_Mtab)), key=lambda i: _dimlessBMF[i])
#             self.Rbub_star[izp] = _Rtab[self._Rbub_star_index[izp]] #the maximum of the BMF
#
#
#             self.Qo_tab[izp] = np.array([np.sum(self.BMF_array[izp] * Vtab * _Mtab)*_dlog_Mtab for Vtab in _Voverlap])
#         self.Rbub_star = np.fmax(self.Rbub_star, 1e-3) #to avoid Nans in other functions
#
#
# def Voverlap(Rb, r):
#     "Overlapping volume of two bubbles of radius Rb separated by r. From FZH04"
#     return ((4 * np.pi/3.0) * Rb**3 - np.pi * r * (Rb**2 - r**2/12.)) * np.heaviside( 2*Rb - r , 0.5)
