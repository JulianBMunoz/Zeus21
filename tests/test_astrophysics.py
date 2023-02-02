"""

Astrophysics tests for Zeus21, SFRD, global, and 21-cm PS

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

import pytest
import zeus21
import numpy as np

from zeus21.sfrd import *
from zeus21.correlations import *


CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 100.) #to speed up a little
ClassyCosmo = zeus21.runclass(CosmoParams_input)
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo)
HMFintclass = zeus21.HMF_interpolator(CosmoParams,ClassyCosmo)



AstroParams = zeus21.Astro_Parameters(CosmoParams)
ZMIN = 20.0 #down to which z we compute the evolution
Coeffs = zeus21.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)

#also for exponential accretion:
AstroParams_expacc = zeus21.Astro_Parameters(CosmoParams, accretion_model=0)

#and for the 21cmfast mode:
CosmoParams_input_21cmfast = zeus21.Cosmo_Parameters_Input(Flag_emulate_21cmfast=True)
ClassyCosmo_21cmfast = zeus21.runclass(CosmoParams_input_21cmfast)
CosmoParams_21cmfast = zeus21.Cosmo_Parameters(CosmoParams_input_21cmfast, ClassyCosmo_21cmfast)
AstroParams_21cmfast = zeus21.Astro_Parameters(CosmoParams_21cmfast, astromodel = 1)


ztest = 20.
iztest = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))

#test the background and globals
def test_background():

    #test SFR first
    sSFR = SFR(AstroParams, CosmoParams, HMFintclass, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR).all()) #positive
    assert( (sSFR/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all()) #make sure sSFR/H < 1 (not all mass forms stars in a Hubble time)

    #repeat for the other 2 cases
    sSFR_exp = SFR(AstroParams_expacc, CosmoParams, HMFintclass, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR_exp).all())
    assert( (sSFR_exp/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all())

    sSFR_21cmfast = SFR(AstroParams_21cmfast, CosmoParams_21cmfast, HMFintclass, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR_21cmfast).all())
    assert( (sSFR_21cmfast/zeus21.cosmology.Hubinvyr(CosmoParams_21cmfast,ztest) <= 1).all())


    assert( (0 <= fesc(AstroParams, HMFintclass.Mhtab)).all())
    assert( (fesc(AstroParams, HMFintclass.Mhtab <= 1)).all())



    #and sfrd calculation
    assert( (Coeffs.ztabRsmoo[iztest] >= Coeffs.zintegral[iztest]).all())

    assert( (Coeffs.sigmaofRtab >= 0.0).all()) #all Ts positive



    assert( (Coeffs.xe_avg >= Coeffs.xe_avg_ad).all()) #xrays only add ionization, not subtract

    assert( (Coeffs.Tk_ad >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Tk_xray >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Tk_avg >= Coeffs.Tk_ad).all()) #and it's a sum
    assert( (Coeffs.T_CMB >= 0.0).all()) #all Ts positive
    assert( (Coeffs._invTs_avg >= 0.0).all()) #all Ts positive
    assert( (Coeffs.invTcol_avg >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Jalpha_avg >= 0.0).all())
    assert( (Coeffs.xa_avg >= 0.0).all())
    assert( (Coeffs._fheat >= 0.0).all())

    #make sure Ts is between Tcmb and Tc
    assert( (Coeffs.invTcol_avg >= Coeffs._invTs_avg).all()) #all Ts positive
    assert( (Coeffs._invTs_avg >= 1/Coeffs.T_CMB).all()) #all Ts positive

    assert( (Coeffs.SFRDbar2D >= 0.0).all())
    assert( (Coeffs.SFRD_avg >= 0.0).all())
    assert( (Coeffs.niondot_avg >= 0.0).all())

    assert( (Coeffs.xHI_avg >= 0.0).all())
    assert( (Coeffs.xHI_avg <= 1.0).all())

    assert( (Coeffs.T21avg <= zeus21.cosmology.T021(CosmoParams,Coeffs.zintegral)).all()) #capped from above
    assert( (Coeffs.T21avg >= (- 300.0) ).all()) #capped from below (IN LCDM!)



    assert( (Coeffs.gamma_index2D >= 0.0).all()) #effective biases have to be larger than 0 in reasonable models, since galaxies live in haloes that are more clustered than average matter (in other words, SFRD grows monotonically with density)




#and test the PS too
CorrFClass = zeus21.Correlations(CosmoParams, ClassyCosmo)
PS21 = zeus21.Power_Spectra(CosmoParams, ClassyCosmo, CorrFClass, Coeffs)


def test_pspec():

    assert((PS21._rs_input_mcfit == CorrFClass.rlist_CF).all())
    assert((PS21.klist_PS == CorrFClass._klistCF).all())
    assert((PS21.kwindow == PS21._kwindowX).all())

    ztest = 20.
    iztest = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))

    assert((PS21.windowalpha[iztest,0] >= PS21.windowalpha[iztest,-1]).all()) #at fixed z it should go down with k
    assert((PS21.windowxray[iztest,0] >= PS21.windowxray[iztest,-1]).all())

    #make sure all correlations are sensible
    assert( (PS21.Deltasq_dxa[iztest]**2 <= 1.01* PS21.Deltasq_d[iztest] * PS21.Deltasq_xa[iztest]).all())
    assert( (PS21.Deltasq_dTx[iztest]**2 <= 1.01* PS21.Deltasq_d[iztest] * PS21.Deltasq_Tx[iztest]).all())
    assert( (PS21.Deltasq_xaTx[iztest]**2 <= 1.01* PS21.Deltasq_Tx[iztest] * PS21.Deltasq_xa[iztest]).all())

    assert( (PS21.Deltasq_dxa_lin[iztest]**2 <= 1.01* PS21.Deltasq_d_lin[iztest] * PS21.Deltasq_xa_lin[iztest]).all())
    assert( (PS21.Deltasq_dTx_lin[iztest]**2 <= 1.01* PS21.Deltasq_d_lin[iztest] * PS21.Deltasq_Tx_lin[iztest]).all())
    assert( (PS21.Deltasq_xaTx_lin[iztest]**2 <= 1.01* PS21.Deltasq_Tx_lin[iztest] * PS21.Deltasq_xa_lin[iztest]).all())

    #also make sure all Pk(k) < avg^2 for all quantities at some k~0.1
    ktest = 0.1
    iktest = min(range(len(PS21.klist_PS)), key=lambda i: np.abs(PS21.klist_PS[i]-ktest))

    assert( (PS21.Deltasq_xa[:,iktest] <= 1.01*Coeffs.xa_avg**2 ).all())
    assert( (PS21.Deltasq_Tx[:,iktest] <= 1.01*Coeffs.Tk_xray**2).all())
    assert( (PS21.Deltasq_T21[:,iktest] <= 1.01*(Coeffs.T21avg)**2).all()) #can fail near T21~0. If so add an offset outside the **2.
