"""
Test the SED.py module, containing the SED adopted in Zeus21, including LyAlpha, Xrays, UV, HAlpha

Authors: zeus21 v2 collaboration - June 2026
    Emily Bregou ;
    Hector Afonso G. Cruz ;
    Sarah Libanore ;
    Julian B. Muñoz ;
    Yonny Sklansky ;
    Emilie Thélie ;
    Alessandra Venditti

arXiv:2302.08506, arXiv:2306.09403, arXiv:2407.18294, Sklansky et al. (in prep)
"""

import pytest
import zeus21
import numpy as np


def test_SED():

    UserParams = zeus21.User_Parameters()
    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams)
    AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams)

    # test Xray SED
    Energylisttest = np.logspace(2,np.log10(AstroParams.Emax_xray_norm),100)

    # popII
    SEDXtabII_test = zeus21.SED_XRAY(AstroParams= AstroParams, 
                                   En = Energylisttest, pop = 2)
    # popIII
    SEDXtabIII_test = zeus21.SED_XRAY(AstroParams= AstroParams, 
                                   En = Energylisttest, pop = 3)
    
    normalization_XraySED_II = np.trapezoid(Energylisttest * SEDXtabII_test, Energylisttest)
    normalization_XraySED_III = np.trapezoid(Energylisttest * SEDXtabIII_test,Energylisttest)

    assert(normalization_XraySED_II == pytest.approx(1.0, 0.05) ) #5% is enough here
    assert( normalization_XraySED_III == pytest.approx(1.0, 0.05) ) #5% is enough here

    # test LyA SED
    nulisttest = np.linspace(zeus21.constants.freqLyA, zeus21.constants.freqLyCont, 100)

    # popII
    SEDLtabII_test = zeus21.SED_LyA(nu_in = nulisttest, pop = 2) 
    
    # popIII
    SEDLtabIII_test = zeus21.SED_LyA(nu_in = nulisttest, pop = 3) 
    
    normalization_LyASED_II = np.trapezoid(SEDLtabII_test,nulisttest)
    normalization_LyASED_III = np.trapezoid(SEDLtabIII_test,nulisttest)

    assert(normalization_LyASED_II == pytest.approx(1.0, 0.05) ) #5% is enough here
    assert(normalization_LyASED_III == pytest.approx(1.0, 0.05) ) #5% is enough here

    # test UV Green function (only popII)
    Mh = 1e12
    Green_UV_1 = zeus21.Greens_function_LUV(AstroParams = AstroParams, 
                                          ageMyrin= 1.,
                                          Mhalos = Mh)[0][0]
    Green_UV_100 = zeus21.Greens_function_LUV(AstroParams = AstroParams, 
                                          ageMyrin= 100.,
                                          Mhalos = Mh)[0][0]

    assert(Green_UV_1 == pytest.approx(1e36, 1e37)) # check unit is erg/s/Msun
    assert(Green_UV_100 == pytest.approx(Green_UV_1/1e2, Green_UV_1/1e1)) # check slope of the Green function 

    # test HA Green function (only popII)
    Green_Ha_1 = zeus21.Greens_function_LHa(AstroParams = AstroParams, 
                                        ageMyrin= 1.,
                                        Mhalos = Mh)[0][0]
    Green_Ha_100 = zeus21.Greens_function_LHa(AstroParams = AstroParams, 
                                        ageMyrin= 100.,
                                        Mhalos = Mh)[0][0]

    assert(Green_Ha_1 == pytest.approx(1e35, 1e36)) # check unit is erg/s/Msun
    assert(Green_Ha_100 == pytest.approx(Green_Ha_1/1e30, Green_Ha_1/1e20)) # check slope of the Green function 
