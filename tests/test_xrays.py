"""

Xray tests for Zeus21

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

"""

import pytest
import zeus21
import numpy as np

from zeus21.xrays import *

CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 10., zmax_CLASS = 10.) #to speed up
ClassyCosmo = zeus21.runclass(CosmoParams_input)
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo)
AstroParams = zeus21.Astro_Parameters(CosmoParams)

Xray_Class = Xray_class(CosmoParams) #initialize Xray class
Energylist = AstroParams.Energylist

def test_xrays():

    z1=10.;
    z2=15.;
    tau1 = Xray_Class.optical_depth(CosmoParams, Energylist,z1,z1)
    assert( (tau1 == np.zeros_like(tau1) ).all())

    tau2 = Xray_Class.optical_depth(CosmoParams, Energylist,z1,z2)
    assert( (tau2 >= np.zeros_like(tau2) ).all())

    opacity1 = Xray_Class.opacity_Xray(CosmoParams, Energylist,z1,z2)
    assert( (np.zeros_like(opacity1) <= opacity1).all())
    assert( (opacity1<= np.ones_like(opacity1) ).all())


    assert( (np.zeros_like(Energylist) <= sigma_HI(Energylist)).all())
    assert( (np.zeros_like(Energylist) <= sigma_HeI(Energylist)).all())
