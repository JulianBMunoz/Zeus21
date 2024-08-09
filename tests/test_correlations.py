"""

Correlations tests for Zeus21

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import pytest
import zeus21
import numpy as np

from zeus21.correlations import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence annyoing warning in mcfit


CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 10., zmax_CLASS = 10.) #to speed up
ClassyCosmo = zeus21.runclass(CosmoParams_input)
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo)

CorrFClass = zeus21.Correlations(CosmoParams, ClassyCosmo)


def test_corrfuncs():

    assert(CorrFClass.xi_RR_CF[0][0][1] >= CorrFClass.xi_RR_CF[1][1][1]) #make sure smoothing goes the right direction
    assert(CorrFClass.xiEta_RR_CF[0][0][1] >= CorrFClass.xiEta_RR_CF[1][1][1]) #make sure smoothing goes the right direction

    #windows
    ktestwin = 1e-4
    Rtestwin = 1.0
    assert(CorrFClass._WinG(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(CorrFClass._WinTH(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(CorrFClass._WinTH1D(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))

    ktestwin = 3.
    assert(CorrFClass._WinG(ktestwin,Rtestwin) < 1.0)
    assert(CorrFClass._WinTH(ktestwin,Rtestwin) < 1.0)
    assert(CorrFClass._WinTH1D(ktestwin,Rtestwin) < 1.0)
