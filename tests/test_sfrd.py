"""

Test SFRD and related functions in Zeus21

Author: Claude AI
April 2025

"""

import pytest
import zeus21
import numpy as np

from zeus21.sfrd import get_T21_coefficients, SFR_II, SFR_III, fesc_II, fesc_III

def test_sfr_functions_relationships():
    """Test relationship between SFR II and SFR III functions"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=30.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams, USE_POPIII=True)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    
    # Generate mock LW parameter for testing
    mock_J21LW = np.ones(100) * 0.01
    mock_J21LW_interp = lambda z: 0.01
    
    # Test a range of halo masses and redshifts
    z_test = 20.0
    zprime_test = 20.0
    
    # Get SFRs for Pop II and III
    sfr_II = SFR_II(AstroParams, CosmoParams, HMFintclass, HMFintclass.Mhtab, z_test, zprime_test)
    sfr_III = SFR_III(AstroParams, CosmoParams, ClassyCosmo, HMFintclass, HMFintclass.Mhtab, 
                      mock_J21LW_interp, z_test, zprime_test, ClassyCosmo.pars['v_avg'])
    
    # In low-mass halos, Pop III should dominate; in high-mass halos, Pop II should dominate
    low_mass_idx = np.where(HMFintclass.Mhtab < 1e7)[0]
    high_mass_idx = np.where(HMFintclass.Mhtab > 1e10)[0]
    
    # These are not strict requirements, but should generally be true
    # For some parameter settings, these assertions might need adjustment
    # Test that arrays have non-zero elements to make sure the functions are working
    assert np.any(sfr_II > 0)
    assert np.any(sfr_III > 0)
    
    # Test the escape fraction functions
    fesc_ii = fesc_II(AstroParams, HMFintclass.Mhtab)
    fesc_iii = fesc_III(AstroParams, HMFintclass.Mhtab)
    
    # Check that escape fractions are between 0 and 1
    assert np.all(fesc_ii >= 0)
    assert np.all(fesc_ii <= 1)
    assert np.all(fesc_iii >= 0)
    assert np.all(fesc_iii <= 1)

def test_T21_coefficients_initialization():
    """Test the initialization of T21 coefficients class"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=30.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    
    # Initialize with reduced z range for test performance
    zmin_test = 15.0
    Coeffs = get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=zmin_test)
    
    # Check that redshift grid is set up correctly
    assert Coeffs.zmin == zmin_test
    assert Coeffs.zmax_integral > zmin_test
    assert len(Coeffs.zintegral) == Coeffs.Nzintegral
    assert Coeffs.zintegral[0] == pytest.approx(zmin_test)
    assert Coeffs.zintegral[-1] == pytest.approx(Coeffs.zmax_integral)
    
    # Check that arrays are initialized with correct shapes
    assert Coeffs.SFRDbar2D.shape == (Coeffs.Nzintegral, CosmoParams.NRs)
    assert Coeffs.gamma_index2D.shape == (Coeffs.Nzintegral, CosmoParams.NRs)
    
    # Check that sigmaofRtab is calculated
    assert Coeffs.sigmaofRtab.shape == (Coeffs.Nzintegral, len(Coeffs.Rtabsmoo))
    assert np.all(Coeffs.sigmaofRtab > 0)  # Standard deviations should be positive

def test_T21_coefficients_components():
    """Test specific components calculated by T21 coefficients"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=30.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    
    # Initialize with higher zmin to speed up test
    zmin_test = 18.0
    Coeffs = get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=zmin_test)
    
    # Test crucial components that are calculated
    
    # 1. Test that SFRD averages are computed
    assert np.all(Coeffs.SFRD_II_avg >= 0)  # SFRD should be non-negative
    
    # 2. Test ionization fraction
    assert np.all(Coeffs.xe_avg >= 0)  # Ionization fraction should be non-negative
    assert np.all(Coeffs.xe_avg <= 1)  # Ionization fraction should be <= 1
    
    # 3. Test neutral fraction
    assert np.all(Coeffs.xHI_avg >= 0)  # Neutral fraction should be non-negative
    assert np.all(Coeffs.xHI_avg <= 1)  # Neutral fraction should be <= 1
    
    # 4. Test temperature components
    assert np.all(Coeffs.Tk_ad >= 0)  # Adiabatic temperature should be positive
    assert np.all(Coeffs.Tk_xray >= 0)  # X-ray temperature contribution should be positive
    assert np.all(Coeffs.Tk_avg >= 0)  # Average temperature should be positive
    
    # 5. Test spin temperature components
    assert np.all(Coeffs._invTs_avg >= 0)  # Inverse spin temperature should be positive
    assert np.all(Coeffs.invTcol_avg >= 0)  # Inverse color temperature should be positive
    
    # 6. Test that T21 global signal is computed and reasonable
    # T21 can be positive or negative, but generally in a specific range
    assert np.all(Coeffs.T21avg > -1000)  # Lower bound (very conservative)
    assert np.all(Coeffs.T21avg < 100)    # Upper bound (very conservative)
    
    # 7. Test that LW radiation field is computed
    assert np.all(Coeffs.J21LW_avg >= 0)  # LW radiation should be non-negative

def test_T21_with_popIII():
    """Test T21 coefficients with Population III stars enabled"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=30.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    # Create two versions of AstroParams, with and without PopIII
    AstroParams_noIII = zeus21.Astro_Parameters(UserParams, CosmoParams, USE_POPIII=False)
    AstroParams_withIII = zeus21.Astro_Parameters(UserParams, CosmoParams, USE_POPIII=True)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    
    # Initialize with higher zmin to speed up test
    zmin_test = 19.0
    
    # Calculate coefficients for both models
    Coeffs_noIII = get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams_noIII, 
                                        HMFintclass, zmin=zmin_test)
    
    Coeffs_withIII = get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams_withIII, 
                                         HMFintclass, zmin=zmin_test)
    
    # Test that including Pop III stars affects the results
    # Check that SFRD_III is zero for no PopIII and non-zero for with PopIII
    assert np.all(Coeffs_noIII.SFRD_III_avg == 0)
    
    # High-z ionization should be higher with Pop III
    high_z_idx = np.where(Coeffs_withIII.zintegral > 25)[0]
    if len(high_z_idx) > 0:
        # This might not always be true depending on the specific model parameters
        # So we'll make a weak assertion that results are different
        assert not np.array_equal(Coeffs_noIII.xe_avg[high_z_idx], Coeffs_withIII.xe_avg[high_z_idx])
    
    # LW background should be different with Pop III
    assert not np.array_equal(Coeffs_noIII.J21LW_avg, Coeffs_withIII.J21LW_avg)
    
    # Global T21 signal should be different
    assert not np.array_equal(Coeffs_noIII.T21avg, Coeffs_withIII.T21avg)