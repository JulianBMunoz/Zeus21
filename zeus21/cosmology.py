"""

Cosmology helper functions and other tools

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

"""

import jax.numpy as np
from classy import Class
from jax.scipy.interpolate import RegularGridInterpolator

from . import constants


def runclass(CosmologyIn):
    "Set up CLASS cosmology. Takes CosmologyIn class input and returns CLASS Cosmology object"
    ClassCosmo = Class()
    ClassCosmo.set({"omega_b": CosmologyIn.omegab, "omega_cdm": CosmologyIn.omegac, "h": CosmologyIn.h_fid, "A_s": CosmologyIn.As, "n_s": CosmologyIn.ns, "tau_reio": CosmologyIn.tau_fid})
    ClassCosmo.set({"output": "mPk", "lensing": "no", "P_k_max_1/Mpc": CosmologyIn.kmax_CLASS, "z_max_pk": CosmologyIn.zmax_CLASS})
    # hfid = ClassCosmo.h() # get reduced Hubble for conversions to 1/Mpc

    # and run it (see warmup for their doc)
    ClassCosmo.compute()

    return ClassCosmo


# def Hub(Cosmo_Parameters, z):
#     # Hubble(z) in km/s/Mpc -- UNUSED
#     return Cosmo_Parameters.h_fid * 100 * np.sqrt(Cosmo_Parameters.OmegaM * pow(1 + z, 3.0) + Cosmo_Parameters.OmegaR * pow(1 + z, 4.0) + Cosmo_Parameters.OmegaL)


def HubinvMpc(Cosmo_Parameters, z):
    # H(z) in 1/Mpc - directly from Boltzmann code to stay authoratitive
    #return Hub(Cosmo_Parameters, z) / constants.c_kms
    return Cosmo_Parameters.Hofzint(z)


def Hubinvyr(Cosmo_Parameters, z):
    # H(z) in 1/yr, just different units
    #return Hub(Cosmo_Parameters, z) * constants.KmToMpc * constants.yrTos
    return HubinvMpc(Cosmo_Parameters, z)* constants.c_kms* constants.KmToMpc * constants.yrTos


def rho_baryon(Cosmo_Parameters, z):
    # \rho_baryon in Msun/Mpc^3 as a function of z
    return Cosmo_Parameters.OmegaB * Cosmo_Parameters.rhocrit * pow(1 + z, 3.0)


def n_baryon(Cosmo_Parameters, z):
    # density of baryons in 1/cm^3
    return rho_baryon(Cosmo_Parameters, z) / Cosmo_Parameters.mu_baryon_Msun / (constants.Mpctocm**3.0)


def Tcmb(Cosmo_Parameters, z):
    return Cosmo_Parameters.T_CMB0 * (1 + z)


def Tadiabatic(CosmoParams, z):
    "Returns T_adiabatic as a function of z from thermodynamics in CLASS"
    return CosmoParams.Tadiabaticint(z)


def xefid(CosmoParams, z):
    "Returns fiducial x_e(z) w/o any sources. Uses thermodynamics in CLASS for z>15, and fixed below to avoid the tanh approx."
    _zcutCLASSxe = 15.0
    _xecutCLASSxe = CosmoParams.xetanhint(_zcutCLASSxe)
    return CosmoParams.xetanhint(z) * np.heaviside(z - _zcutCLASSxe, 0.5) + _xecutCLASSxe * np.heaviside(_zcutCLASSxe - z, 0.5)


def adiabatic_index(z):
    "Returns adiabatic index (delta_Tad/delta) as a function of z. Fit from 1506.04152. to ~3% on z = 6 − 50)."
    return 0.58 - 0.005 * (z - 10.0)


def MhofRad(Cosmo_Parameters, R):
    # convert input Radius in Mpc comoving to Mass in Msun
    return Cosmo_Parameters.constRM * pow(R, 3.0)


def RadofMh(Cosmo_Parameters, M):
    # convert input M halo in Msun radius in cMpc
    return pow(M / Cosmo_Parameters.constRM, 1 / 3.0)


def ST_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM):
    A_ST = Cosmo_Parameters.Amp_ST
    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST

    nutilde = np.sqrt(a_ST) * delta_crit_ST / sigmaM

    return -A_ST * np.sqrt(2.0 / np.pi) * nutilde * (1.0 + nutilde ** (-2.0 * p_ST)) * np.exp(-(nutilde**2) / 2.0) * (Cosmo_Parameters.rho_M0 / (Mass * sigmaM)) * dsigmadM


def PS_HMF_unnorm(Cosmo_Parameters, Mass, nu, dlogSdM):
    "Returns the Press-Schechter HMF (unnormalized since we will take ratios), given a halo Mass [Msun], nu = delta_tilde/S_tilde, with delta_tilde = delta_crit - delta_R, and variance S = sigma(M)^2 - sigma(R)^2. Used for 21cmFAST mode."

    return nu * np.exp(-Cosmo_Parameters.a_corr_EPS * nu**2 / 2.0) * dlogSdM * (1.0 / Mass)
    # written so that dsigmasq/dM appears directly, since that is not modified by EPS, whereas sigma_tot^2 = sigma^2(M) - sigma^2(R). The sigma in denominator will be sigma_tot


class HMF_interpolator:
    "Class that builds an interpolator of the HMF. Returns an interpolator"

    def __init__(self, Cosmo_Parameters, ClassCosmo):
        self._Mhmin = 1e5
        self._Mhmax = 1e14
        self._NMhs = np.floor(35 * constants.precisionboost).astype(int)
        self.Mhtab = np.logspace(np.log10(self._Mhmin), np.log10(self._Mhmax), self._NMhs)  # Halo mases in Msun
        self.RMhtab = RadofMh(Cosmo_Parameters, self.Mhtab)

        self.logtabMh = np.log(self.Mhtab)

        self._zmin = Cosmo_Parameters.zmin_CLASS
        self._zmax = Cosmo_Parameters.zmax_CLASS
        self._Nzs = np.floor(100 * constants.precisionboost).astype(int)
        self.zHMFtab = np.linspace(self._zmin, self._zmax, self._Nzs)

        # check resolution
        if Cosmo_Parameters.kmax_CLASS < 1.0 / self.RMhtab[0]:
            print("Warning! kmax_CLASS may be too small! Run CLASS with higher kmax")

        self.sigmaMhtab = np.array([[ClassCosmo.sigma(RR, zz) for zz in self.zHMFtab] for RR in self.RMhtab])

        self._depsM = 0.01  # for derivatives, relative to M
        self.dsigmadMMhtab = np.array([[(ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM * (1 + self._depsM)), zz) - ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM * (1 - self._depsM)), zz)) / (MM * 2.0 * self._depsM) for zz in self.zHMFtab] for MM in self.Mhtab])

        if Cosmo_Parameters.Flag_emulate_21cmfast == True:
            # ADJUST BY HAND adjust sigmas to match theirs, since the CLASS TF they use is at a fixed cosmology from 21cmvFAST but the input cosmology is different
            self.sigmaMhtab *= np.sqrt(0.975)
            self.dsigmadMMhtab *= np.sqrt(0.975)

            # this correction is because 21cmFAST uses the dicke() function to compute growth, which is ~0.5% offset at high z. This offset makes our growth the same as dicke() for a Planck2018 cosmology. Has to be added separately to the growth(z) correction above since they come in different places
            _offsetgrowthdicke21cmFAST = 1 - 0.000248 * (self.zHMFtab - 5.0)
            self.sigmaMhtab *= _offsetgrowthdicke21cmFAST
            self.dsigmadMMhtab *= _offsetgrowthdicke21cmFAST
            # Note that these two changes may be different if away from Planck2018

        self.HMFtab = ST_HMF(Cosmo_Parameters, self.Mhtab[:, None], self.sigmaMhtab, self.dsigmadMMhtab).clip(min=np.exp(-300.0))

        self.fitMztab = [np.log(self.Mhtab), self.zHMFtab]

        self.logHMFint = RegularGridInterpolator(self.fitMztab, np.log(self.HMFtab))
        self.sigmaintlog = RegularGridInterpolator(self.fitMztab, self.sigmaMhtab)  # no need to log since it doesnt vary dramatically
        self.dsigmadMintlog = RegularGridInterpolator(self.fitMztab, self.dsigmadMMhtab)

        # also build an interpolator for sigma(R) of the R we integrate over (for CD and EoR). These R >> Rhalo typically, so need new table.
        self.sigmaofRtab = np.array([[ClassCosmo.sigma(RR, zz) for zz in self.zHMFtab] for RR in Cosmo_Parameters._Rtabsmoo])
        self.fitRztab = [np.log(Cosmo_Parameters._Rtabsmoo), self.zHMFtab]
        self.sigmaRintlog = RegularGridInterpolator(self.fitRztab, self.sigmaofRtab)  # no need to log either

    def HMF_int(self, Mh, z):
        "Interpolator to find HMF(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)

        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM, z] for LM in logMhvec])

        return np.exp(self.logHMFint(inarray))

    def sigma_int(self, Mh, z):
        "Interpolator to find sigma(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM, z] for LM in logMhvec])
        return self.sigmaintlog(inarray)

    def sigmaR_int(self, RR, z):
        "Interpolator to find sigma(RR,z), designed to take a single z but an array of RR in cMpc"
        _logRR = np.log(RR)
        # logRRvec = np.asarray([_logRR]) if np.isscalar(_logRR) else np.asarray(_logRR)
        # inarray = np.array([[LR, z] for LR in logRRvec])
        return self.sigmaRintlog(np.array([_logRR, z]))

    def dsigmadM_int(self, Mh, z):
        "Interpolator to find dsigma/dM(M,z), designed to take a single z but an array of Mh in Msun. Used in 21cmFAST mode"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM, z] for LM in logMhvec])
        return self.dsigmadMintlog(inarray)


def growth(Cosmo_Parameters, z):
    "Scale-independent growth factor, interpolated from CLASS"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    if Cosmo_Parameters.Flag_emulate_21cmfast == True:
        _offsetgrowthdicke21cmFAST = 1 - 0.000248 * (zlist - 5.0)  # as in HMF, to fix growth. have to do it independently since it depends on z.
        return Cosmo_Parameters.growthint(zlist) * _offsetgrowthdicke21cmFAST
    else:
        return Cosmo_Parameters.growthint(zlist)


def dgrowth_dz(CosmoParams, z):
    "Derivative of growth factor growth() w.r.t. z"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    dzlist = zlist * 0.001
    return (growth(CosmoParams, z + dzlist) - growth(CosmoParams, z - dzlist)) / (2.0 * dzlist)


def redshift_of_chi(CosmoParams, z):
    "Returns z(chi) for any input comoving distance from today chi in Mpc"
    return CosmoParams.zfofRint(z)


def T021(Cosmo_Parameters, z):
    "Prefactor in mK to T21 that only depends on cosmological parameters and z. Eg Eq.(21) in 2110.13919"
    return 34 * pow((1 + z) / 16.0, 0.5) * (Cosmo_Parameters.omegab / 0.022) * pow(Cosmo_Parameters.omegam / 0.14, -0.5)


# UNUSED:
# def interp2Dlinear_only_y(arrayxy, arrayz, x, y):
#     "2D interpolator where the x axis is assumed to be an array identical to the trained x. That is, an array of 1D linear interpolators. arrayxy is [x,y]. arrayz is result. x is the x input (=arrayxy[0]),  and y the y input. Returns z result (array)"
#     if((x != arrayxy[0]).all()):
#         print('ERROR on interp2Dlinear_only_y, x need be the same in interp and input')
#         return -1
#     Ny = len(arrayxy[1])
#     ymin, ymax = arrayxy[1][[0,-1]]
#     if((y > ymax or y<ymin).all()):
#         print('EXTRAPOLATION on interp2Dlinear_only_y on y axis. max={} curr={}'.format(ymax, y))
#
#     ystep = (ymax-ymin)/(Ny-1.)
#     ysign = np.sign(ystep).astype(int)
#
#     iy = np.floor((y-ymin)/ystep).astype(int)
#     iy=np.fmin(iy,Ny-1-(ysign+1)//2) #if positive ysign stop 1 lower
#     iy=np.fmax(iy,0-(ysign-1)//2) #if negative ysign stop 1 higher
#
#     y1 = ymin + iy * ystep
#     y2 = y1 + ystep
#
#     fy1 = arrayz[:,iy]
#     fy2 = arrayz[:,iy+ysign]
#
#     fy = fy1 + (fy2-fy1)/ystep * (y - y1)
#     return fy
