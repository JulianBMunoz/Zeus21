"""

Make maps! For fun and science

Author: Julian B. Muñoz
UT Austin - August 2024

"""

from . import cosmology
from . import constants

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from pyfftw import empty_aligned as empty


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

            # self.DeltasqdT21 = T21_coefficients.T21avg * ( Power_Spectrum._betaxa * Power_Spectrum.Deltasq_dxa_lin.T + Power_Spectrum._betaT * Power_Spectrum.Deltasq_dTx_lin.T+ Power_Spectrum._betad * Power_Spectrum.Deltasq_d_lin.T )
            # PdT21 = self.DeltasqdT21[:,_iz]/k3over2pi2

            powerratioint = interp1d(klist,PdT21/Pd,fill_value=0.0,bounds_error=False)


            deltak = pb.delta_k()

            powerratio = powerratioint(pb.k())
            T21lin_k = powerratio * deltak
            self.T21maplin= self.T21global + powerboxCtoR(pb,mapkin = T21lin_k)

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




def powerboxCtoR(pbobject,mapkin = None):
    'Function to convert a complex field to real 3D (eg density, T21...) on the powerbox notation'
    'Takes a powerbox object pbobject, and a map in k space (mapkin), or otherwise assumes its pbobject.delta_k() (tho in that case it should be delta_x() so...'

    realmap = empty((pbobject.N,) * pbobject.dim, dtype='complex128')
    if (mapkin is None):
        realmap[...] = pbobject.delta_k()
    else:
        realmap[...] = mapkin
    realmap[...] = pbobject.V * pbox.dft.ifft(realmap, L=pbobject.boxlength, a=pbobject.fourier_a, b=pbobject.fourier_b)[0]
    realmap = np.real(realmap)

    return realmap