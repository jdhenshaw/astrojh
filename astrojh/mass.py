#==============================================================================#
# mass.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from .conversions import *
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import constants as const

def mass( integrated_flux, wave, t, kappa, distance ):
    """
    Accepts integrated flux in Jy, wavelength in m, temperature in K, dust
    opacity per unit mass in cm^2/g, and the distance to the object in pc,
    and computes the mass

    Parameters
    ----------
    integrated_Flux : float
        integrated flux in Jy.
    wave : float
        wavelength (m)
    t: float (optional)
        temperature (K)
    kappa: float
        dust opacity per unit mass (cm^2g^-1)
    distance: float
        distance to the object (pc)
    """
    integrated_flux=integrated_flux * (ap.Jy )
    wave=wave*u.m
    t=t*u.K
    kappa=kappa*(u.cm**2/u.g)
    distance=distance*(u.pc)

    B = blackbody_nu(wave.to(u.Hz, equivalencies=u.spectral()), t)
    B = B * u.sr

    m = (distance**2. * integrated_flux) / ( kappa * B)
    m = m.to(ap.solMass)
    return m

def mass_k08( integrated_flux, wave, t, kappa, distance  ):
    """
    Accepts integrated flux in Jy, wavelength in m, temperature in K, dust
    opacity per unit mass in cm^2/g, and the distance to the object in pc,
    and computes the mass based on the equation in Kauffmann et al. 2008

    Parameters
    ----------
    integrated_Flux : float
        integrated flux in Jy.
    wave : float
        wavelength (m)
    t: float (optional)
        temperature (K)
    kappa: float
        dust opacity per unit mass (cm^2g^-1)
    distance: float
        distance to the object (pc)
    """
    integrated_flux=integrated_flux * ( ap.Jy )
    wave=wave*u.m
    t=t*u.K
    kappa=kappa*(u.cm**2/u.g)
    distance=distance*(u.pc)
    
    m = 0.12 * ( np.exp( 1.439 * ( wave/(1e-3 * u.m) )**-1.0 *
               ( t/(10.0 * u.K) )**-1.0 ) - 1.0 ) * \
                    (( kappa/ (0.01 * (u.cm**2/u.g)) )**-1.0) * \
                    ( integrated_flux/ (1.0 * u.Jy) ) * \
                    (( distance/ (100.0 * u.pc) )**2.0) * \
                    (( wave/ (1e-3*u.m) )**3.0) * ap.solMass
    return m
