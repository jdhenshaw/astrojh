#==============================================================================#
# feedback.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import constants as const

def Nly(integrated_flux, frequency, electron_temperature, distance):
    """
    Estimates the total number of Lyman continuum emitting photons. Equation
    from Mezger & Henderson 1967 under the assumptions of Rubin 1968.

    Parameters
    ----------
    integreated_flux : number
        integrated radio continuum flux in units of Jy
    frequency : number
        frequency in units of GHz
    electron_temperature : number
        electron temperature in units of K
    distance : number
        source distance in units of pc
    """
    integrated_flux=integrated_flux * (ap.Jy )
    frequency=frequency*1.e9*u.Hz
    electron_temperature=electron_temperature*u.K
    distance=distance*u.pc

    Nly = 8.9e40 * (integrated_flux / (1*ap.Jy)) * \
                   (frequency / (1.e9*u.Hz))**0.1 * \
                   (electron_temperature / (1.e4*u.K))**-0.45 * \
                   (distance / (1.*u.pc))**2

    Nly = Nly * u.s**-1
    return Nly

def n_e(integrated_flux, frequency, electron_temperature, distance, angular_size):
    """
    Estimates the electron density in a HII region

    Parameters
    ----------
    integreated_flux : number
        integrated radio continuum flux in units of Jy
    frequency : number
        frequency in units of GHz
    electron_temperature : number
        electron temperature in units of K
    distance : number
        source distance in units of pc
    angular_size : number
        angular size of the observed HII region in units of arcseconds
    """
    integrated_flux=integrated_flux * (ap.Jy )
    frequency=frequency*1.e9*u.Hz
    electron_temperature=electron_temperature*u.K
    distance=distance*u.pc
    angular_size=angular_size*u.arcsec

    n_e = 2.30e6 * (integrated_flux / (1*ap.Jy))**0.5 * \
                   (frequency / (1.e9*u.Hz))**0.05 * \
                   (electron_temperature / (u.K))**0.175 * \
                   (distance / (1.*u.pc))**-0.5 * \
                   (angular_size / (1.*u.arcsec))**-1.5

    n_e = n_e * u.cm**-3
    return n_e

def t_rec(n_e, alpha=2.6e-13):
    """
    Estimates the recombination time of ionised gas in an HII region assuming an
    electron density and type B recombination rate coefficient.

    Parameters
    ----------
    n_e : number
        electron density in units of electrons per cc
    alpha (optional): number
        Recombination coefficient in units of cm^3 s^-1. If not supplied a
        value of 2.6x10^-13 will be assumed - relevant for an electron
        temperature of 10000K. See Draine 2011 table 14.1 for more.
    """
    n_e = n_e * u.cm**-3
    alpha = alpha * u.cm**3 * u.s**-1

    t_rec = 1./(n_e*alpha)

    t_rec = t_rec.to(u.yr)

    return t_rec

def r_strom(Nly, n_e, alpha=2.6e-13):
    """
    Estimates the Strömgren radius of an HII region

    Parameters
    ----------
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    n_e : number
        electron density in units of electrons per cc
    alpha (optional): number
        Recombination coefficient in units of cm^3 s^-1. If not supplied a
        value of 2.6x10^-13 will be assumed - relevant for an electron
        temperature of 10000K. See Draine 2011 table 14.1 for more.
    """
    Nly = Nly * u.s**-1
    n_e = n_e * u.cm**-3
    alpha = alpha * u.cm**3 * u.s**-1
    r_strom = ((Nly*3.) / (4.*np.pi*alpha*n_e**2))**(1./3.)
    r_strom = r_strom.to(u.pc)

    return r_strom

def t_dyn(r_strom, radius, temperature, mu_p=0.64):
    """
    estimates the dynamical time of an HII region. Equation from Dyson &
    Williams

    Parameters
    ----------
    r_strom : number
        The Strömgren radius of the HII region in units of pc
    radius : number
        The observed radius of the HII region in units of pc
    temperature : number
        Temperature of ionised gas in the HII region
    mu_p (optional): number
        The mean particle mass to compute the sound speed. Assumed 0.64 relevant
        for 90% H and 10% He.
    """
    from .kinematics import cs
    r_strom = r_strom * u.pc
    radius = radius * u.pc

    t_dyn = ((4.*r_strom)/(7.*cs(temperature, mu_p=mu_p)))*((radius/r_strom)**(7./4.) - 1)
    t_dyn = t_dyn.to(u.yr)

    return t_dyn

def momentum(density, Nly, time, temperature):
    """
    estimates the momentum of a shell driven by a HII region. Equation from
    Krumholz 2017 - equation 7.36

    Parameters
    ----------
    density : number
        density of the ambient medium within which the shell is expanding into.
        units of particles per cc
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    time : number
        age of the HII region in yrs
    temperature : number
        Temperature of ionised gas in the HII region in units of K

    """
    density = density * u.cm**-3
    Nly = Nly * u.s**-1
    time = time * u.yr / 1.e6
    temperature = temperature * u.K / 1.e4

    P = 1.5e5 * (density/(100.* u.cm**-3))**(-1./7.) * \
                (Nly/(1e49 * u.s**-1))**(4./7.) * \
                (time/( 1.* u.yr))**(9./7.) * \
                (temperature /(1. * u.K))**(-8./7.)

    P = P * ap.solMass * u.km * u.s**-1

    return P

def energy(density, Nly, time, temperature):
    """
    Estimates the energy in an expanding shell. Equation from
    Krumholz 2017 - equation 7.35

    Parameters
    ----------
    density : number
        density of the ambient medium within which the shell is expanding into.
        units of particles per cc
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    time : number
        age of the HII region in yrs
    temperature : number
        Temperature of ionised gas in the HII region in units of K

    """
    density = density * u.cm**-3
    Nly = Nly * u.s**-1
    time = time * u.yr / 1.e6
    temperature = temperature * u.K / 1.e4

    energy = 8.1e47 * (density/(100.* u.cm**-3))**(-10./7.) * \
                (Nly/(1e49 * u.s**-1))**(5./7.) * \
                (time/(1. * u.yr))**(6./7.) * \
                (temperature /(1. * u.K))**(10./7.)

    energy = energy * u.erg

    return energy
