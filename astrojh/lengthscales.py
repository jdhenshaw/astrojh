#==============================================================================#
# lengthscales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from .conversions import ntorho, Ntomsd
from .kinematics import cs
from astropy import constants as const

def jeans_length( number_density, t, mu_p=2.8 ):
    """
    Accepts a number density in units of particles per cubic centimetre, the
    temperature in kelvin, and the mean particle mass and derives the jeans
    length

    Parameters
    ----------
    number_density : float
        number density in particles per cubic centimetre
    t : float
        temperature (K)
    mu_p : float (optional)
        Molecular mass (default = 2.8)

    """
    mass_density = ntorho(number_density, mu_p)
    l = cs( t, 2.33 ) * (np.pi  /  ( const.G * mass_density ) )**0.5
    l = l.to(u.pc)
    return l

def jeans_length_nt( sig, number_density, mu_p=2.8 ):
    """
    Accepts a velocity dispersion in km/s, number density in units of particles
    per cubic centimetre and the mean particle mass and derives the jeans length
    accounting for non-thermal motions

    Parameters
    ----------
    sig : float
        velocity dispersion (km/s)
    number_density : float
        number density in particles per cubic centimetre
    mu_p : float (optional)
        Molecular mass (default = 2.8)

    """
    sig = sig * (u.km/u.s)
    mass_density = ntorho(number_density, mu_p)
    l = sig * (np.pi  /  ( const.G * mass_density ) )**0.5
    l = l.to(u.pc)
    return l

def scale_height_cylinder( number_density, t, mu_p=2.8, sigma=None ):
    """
    Accepts a number density in units of particles per cubic centimetre, the
    temperature in kelvin, and the mean particle mass and derives the thermal
    scale height of a cylinder

    Parameters
    ----------
    number_density : float
        number density in particles per cubic centimetre
    t : float
        temperature (K)
    mu_p : float (optional)
        Molecular mass (default = 2.8)

    """
    from .kinematics import cs
    mass_density = ntorho(number_density, mu_p)
    if sigma is None:
        H = cs( t, 2.33 ) / ( 4. * np.pi * const.G * mass_density )**0.5
    else:
        H = sigma / ( 4. * np.pi * const.G * mass_density )**0.5
    H = H.to(u.pc)
    return H

def jeans_cylinder( number_density, t, mu_p=2.8 ):
    """
    Accepts a number density in units of particles per cubic centimetre, the
    temperature in kelvin, and the mean particle mass and derives the jeans
    type instability lengthscale for an isothermal cylinder. Eq. from
    Nagasawa 1987

    Parameters
    ----------
    number_density : float
        number density in particles per cubic centimetre
    t : float
        temperature (K)
    mu_p : float (optional)
        Molecular mass (default = 2.8)

    """
    from .lengthscales import scale_height_cylinder
    H = scale_height_cylinder(number_density, t, mu_p)
    l = 22. * H
    l = l.to(u.pc)
    return l

def jeans_cylinder_nt( sig, number_density, mu_p=2.8 ):
    """
    Accepts a number density in units of particles per cubic centimetre, the
    temperature in kelvin, and the mean particle mass and derives the jeans
    type instability lengthscale for an isothermal cylinder accounting for
    non-thermal motions

    Parameters
    ----------
    sig : float
        velocity dispersion (km/s)
    number_density : float
        number density in particles per cubic centimetre
    mu_p : float (optional)
        Molecular mass (default = 2.8)

    """
    from .lengthscales import scale_height_cylinder
    sig = sig * (u.km/u.s)
    H = scale_height_cylinder(number_density, 0.0, mu_p, sigma=sig)
    l = 22. * H
    l = l.to(u.pc)
    return l

def toomre_length(column_density, epicyclic_frequency, mu=2.8):
    """
    Accepts a column density in units of particles per square centimetre and the
    epicyclic frequency in units of Myr**-1 and computes the Toomre length

    Parameters
    ----------
    column_density : float
        column density (1/cm^2)
    epicyclic_frequency : float
        epicyclic frequency (1/Myr)

    """
    msd = Ntomsd(column_density, mu)
    k = k * (1./(1.e6*u.yr))
    toomre = ( 4. * np.pi**2. * const.G * msd ) / epicyclic_frequency**2.
    toomre = toomre.to(u.pc)
    return toomre

def sonic_scale( sig, number_density, t, size, B, mol, mu=2.8, mu_p=2.33 ):
    """
    Accepts the observed 1D velocity dispersion in km/s, the size of an object
    in pc, the temperature in K, the magnetic field strength in Gauss, the
    number density in particles per cubic cm, and the mass of the molecular mass
    and computes the sonic scale - the scale at which turbulent gas motions are
    expected to transitionn from super- to subsonic. Returns in pc

    Parameters
    ----------
    sig : float
        The observed 1D velocity dispersion (km/s)
    number_density : float
        The number density (cm^-3) - default mu assumes given in molecular H2
    t : float
        The temperature (K)
    size : float
        Size of the object (pc)
    B : float
        Magnetic field strength (G)
    mol : float
        molecular weight
    mu : float (optional)
        molecular weight used for number_density --> mass_density conv.
        (default = 2.8; i.e. assumes density is given as density of hydrogen
        molecules)
    mu_p : float (optional)
        molecular weight of mean particle (default=2.33)

    """
    from .kinematics import plasma_beta
    from .kinematics import mach_3d
    size = size*u.pc
    beta = plasma_beta( t, B, number_density, mu, mu_p)
    mach3d = mach_3d( sig, t, mu, mu_p )
    l =( size * (1. + beta**-1.) )/ mach3d**2
    l = l.to(u.pc)
    return l
