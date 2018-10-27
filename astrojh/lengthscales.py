#==============================================================================#
# lengthscales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from .conversions import *
from kinematics import cs

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
    mass_density = n2rho(number_density, mu_p)
    l = cs( t, 2.33 ) * (np.pi  /  ( cds.G * mass_density ) )**0.5
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
    mass_density = n2rho(number_density, mu_p)
    l = sig * (np.pi  /  ( cds.G * mass_density ) )**0.5
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
    mass_density = n2rho(number_density, mu_p)
    if sigma is None:
        H = cs( t, 2.33 ) / ( 4. * np.pi * cds.G * mass_density )**0.5
    else:
        H = sigma / ( 4. * np.pi * cds.G * mass_density )**0.5
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
    msd = N2msd(column_density, mu)
    k = k * (1./(1.e6*u.yr))
    toomre = ( 4. * np.pi**2. * cds.G * msd ) / epicyclic_frequency**2.
    toomre = toomre.to(u.pc)
    return toomre
