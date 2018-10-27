#==============================================================================#
# conversions.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap

def N2msd(column_density, mu_p=2.8):
    """
    Accepts a column density in units of particles per square centimetre and the
    mean molecular mass and converts this to a mass surface density

    Parameters
    ----------
    N : float
        column density (1/cm^2)
    mu_p : float (optional)
        mean molecular weight (default=2.8)
    """
    column_density = column_density*u.cm**-2
    msd = mu_p * ap.M_p * n
    msd = msd.to(ap.solMass / u.pc**2)
    return msd

def msd2N(msd, mu_p=2.8):
    """
    Accepts a mass surface density in units of solar masses per square pc and
    the mean molecular mass and converts this to a column density

    Parameters
    ----------
    msd : float
        mass surface density (1/cm^2)
    mu_p : float (optional)
        mean molecular weight (default=2.8)
    """
    msd = msd * (ap.solMass / u.pc**2)
    column_density = msd / (mu_p * ap.M_p)
    column_density = column_density.to(1./u.cm**2.)
    return column_density

def n2rho(number_density, mu_p=2.8):
    """
    Accepts a number density in units of particles per cubic centimetre and the
    mean molecular mass and converts this to a mass density

    Parameters
    ----------
    number_density : float
        number density (1/cm^3)
    mu_p : float (optional)
        mean molecular weight (default=2.8)
    """
    number_density = number_density*u.cm**-3
    mass_density = mu_p * ap.M_p * number_density
    mass_density = mass_density.to(u.kg / u.m**3)
    return mass_density

def rho2n(mass_density, mu_p=2.8):
    """
    Accepts a volume density in units of particles per cubic centimetre and the
    mean molecular mass and converts this to a number density

    Parameters
    ----------
    mass_density : float
        mass density (kg/cm^3)
    mu_p : float (optional)
        mean molecular weight (default=2.8)
    """
    mass_density = mass_density * (u.kg / u.m**3)
    number_density = mass_density / mu_p * ap.M_p
    number_density = number_density.to(u.cm**-3)
    return number_density
