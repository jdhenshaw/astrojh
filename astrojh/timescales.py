#==============================================================================#
# timescales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy import constants as const

def tff( number_density, mu_p=2.8, outputunit='yr' ):
    """
    Accepts a number density in units of particles per cubic centimetre and
    derives the free fall time in yrs

    Parameters
    ----------
    number_density : float
        number density in particles per cubic centimetre
    mu_p : float (optional)
        Molecular mass. If none given mu=2.8 (molecular hydrogen) is used
    outputunit : string (optional)
        Choose the output unit (default = yr)
    """
    mass_density = ntorho(number_density, mu_p)
    tff = np.sqrt( (3. * np.pi) / (32. * const.G * mass_density) )

    if outputunit=='yr':
        tff=tff.to(u.yr)
    elif outputunit=='s':
        tff=tff.to(u.s)
    else:
        raise ValueError('Please enter a valid ouput unit')

    return tff

def tcross( size, sigma ):
    """
    Accepts size in pc and 3D velocity dispersion in km/s and computes the
    turbulent crossing time. Returns in yr

    Parameters
    ----------
    size : float
        The size of the source (pc)
    sigma : float
        velocity dispersion (km/s) assumes already in 3D i.e. sqrt(3)*sigma1D

    """
    size=size*u.pc
    sigma=sigma*(u.km/u.s)

    tcross=size/sigma
    tcross=tcross.to(u.yr)
    return tcross

def tdyn( size, vel ):
    """
    Accepts size in pc and velocity in km/s and computes the dynamical age of an
    object (e.g. HII region or expanding shell). Returns in yr

    Parameters
    ----------
    size : float
        The size of the source (pc)
    vel : float
        velocity dispersion (km/s) assumes already in 3D i.e. sqrt(3)*sigma1D

    """
    size=size*u.pc
    vel=vel*(u.km/u.s)

    tdyn=size/vel
    tdyn=tcross.to(u.yr)
    return tdyn
