#==============================================================================#
# timescales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from .conversions import *

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
    mass_density = n2rho(number_density, mu_p)
    tff = np.sqrt( (3. * np.pi) / (32. * cds.G * mass_density) )

    print(outputunit)
    if outputunit=='yr':
        tff=tff.to(u.yr)
    elif outputunit=='s':
        tff=tff.to(u.s)
    else:
        raise ValueError('Please enter a valid ouput unit')

    return tff
