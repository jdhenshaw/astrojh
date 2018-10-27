#==============================================================================#
# timescales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds

# Global parameters

mh = 1.6737236e-27 * u.kg # Mass of a hydrogen atom in kg

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
    number_density = number_density * u.cm**-3
    mass_density = mu_p * mh * number_density
    mass_density.to(u.kg / u.m**3)
    tff = np.sqrt( (3. * np.pi) / (32. * cds.G * mass_density) )

    print(outputunit)
    if outputunit=='yr':
        tff=tff.to(u.yr)
    elif outputunit=='s':
        tff=tff.to(u.s)
    else:
        raise ValueError('Please enter a valid ouput unit')

    return tff
