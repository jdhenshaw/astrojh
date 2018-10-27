#==============================================================================#
# lengthscales.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap

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
    from .kinematics import cs
    number_density = number_density * u.cm**-3
    mass_density = mu_p * ap.M_p * number_density
    mass_density = mass_density.to(u.kg / u.m**3)
    jl = cs( t, 2.33 ) * (np.pi  /  ( cds.G * mass_density ) )**0.5
    jl = jl.to(u.pc)
    return jl
