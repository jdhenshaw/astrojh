#==============================================================================#
# functionalforms.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy import constants as const
from .conversions import *

def polynomial_plane1D(x, y, p):
    """
    The model function with parameters p

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values

    """
    a = p[0]
    b = p[1]
    c = p[2]
    return a + b*x + c*y

def residuals(p, fjac=None, x=None, y=None, z=None, err=None ):
    """
    Function that returns the weighted deviates
    """
    model  = polynomial_plane1D(x, y, p)
    status = 0
    return([status, (z-model)/err])
