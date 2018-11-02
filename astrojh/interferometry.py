#==============================================================================#
# interferometry.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import constants as const

def beam_solid_angle(beam):
    """
    Computes beam solid angle

    Parameters
    ----------
    beam : array like
        beam size in arcseconds
    """
    return beam[0]*beam[1] * np.pi / (4.0*np.log(2.))

def beamtopix(beam, pixel_size):
    """
    Computes the number of pixels in a single beam

    Parameters
    ----------
    beam : array like
        beam size in arcseconds
    pixel_size : number
        pixel size in arcseconds

    """
    from .interferometry import beam_solid_angle
    pixel_size=pixel_size*u.arcsec
    pixel_area = pixel_size**2.
    omega_beam = beam_solid_angle(beam*u.arcsec)
    return omega_beam/pixel_area
