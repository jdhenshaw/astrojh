#==============================================================================#
# density.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import constants as const
from .interferometry import beam_solid_angle
from .conversions import ntorho, Ntomsd, rhoton

def column_density(flux, beam, wave, t, kappa, mu=2.8):
    """
    Computes column density from continuum data

    Parameters
    ----------
    flux : float
        flux in Jy/beam
    beam : array like
        beam size in arcseconds
    wave : float
        wavelength (m)
    t: float (optional)
        temperature (K)
    kappa: float
        dust opacity per unit mass (cm^2g^-1)
    mu : float (optional)
        molecular weight (default = 2.8; i.e. assumes density is given as density
        of hydrogen molecules)
    """
    beam = beam*u.arcsec
    omega_beam = beam_solid_angle(beam)
    flux = (flux*u.Jy/u.beam).to(u.Jy/u.sr, equivalencies=u.beam_angular_area(omega_beam))
    wave=wave*u.m
    t=t*u.K
    kappa=kappa*(u.cm**2/u.g)

    B = blackbody_nu(wave.to(u.Hz, equivalencies=u.spectral()), t)

    N = flux / (mu * ap.M_p * kappa * B)
    N = N.to(1./u.cm**2)

    return N

def density_cylinder_cs(lambda_frag, t, mu_p = 2.8):
    """
    Estimates the density of a filament assuming a fragmentation length scale
    and a temperature from: lambda = 22. * c_s / ( 4 * pi * G * rho )**0.5

    Parameters
    ----------
    lambda_frag : number
        fragmentation lengthscale (pc)
    t : number
        temperature (K)
    mu_p : float (optional)
        molecular weight (default = 2.8; i.e. assumes density is given as density
        of hydrogen molecules)
    """
    from .kinematics import cs
    lambda_frag = lambda_frag * (u.pc)
    mass_density = (1./(4.*np.pi*const.G)) * ((22. * cs( t, 2.33 )) / lambda_frag)**2
    mass_density = mass_density.to(u.kg / u.m**3)
    number_density = rhoton(mass_density.value, mu_p)
    return number_density

def density_cylinder(lambda_frag, sigma, mu_p = 2.8):
    """
    Estimates the density of a filament assuming a fragmentation length scale
    and a temperature from: lambda = 22. * sigma / ( 4 * pi * G * rho )**0.5

    Parameters
    ----------
    lambda_frag : number
        fragmentation lengthscale (pc)
    sigma : number
        velocity dispersion (km/s)
    mu_p : float (optional)
        molecular weight (default = 2.8; i.e. assumes density is given as density
        of hydrogen molecules)
    """
    sigma = sigma * (u.km/u.s)
    lambda_frag = lambda_frag * (u.pc)
    mass_density = (1./(4.*np.pi*const.G)) * ((22. * sigma) / lambda_frag)**2
    mass_density = mass_density.to(u.kg / u.m**3)
    number_density = rhoton(mass_density.value, mu_p)
    return number_density
