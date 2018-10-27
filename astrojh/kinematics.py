#==============================================================================#
# kinematics.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap

# Define subroutines

def cs( t, mu_p=2.33 ):
    """
    Accepts temperature in kelvin and a molecular weight and computes the sound
    speed of that medium. Returns in km/s.

    Parameters
    ----------
    t : float
        Temperature (K)
    mu : float (optional)
        Mean molecular mass (default=2.33)

    """
    t = t*u.K
    cs = np.sqrt(( cds.k * t ) / ( mu_p * ap.M_p ))
    cs = cs.to(u.km/u.s)
    return cs

def sigma_nt( sig, t, mu ):
    """
    Accepts observed velocity dispersion in km/s, temperature in kelvin, and a
    molecular weight and computes the non-thermal contribution to sigma.
    Returns in km/s.

    Parameters
    ----------
    sig : float
        Observed velocity dipsersion (km/s)
    t : float
        Temperature (K)
    mu : float
        Molecule mass

    """
    from .kinematics import cs
    sig = sig * (u.km/u.s)
    sig_nt = np.sqrt( sig**2 - cs( t, mu )**2 )
    return sig_nt

def sigma_tot( sig, t, mu, mu_p=2.33 ):
    """
    From Fuller et al. 1992 - Accepts observed velocity dispersion in km/s,
    temperature in kelvin, the molecular weight, and the mean molecular weight,
    and estimates the total velocity dispersion of the mean molecule.

    sigma_tot**2 = cs**2 + sigma_nt**2

    Where it is assumed sigma_nt is equivalent to that derived from the
    molecular species in question.

    Returns in km/s.

    Parameters
    ----------
    sig : float
        Observed velocity dipsersion (km/s)
    t : float
        Temperature (K)
    mu : float
        Molecular mass
    mu_p : float (optional)
        Mean molecular mass (default=2.33)

    """
    from .kinematics import cs
    from .kinematics import sigma_nt
    sig_nt = sigma_nt(sig, t, mu)
    sig_tot = np.sqrt(sig_nt**2 + cs(t, mu_p)**2 )
    return sig_tot

def mach_1d( sig, t, mu, mu_p=2.33 ):
    """
    Accepts the observed velocity dispersion in km/s, temperature in kelvin,
    the molecular weight, and the mean molecular weight, estimates the 1-D mach
    number:

    Mach = sigma_nt / cs

    Returns in km/s.

    Parameters
    ----------
    sig : float
        Observed velocity dipsersion (km/s)
    t : float
        Temperature (K)
    mu : float
        Molecular mass
    mu_p : float (optional)
        Mean molecular mass (default=2.33)

    """
    from .kinematics import cs
    from .kinematics import sigma_nt
    sig_nt = sigma_nt(sig, t, mu)
    mach = sig_nt / cs(t, mu_p)
    return mach

def mach_3d( sig, t, mu, mu_p=2.33 ):
    """
    Accepts the observed velocity dispersion in km/s, temperature in kelvin,
    the molecular weight, and the mean molecular weight, estimates the 3-D mach
    number (assumes isotropy in all spatial dimensions):

    Mach = (3)**0.5 sigma_nt / cs

    Returns in km/s.

    Parameters
    ----------
    sig : float
        Observed velocity dipsersion (km/s)
    t : float
        Temperature (K)
    mu : float
        Molecular mass
    mu_p : float (optional)
        Mean molecular mass (default=2.33)

    """
    from .kinematics import mach_1d
    mach = mach_1d( sig, t, mu, mu_p )
    mach3d = mach*np.sqrt(3.)
    return mach3d
