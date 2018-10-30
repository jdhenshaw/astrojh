#==============================================================================#
# kinematics.py
#==============================================================================#
import numpy as np
import sys
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy import constants as const
from .conversions import *
sys.path.append('/Users/henshaw/Dropbox/Work/Code/GitHub/astrolibpy/mpfit/')
import mpfit
from .functionalforms import *
from .imagetools import index_coords

def cs( t, mu_p=2.33 ):
    """
    Accepts temperature in kelvin and a molecular weight and computes the sound
    speed of that medium. Returns in km/s.

    Parameters
    ----------
    t : float
        Temperature (K)
    mu_p : float (optional)
        Mean molecular mass (default=2.33)

    """
    t = t*u.K
    cs = np.sqrt(( const.k_B * t ) / ( mu_p * ap.M_p ))
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

def va( B, number_density, mu=2.8):
    """
    Accepts magnetic field strength in Gauss and number density in particles per
    cubic centimetre and computes the alfven speed. Returns in km/s

    Parameters
    ----------
    B : float
        Magnetic field strength (G)
    number_density : float
        number density (cm^-3)
    mu : float (optional)
        Molecular mass used to estimate the mass density (default=2.8)

    """
    mass_density = ntorho(number_density, mu)
    B = B*u.G

    alfven_speed = B / (const.mu0 * mass_density )**0.5
    alfven_speed = alfven_speed.to(u.km/u.s)
    return alfven_speed

def plasma_beta( t, B, number_density, mu=2.8, mu_p=2.33 ):
    """
    Accepts temperature in K, magnetic field strength in Gauss and number
    density in particles per cubic centimetre and computes the alfven speed.
    Returns in km/s

    Parameters
    ----------
    t : float
        Temperature (K)
    B : float
        Magnetic field strength (G)
    number_density : float
        number density (cm^-3)
    mu : float (optional)
        Molecular mass used to estimate the mass density (default=2.8)
    mu_p : float (optional)
        Mean molecular mass (default=2.33)

    """
    from .kinematics import va
    from .kinematics import cs

    sound_speed = cs( t, mu_p )
    alfven_speed = va( B, number_density, mu )

    beta = 2. * (sound_speed/alfven_speed)**2.
    return beta

def planefit(x, y, z, errz, pinit=None):
    """
    Fits a first-degree bivariate polynomial to data

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    z : ndarray
        data to be fit
    errz : ndarray
        uncertainties on z data
    pinit : ndarray (optional)
        initial guesses for fitting
    """
    if pinit is None:
        pinit=[0,0,0]
    data = {'x':x, 'y':y, 'z':z, 'err':errz}
    model = mpfit.mpfit(residuals, pinit, functkw=data, quiet = True)
    popt = model.params
    perr = model.perror
    return popt, perr

def subtractplane_img(img, model):
    """
    Subtracts a model plane from data

    Parameters
    ----------
    img : ndarray
        2-D image
    model : ndarray
        Model parameters output from planefit
        
    """

    modelimg=np.empty(np.shape(img))
    modelimg.fill(np.NaN)
    for x in range(len(modelimg[0,:])):
        for y in range(len(modelimg[:,0])):
            if ~np.isnan(img[y,x]):
                modelimg[y,x]=polynomial_plane1D(x,y,model)
    residualimg = img-modelimg
    return residualimg, modelimg
