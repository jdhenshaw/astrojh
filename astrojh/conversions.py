#==============================================================================#
# conversions.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5

def sexstrgal(ra, dec, frame='fk5'):
    """
    Sexagesimal Ra/dec string to Galactic coordinates

    Parameters
    ----------
    ra : string
        Sexagesimal Ra string
    dec : string
        Sexagesimal dec string
    frame : string
        coordinate frame
    """
    coords = SkyCoord(ra, dec, frame=frame, unit=(u.hr, u.deg))
    coords = coords.transform_to('galactic')
    lon = coords.l.deg
    lat = coords.b.deg
    return lon, lat

def Ntomsd(column_density, mu_p=2.8):
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

def msdtoN(msd, mu_p=2.8):
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

def ntorho(number_density, mu_p=2.8):
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

def rhoton(mass_density, mu_p=2.8):
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

def masstorho(mass, radius):
    """
    Accepts mass in solar masses and radius in pc and computes density

    Parameters
    ----------
    mass : float
        mass (Msun)
    radius : float
        radius (pc)
    """
    mass = mass * ap.solMass
    radius = radius * u.pc
    volume = (4. / 3.) * np.pi * radius**3.0
    mass_density = mass / volume
    mass_density=mass_density.to(u.kg/u.m**3)
    return mass_density

def masstorhoau(mass, radius):
    """
    Accepts mass in solar masses and radius in au and computes density

    Parameters
    ----------
    mass : float
        mass (Msun)
    radius : float
        radius (au)
    """
    mass = mass * ap.solMass
    radius = radius * u.AU
    volume = (4. / 3.) * np.pi * radius**3.0
    mass_density = mass / volume
    mass_density=mass_density.to(u.kg/u.m**3)
    return mass_density

def rhotomass(mass_density, radius):
    """
    Accepts density in kg/m^3 and radius in pc and computes mass

    Parameters
    ----------
    mass_density : float
        density (kg/m^3)
    radius : float
        radius (pc)

    """
    mass_density = mass_density * (u.kg/u.m**3)
    radius = radius * u.pc
    volume = (4. / 3.) * np.pi * radius**3.0
    mass = mass_density * volume
    mass = mass*(ap.solMass)
    return mass

def masston( mass, radius, mu=2.8):
    """
    Accepts mass in solar masses and radius in pc and computes number density

    Parameters
    ----------
    mass : float
        mass (Msun)
    radius : float
        radius (pc)
    mu : float (optional)
        molecular weight (default = 2.8; i.e. assumes density is given as density
        of hydrogen molecules)

    """
    mass = mass * ap.solMass
    radius = radius * u.pc
    volume = (4. / 3.) * np.pi * radius**3.0
    mass_density = mass / volume
    number_density = mass_density/(mu*ap.M_p)
    number_density=number_density.to(1./u.cm**3)
    return number_density

def masstonau( mass, radius, mu=2.8):
    """
    Accepts mass in solar masses and radius in au and computes number density

    Parameters
    ----------
    mass : float
        mass (Msun)
    radius : float
        radius (au)
    mu : float (optional)
        molecular weight (default = 2.8; i.e. assumes density is given as density
        of hydrogen molecules)

    """
    mass = mass * ap.solMass
    radius = radius * u.AU
    volume = (4. / 3.) * np.pi * radius**3.0
    mass_density = mass / volume
    number_density = mass_density/(mu*ap.M_p)
    number_density=number_density.to(1./u.cm**3)
    return number_density
