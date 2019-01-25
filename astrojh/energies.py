#==============================================================================#
# energies.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap

def Edot_turb(M, sigma, L, obs=False, angle=45.0):
    """
    Computes the total loss of turbulent kinetic energy (E=1/2 M sigma^2), to a
    system / cloud with total mass, M through turbulent decay

    Parameters
    ----------
    M : number
        Total mass of the system / cloud mass (solar masses)
    sigma : number
        3D turbulent velocity dispersion (km/s)
    L : number
        Driving scale of the turbulence (pc)

    """
    M = M * ap.solMass
    sigma = sigma * (u.km / u.s)
    L = L * u.pc
    if obs:
        # if we are dealing with observed quantities the true driving scale
        # may depend on the angle
        angle_radians = np.deg2rad(angle)
        L = L / np.sin(angle_radians)

    Edot = (-1./2.) * M * sigma**3. / L
    Edot = Edot.to(u.erg/u.s)

    return Edot

def Edot_acc(M_gas, v_acc, sigma_gas, r, obs=False, angle=45.0):
    """
    Computes the associated kinetic energy gain of a system due to accretion

    Equation
    --------
    E = 1/2 M_gas(r) sigma_gas^2(r)

    M_gas is the mass of gas at a distance r, and sigma_gas is the gas velocity.
    The gas velocity can be tailored to include all motions (as in e.g. a
    lw-size relationship) or those specfically attributed to accretion, for
    example.

    The time taken for the gas to accrete from a radius r to a central point
    (r=0), assuming a constant accretion velocity, is t_acc

    t_acc = r/v_acc

    Therefore

    Edot = 1/2 M_gas(r) sigma_gas^2(r) v_acc / r

    Parameters
    ----------
    M_gas : number
        Gas mass at a distance r (solar masses)
    v_acc : number
        Accretion velocity (km/s)
    sigma_gas : number
        Gas velocity at a distance r (km/s)
    r : number
        Radius over which M_gas and sigma_gas are measured (pc)

    """
    M_gas = M_gas * ap.solMass
    v_acc = v_acc * (u.km / u.s)
    sigma_gas = sigma_gas * (u.km / u.s)
    r = r * (u.pc)
    if obs:
        # if we are dealing with observed quantities the true driving scale
        # may depend on the angle
        angle_radians = np.deg2rad(angle)
        v = v / np.cos(angle_radians)

    Edot = (1./2.) * M_gas * sigma_gas**2 * v_acc / r
    Edot = Edot.to(u.erg/u.s)

    return Edot
