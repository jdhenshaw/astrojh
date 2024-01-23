#==============================================================================#
# feedback.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy import constants as const

def Nly(integrated_flux, frequency, electron_temperature, distance):
    """
    Estimates the total number of Lyman continuum emitting photons. Equation
    from Mezger & Henderson 1967 under the assumptions of Rubin 1968.

    Parameters
    ----------
    integreated_flux : number
        integrated radio continuum flux in units of Jy
    frequency : number
        frequency in units of GHz
    electron_temperature : number
        electron temperature in units of K
    distance : number
        source distance in units of pc
    """
    integrated_flux=integrated_flux * (ap.Jy )
    frequency=frequency*1.e9*u.Hz
    electron_temperature=electron_temperature*u.K
    distance=distance*u.pc

    Nly = 8.9e40 * (integrated_flux / (1*ap.Jy)) * \
                   (frequency / (1.e9*u.Hz))**0.1 * \
                   (electron_temperature / (1.e4*u.K))**-0.45 * \
                   (distance / (1.*u.pc))**2

    Nly = Nly * u.s**-1
    return Nly

def n_e(integrated_flux, frequency, electron_temperature, distance, angular_size):
    """
    Estimates the electron density in a HII region

    Parameters
    ----------
    integreated_flux : number
        integrated radio continuum flux in units of Jy
    frequency : number
        frequency in units of GHz
    electron_temperature : number
        electron temperature in units of K
    distance : number
        source distance in units of pc
    angular_size : number
        angular size of the observed HII region in units of arcseconds
    """
    integrated_flux=integrated_flux * (ap.Jy )
    frequency=frequency*1.e9*u.Hz
    electron_temperature=electron_temperature*u.K
    distance=distance*u.pc
    angular_size=angular_size*u.arcsec

    n_e = 2.30e6 * (integrated_flux / (1*ap.Jy))**0.5 * \
                   (frequency / (1.e9*u.Hz))**0.05 * \
                   (electron_temperature / (u.K))**0.175 * \
                   (distance / (1.*u.pc))**-0.5 * \
                   (angular_size / (1.*u.arcsec))**-1.5

    n_e = n_e * u.cm**-3
    return n_e

def t_rec(n_e, alpha=2.6e-13):
    """
    Estimates the recombination time of ionised gas in an HII region assuming an
    electron density and type B recombination rate coefficient.

    Parameters
    ----------
    n_e : number
        electron density in units of electrons per cc
    alpha (optional): number
        Recombination coefficient in units of cm^3 s^-1. If not supplied a
        value of 2.6x10^-13 will be assumed - relevant for an electron
        temperature of 10000K. See Draine 2011 table 14.1 for more.
    """
    n_e = n_e * u.cm**-3
    alpha = alpha * u.cm**3 * u.s**-1

    t_rec = 1./(n_e*alpha)

    t_rec = t_rec.to(u.yr)

    return t_rec

def r_strom(Nly, n_e, alpha=2.6e-13):
    """
    Estimates the Strömgren radius of an HII region

    Parameters
    ----------
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    n_e : number
        electron density in units of electrons per cc
    alpha (optional): number
        Recombination coefficient in units of cm^3 s^-1. If not supplied a
        value of 2.6x10^-13 will be assumed - relevant for an electron
        temperature of 10000K. See Draine 2011 table 14.1 for more.
    """
    Nly = Nly * u.s**-1
    n_e = n_e * u.cm**-3
    alpha = alpha * u.cm**3 * u.s**-1
    r_strom = ((Nly*3.) / (4.*np.pi*alpha*n_e**2))**(1./3.)
    r_strom = r_strom.to(u.pc)

    return r_strom

def t_dyn(r_strom, radius, temperature, mu_p=0.64):
    """
    estimates the dynamical time of an HII region. Equation from Dyson &
    Williams

    Parameters
    ----------
    r_strom : number
        The Strömgren radius of the HII region in units of pc
    radius : number
        The observed radius of the HII region in units of pc
    temperature : number
        Temperature of ionised gas in the HII region
    mu_p (optional): number
        The mean particle mass to compute the sound speed. Assumed 0.64 relevant
        for 90% H and 10% He.
    """
    from .kinematics import cs
    r_strom = r_strom * u.pc
    radius = radius * u.pc

    t_dyn = ((4.*r_strom)/(7.*cs(temperature, mu_p=mu_p)))*((radius/r_strom)**(7./4.) - 1)
    t_dyn = t_dyn.to(u.yr)

    return t_dyn

def momentum(density, Nly, time, temperature):
    """
    estimates the momentum of a shell driven by a HII region. Equation from
    Krumholz 2017 - equation 7.36

    Parameters
    ----------
    density : number
        density of the ambient medium within which the shell is expanding into.
        units of particles per cc
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    time : number
        age of the HII region in yrs
    temperature : number
        Temperature of ionised gas in the HII region in units of K

    """
    density = density * u.cm**-3
    Nly = Nly * u.s**-1
    time = time * u.yr
    temperature = temperature * u.K

    P = 1.5e5 * (density/(100.* u.cm**-3))**(-1./7.) * \
                (Nly/(1e49 * u.s**-1))**(4./7.) * \
                (time/( 1.e6* u.yr))**(9./7.) * \
                (temperature /(1.e4 * u.K))**(-8./7.)

    P = P * ap.solMass * u.km * u.s**-1

    return P

def energy(density, Nly, time, temperature):
    """
    Estimates the energy in an expanding shell. Equation from
    Krumholz 2017 - equation 7.35

    Parameters
    ----------
    density : number
        density of the ambient medium within which the shell is expanding into.
        units of particles per cc
    Nly : number
        Number of Lyman continuum emitting photons in units of s^-1
    time : number
        age of the HII region in yrs
    temperature : number
        Temperature of ionised gas in the HII region in units of K

    """
    density = density * u.cm**-3
    Nly = Nly * u.s**-1
    time = time * u.yr
    temperature = temperature * u.K

    energy = 8.1e47 * (density/(100.* u.cm**-3))**(-10./7.) * \
                (Nly/(1e49 * u.s**-1))**(5./7.) * \
                (time/(1.e6 * u.yr))**(6./7.) * \
                (temperature /(1.e4 * u.K))**(10./7.)

    energy = energy * u.erg

    return energy

def r_spitz(Nly, n_e, temperature, t, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    t = t * u.yr
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)

    r_s = r_st * (1.+((7.*cs_ion*t)/(4.*r_st)))**(4./7.)

    r_s = r_s.to(u.pc)
    return r_s

def v_spitz(Nly, n_e, temperature, t, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    t = t * u.yr
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)

    v_s = cs_ion * (1.+((7.*cs_ion*t)/(4.*r_st)))**(-3./7.)
    v_s = v_s.to(u.km/u.s)
    return v_s

def t_spitz(Nly, n_e, temperature, r, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    r = r*u.pc
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)
    t_s=(4./7.)*(r_st/cs_ion)*((r/r_st)**(7.0/4.0)-1.0)
    t_s=t_s.to(u.yr)
    return t_s

def r_hosokawa(Nly, n_e, temperature, t, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    t = t * u.yr
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)

    r_h = r_st * (1.+((7.*np.sqrt(4)*cs_ion*t)/(4.*np.sqrt(3)*r_st)))**(4./7.)
    return r_h

def v_hosokawa(Nly, n_e, temperature, t, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    t = t * u.yr
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)

    v_h = cs_ion * (4./3.)**(0.5)*(1.+((7.*np.sqrt(4)*cs_ion*t)/(4.*np.sqrt(3)*r_st)))**(-3./7.)
    v_h = v_h.to(u.km/u.s)
    return v_h

def t_hosokawa(Nly, n_e, temperature, r, mu_p=0.64, alpha=2.6e-13):
    from .kinematics import cs
    temperature = temperature * u.K
    r = r*u.pc
    cs_ion = cs(temperature.value,mu_p=mu_p)
    r_st = r_strom(Nly, n_e, alpha=alpha)
    t_h=(4./7.)*(3./4.)**(0.5)*(r_st/cs_ion)*((r/r_st)**(7.0/4.0)-1.0)
    t_h=t_h.to(u.yr)
    return t_h

def Lwind(mass_loss, vinf):
    mass_loss = mass_loss * ap.solMass * u.yr**-1
    vinf = vinf * u.km * u.s**-1

    Lwind = 0.5 * mass_loss * vinf**2
    Lwind = Lwind.to(u.erg * u.s**-1)

    return Lwind

def r_weaver(mass_loss, vinf, rho0, tdyn):
    Lw=Lwind(mass_loss, vinf)
    rho0 = rho0 * u.g* u.cm**-3
    tdyn = tdyn * u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    rw = cons * (Lw/rho0)**(1./5.) * tdyn**(3./5.)

    rw = rw.to(u.pc)
    return rw

def r_weaver_tcool(mass_loss, vinf, rho0, tdyn, tcool):
    Lw=Lwind(mass_loss, vinf)
    rho0 = rho0 * u.g* u.cm**-3
    tdyn = tdyn * u.yr
    tcool=tcool*u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    rw = cons * (Lw/rho0)**(1./5.) * tdyn**(1./4.) * tcool **(7./20.)

    rw = rw.to(u.pc)
    return rw

def r_weaver_Lwind(Lw, rho0, t):
    Lw=Lw*(u.erg*u.s**-1)
    rho0 = rho0 * u.g* u.cm**-3
    t= t * u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    rw = cons * (Lw/rho0)**(1./5.) * t**(3./5.)

    rw = rw.to(u.pc)
    return rw

def t_weaver(r,Lw,rho0):
    Lw=Lw*(u.erg*u.s**-1)
    rho0 = rho0 * u.g* u.cm**-3
    rw=r*u.pc
    cons=(125./(154.*np.pi))**(1./5.)

    tw = (rw/cons)**(5./3.)*(Lw/rho0)**(-1./3.)
    tw=tw.to(u.yr)
    return tw

def t_weaver_tcool(r,Lw,rho0, tcool):
    Lw=Lw*(u.erg*u.s**-1)
    rho0 = rho0 * u.g* u.cm**-3
    rw=r*u.pc
    tcool=tcool*u.yr
    cons=(125./(154.*np.pi))**(1./5.)

    tw = (rw/cons)**(4.)*(Lw/rho0)**(-4./5.)*tcool**(-7./5.)
    tw=tw.to(u.yr)
    return tw

def t_weaver_constv(r, v):
    r=r*u.pc
    v=v*u.km/u.s

    tw = (3./5.)*r/v
    tw=tw.to(u.yr)
    return tw


def l_weaver(mass, r, v):
    mass = mass * ap.solMass
    r = r *u.pc
    v=v*u.km/u.s

    lw=(77./18.)*mass*v**3 / r
    lw=lw.to(u.erg/u.s)
    return lw

def v_weaver(mass_loss, vinf, rho0, t):
    Lw=Lwind(mass_loss, vinf)
    rho0 = rho0 * u.g* u.cm**-3
    t = t * u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    vw = (3./5.) * cons * (Lw/rho0)**(1./5.) * t**(-2./5.)

    vw = vw.to(u.km*u.s**-1)
    return vw

def v_weaver_Lwind(lw, rho0, t):
    Lw=lw*u.erg/u.s
    rho0 = rho0 * u.g* u.cm**-3
    t = t * u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    vw = (3./5.) * cons * (Lw/rho0)**(1./5.) * t**(-2./5.)

    vw = vw.to(u.km*u.s**-1)
    return vw

def v_weaver_Lwind_tcool(lw, rho0, t, tcool):
    Lw=lw*u.erg/u.s
    rho0 = rho0 * u.g* u.cm**-3
    t = t * u.yr
    tcool=tcool*u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    vw = (1./4.) * cons * (Lw/rho0)**(1./5.) * t**(-3./4.) * tcool**(7./20.)

    vw = vw.to(u.km*u.s**-1)
    return vw
# def energy_weaver(lw, t):
#     lw=lw*u.erg/u.s
#     t=t*u.yr
#     ew = 1.4e49*u.erg*(lw/(1e36*u.erg/u.s))*(t/(1e6*u.yr))
#     return ew

def energy_weaver(Lw, t):
    Lw=Lw*u.erg/u.s
    t=t*u.yr
    ew = (15./77.)*Lw*t
    ew=ew.to(u.erg)
    return ew

def tcool(z, lw, n0):
    Lw=lw*u.erg/u.s
    n0 = n0 * u.cm**-3
    tcool=0.96*(z)**(-35./22.) * (Lw/(1e37*u.erg/u.s))**(3./11.) * (n0/(20.* u.cm**-3))**(-8./11.)
    tcool=tcool*u.Myr
    tcool=tcool.to(u.yr)
    return tcool

def rcool(mass_loss, vinf, rho0, z):
    Lw=Lwind(mass_loss, vinf)
    from .conversions import rhoton
    rho0 = rho0 * u.g* u.cm**-3
    rho0 = rho0.to(u.kg * u.m**-3)
    n0 = rhoton(rho0.value, mu_p=1.4)
    tc = tcool(z, Lw.value, n0.value)

    rc = r_weaver_Lwind(Lw.value,rho0.to(u.g * u.cm**-3).value, tc.value)

    return rc



def r_weaver_Lwind_tcool(Lw, rho0, t, tcool):
    Lw=Lw*(u.erg*u.s**-1)
    rho0 = rho0 * u.g* u.cm**-3
    t= t * u.yr
    tcool=tcool * u.yr
    cons=(125./(154.*np.pi))**(1./5.)
    rw = cons * (Lw/rho0)**(1./5.) * t**(1./4.)*tcool**(7./20.)

    rw = rw.to(u.pc)
    return rw
