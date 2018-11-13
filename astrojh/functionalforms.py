#==============================================================================#
# functionalforms.py
#==============================================================================#
import numpy as np
from scipy.stats import norm

def polynomial_1D(x, mx, c):
    """
    Model 1D polynomial

    Parameters
    ----------
    x : ndarray
        array of x values
    mx : float
        gradient in x
    c : float
        offset

    """
    return mx*x + c

def polynomial_2D(x, mmx, mx, c):
    """
    Model 2D polynomial

    Parameters
    ----------
    x : ndarray
        array of x values
    mmx : float
    mx : float
    c : float
        offset

    """
    return mmx*x**2 + mx*x + c

def polynomial_plane1D(x, y, mx, my, c):
    """
    A 1D polynomial function for 2D data

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    mx : float
        gradient in x
    my : float
        gradient in y
    c : float
        offset

    """
    return mx*x + my*y + c

def polynomial_plane2D(x, y, mmx, mx, mmy, my, c):
    """
    Quadratic function for 2D data

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    mmx : float
    mx : float
    mmy : float
    my : float
    c : float
        offset

    """
    return mmx*x**2 + mx*x + mmy*y**2 + my*y + c

def sinusoid(x, A, T, p):
    """
    A sinusoidal function

    Parameters
    ----------
    x : ndarray
        array of x values
    A : float
        Amplitude of the sine wave
    T : float
        1/T = frequency of the wave
    p : float
        phase correction
    """
    return A*np.sin(2.*np.pi*(x-p)/T)

def sinusoid_varyfreq(x, A, T, p):
    """
    A sinusoidal function

    Parameters
    ----------
    x : ndarray
        array of x values
    A : float
        Amplitude of the sine wave
    T : ndarray
        1/T = frequency of the wave an array of T values
    p : float
        phase correction
    """
    return A*np.sin(2.*np.pi*(x-p)/T)

def gaussian(x, mu, std):
    """
    Model Gaussian profile - will accept multiple centroid values

    Parameters
    ----------
    x : ndarray
        array of xvalues
    mu : ndarray
        array of centroids (make sure this is an array otherwise it will fail)
    std : float
        array of standard devations

    """
    gauss=np.zeros_like(x, dtype=float)
    for i in range(len(mu)):
        normdist = norm(loc=mu[i], scale=std[i])
        gauss+=normdist.pdf(x)

    return gauss

def spiral_RM09(N, B, A, theta):
    """
    Model spiral pattern from Ringermacher & Mead 09. This function
    intrinsically generates a bar in a continuous, fixed relationship relative
    to an arm of arbitrary winding sweep. Unlike the logarithmic spiral, this
    spiral does not have constant pitch.

    Parameters
    ----------
    N : float
        Winding number
    B : float
        Together with N determines the spiral pitch. Greater B results in
        greater arm sweep and smaller bar/bulge, while smaller B fits larger
        bar/bulge with a sharper bar/arm junction. Thus, B controls the
        ‘bulge-to-arm’ size, while N controls the tightness much like the Hubble
        scheme
    A : float
        A scaling factor
    theta : ndarray
        an array of angles. Directly relates circular to hyperbolic functions
        via the Gudermannian function theta(x) = 2*arctan(exp(x))
    """
    return A/np.log10(B*np.tanh(theta/2/N))

def logspiral(a, b, theta):
    """
    Logarithmic spiral with constant pitch angle

    Parameters:
    -----------
    a : float
        determines the initial distance of the spiral from the origin
    b : float
        determines the winding properties of the arm
    theta : ndarray
        the azimuthal angle

    Notes:
    ------
    see https://github.com/dh4gan/tache for more spiral definitions
    """
    return a*np.exp(b*theta)

def archimedesspiral(a, theta):
    """
    archimedes spiral - power spiral function

    Parameters:
    -----------
    a : float
        determines the initial distance of the spiral from the origin
    theta : ndarray
        the azimuthal angle

    Notes:
    ------
    see https://github.com/dh4gan/tache for more spiral definitions
    """
    return a*np.power(theta,1)

def fermatspiral(a, theta):
    """
    fermat spiral - power spiral function

    Parameters:
    -----------
    a : float
        determines the initial distance of the spiral from the origin
    theta : ndarray
        the azimuthal angle

    Notes:
    ------
    see https://github.com/dh4gan/tache for more spiral definitions
    """
    return a*np.power(theta,0.5)

def hyperbolicspiral(c, theta):
    """
    hyperbolic spiral - power spiral function

    Parameters:
    -----------
    c : float

    theta : ndarray
        the azimuthal angle

    Notes:
    ------
    see https://github.com/dh4gan/tache for more spiral definitions
    """
    return c/theta
