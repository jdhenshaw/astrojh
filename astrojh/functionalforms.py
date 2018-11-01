#==============================================================================#
# functionalforms.py
#==============================================================================#
import numpy as np

def polynomial_plane1D(x, y, mx, my, c):
    """
    The model function with parameters p

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values

    """
    return np.array([mx*x + my*y + c])

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
    Logarithmic spiral

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
