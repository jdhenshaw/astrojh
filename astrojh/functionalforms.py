#==============================================================================#
# functionalforms.py
#==============================================================================#
import numpy as np
from scipy.stats import norm
from .datatools import interpolate1D

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

def exponential(x, a, b, c):
    """
    Model 1D polynomial

    Parameters
    ----------
    x : ndarray
        array of x values

    """
    return a*np.exp(-b*x)+c

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

def polynomial_plane2Dy(x, y, mx, mmy, my, c):
    """
    Quadratic function in y linear in x

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    mx : float
    mmy : float
    my : float
    c : float
        offset

    """
    return mx*x + mmy*y**2 + my*y + c

def sinusoid(x, amp, wavelength, phase, offset):
    """
    A sinusoidal function

    Parameters
    ----------
    x : ndarray
        array of x values
    amp : float
        Amplitude of the sine wave
    wavelength : float
        wavelength
    phase : float
        phase correction
    offset: float
        mean
    """
    #return A*np.sin(2.*np.pi*(x-p)/T)
    return amp*np.sin((2.*np.pi*x / wavelength) + phase) + offset

def sinusoid_rand(x0, amp, wavelength, method='linear', npoints=100, alt=False):
    """
    A sinusoidal function

    Parameters
    ----------
    x : ndarray
        array of x values
    amp : float
        Array of amplitude values
    wavelength : float
        Array of wavelength values
    phase : float
        Array of phase values
    offset: float
        array of offset values
    """

    _x = []
    _y = []

    r = range(len(amp))
    r = list(r)
    for j in range(len(amp)):
        if (j % 2 != 0):
            r[j]=r[j]-1
    _r = np.copy(r)
    del r[0::2]
    rshort = list(range(len(r)))
    ids = []
    for j in rshort:
        if (j % 2 != 0):
            id = list(np.where(_r == r[j])[0])

            ids.append(id)
    ids = [val for sublist in ids for val in sublist]

    for i in range(len(amp)):

        if i == 0:
            x0 = x0
        else:
            x0 = x0+wavelength[i-1]/2.

        x = np.linspace(0.0, wavelength[i]*10, num=1000)
        y = sinusoid(x, amp[i], wavelength[i], 0.0, 0.0)
        id = np.where(x<((wavelength[i]/2)))
        id = id[0]
        x=x[id]
        y=y[id]
        x = x+x0


        if alt:
            if i in ids:
                y=y*-1
        else:
            if (i % 2 != 0):
                y=y*-1

        _x.append([val for val in x])
        _y.append([val for val in y])

    _x = [_val for _list in _x for _val in _list]
    _y = [_val for _list in _y for _val in _list]

    xarr = np.asarray(_x)
    yarr = np.asarray(_y)

    _xarr = np.linspace(np.min(xarr), np.max(xarr), npoints)
    _yarr = interpolate1D(xarr, yarr, _xarr, kind=method)

    return _xarr, _yarr



def sinusoid_varyfreq(x, A, B, C, D):
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
    return A*np.sin((2.*np.pi*x / B) + C) + D
    #return A*np.sin(2.*np.pi*(x-p)/T)

def gaussian(x, mu, std, amp=None):
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
    amp : float (optional)
        array of amplitudes, otherwise set to 1.0


    """
    if amp is None:
        amp = np.ones(len(mu))

    gauss=np.zeros_like(x, dtype=float)
    for i in range(len(mu)):
        normdist = norm(loc=mu[i], scale=std[i])
        gaussfunc = (normdist.pdf(x)/np.max(normdist.pdf(x))) * amp[i]
        if ~np.all(np.isnan(gaussfunc)):
            gauss+=gaussfunc

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
