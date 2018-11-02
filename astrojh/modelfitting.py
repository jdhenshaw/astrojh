#==============================================================================#
# modelfitting.py
#==============================================================================#
import numpy as np
import lmfit
from .functionalforms import polynomial_plane1D, spiral_RM09, logspiral, \
                             archimedesspiral, fermatspiral, hyperbolicspiral
from .imagetools import cart2polar, polar2cart
import sys
from scipy import optimize

def planefit(x, y, z, err=None, pinit=None, method='leastsq', report_fit=False):
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
    err : ndarray (optional)
        uncertainties on z data
    pinit : ndarray (optional)
        initial guesses for fitting. Format = [mx, my, c]
    method : string (optional)
        method used for the minimisation (default = leastsq)

    """
    if pinit is None:
        pars=lmfit.Parameters()
        pars.add('mx', value=1.0)
        pars.add('my', value=1.0)
        pars.add('c', value=1.0)
    else:
        pars=lmfit.Parameters()
        pars.add('mx', value=pinit[0])
        pars.add('my', value=pinit[1])
        pars.add('c', value=pinit[2])

    fitter = lmfit.Minimizer(residual_planefit, pars,
                             fcn_args=(x,y),
                             fcn_kws={'data':z, 'err':err},
                             nan_policy='propagate')

    result = fitter.minimize(method=method)

    if report_fit:
        lmfit.report_fit(result)

    popt = np.array([result.params['mx'].value,
                     result.params['my'].value,
                     result.params['c'].value])
    perr = np.array([result.params['mx'].stderr,
                     result.params['my'].stderr,
                     result.params['c'].stderr])

    return popt, perr, result

def residual_planefit(pars, x, y, data=None, err=None):
    """
    Minmizer for lmfit for fitting a 2-D plane to data

    Parameters
    ----------
    pars : lmfit.Parameters()

    x : ndarray
        array of x positions
    y : ndarray
        array of y positions
    data : ndarray
        2-D image containing the data
    err : ndarray
        uncertainties on the data

    """
    parvals = pars.valuesdict()
    mx = parvals['mx']
    my = parvals['my']
    c = parvals['c']
    model = polynomial_plane1D(x, y, mx, my, c)

    if data is None:
        min = np.array([model])
        return min
    if err is None:
        min = np.array([model - data])
        return min
    min = np.array([(model-data) / err])

    return min

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
                modelimg[y,x]=polynomial_plane1D(x,y,model[0],model[1],model[2])
    residualimg = img-modelimg
    return residualimg, modelimg

def spiralfit(xdata, ydata, pinit=None, method='leastsq', model='RM09',
              project=True, report_fit=False, full=True, flip=False):
    """
    Fits a spiral to a series of data points
    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    pinit : ndarray (optional)
        initial guesses for fitting.
    method : string (optional)
        method used for the minimisation (default = leastsq)
    model : string (optional)
        defines the model for fitting
    project : string (optional)
        project from galactic frame to plane of the sky
    report_fit : bool
        if True prints out best fit results
    full : bool
        Fit from 180 to 0 deg and also 180 to 360
    flip : bool
        Fit from -180 to 0 deg and also -180 to -360
    """
    # first get the parameters
    pars, parnames = get_spiral_pars(pinit, model, project)
    fitter = lmfit.Minimizer(residual_spiral, pars,
                             fcn_args=(xdata, ydata),
                             fcn_kws={'model':model, 'project':project,
                                      'full':full, 'flip':flip},
                             nan_policy='propagate')
    result = fitter.minimize(method=method, params=pars,maxfev=1800)

    if report_fit:
        lmfit.report_fit(result)

    popt=[result.params[name].value for name in parnames]
    popt=np.asarray(popt)
    perr=[result.params[name].stderr for name in parnames]
    perr=np.asarray(popt)

    return popt, perr, result

def residual_spiral(pars, xdata, ydata, model=None, project=None,
                    full=True, flip=False):
    """
    Minmizer for fitting a spiral. Minmizes the distance between the input data
    and the cartesian coordinates of the model spiral generated by
    transform_to_cart

    Parameters
    ----------
    pars : lmfit.Parameters()
    x : ndarray
        array of x positions
    y : ndarray
        array of y positions
    model : string (optional)
        defines the model for fitting
    project : string (optional)
        project from galactic frame to plane of the sky
    """
    model_x, model_y = make_spiral(pars,model,project=project,
                                   full=full,flip=flip)

    dist = np.zeros_like(xdata)
    xmin = np.zeros_like(xdata)
    ymin = np.zeros_like(xdata)

    for i in range(dist.shape[0]):
        d = np.sqrt((model_x - xdata[i])**2 + (model_y - ydata[i])**2 )
        dist[i] = np.min(d)
    return dist

def make_spiral(pars,model,project=None, full=True, flip=False, npoints=100):
    """
    Makes a spiral model for minimisation

    Parameters
    ----------
    pars : ndarray
        array of parameters describing the model
    model : string
        defines the model for fitting
    project : string (optional)
        project from galactic frame to plane of the sky
    full : bool
        Fit from 180 to 0 deg and also 180 to 360
    flip : bool
        Fit from -180 to 0 deg and also -180 to -360
    npoints : int
        number of points for the model
	"""
    parvals = pars.valuesdict()
    theta = np.arange(0,180*np.pi/180.,1./npoints)

    if model=='RM09':
        r = spiral_RM09(parvals['N'], parvals['B'], parvals['A'], theta)
    elif model=='logspiral':
        r = logspiral(parvals['a'], parvals['b'], theta)
    elif model=='archimedesspiral':
        r = archimedesspiral(parvals['a'], theta)
    elif model=='fermatspiral':
        r = fermatspiral(parvals['a'], theta)
    elif model=='hyperbolicspiral':
        r = hyperbolicspiral(parvals['c'], theta)

    if full:
        r = np.hstack((r[::-1],r[1:]))
        theta = np.hstack((theta[::-1], (theta+np.pi)[1:] ))

    if flip:
        theta=theta*-1

    xc,yc = transform_to_cart(r, theta, parvals['x0'], parvals['y0'])
    zc = np.zeros_like(xc)

    if project:
        euler = np.array((parvals['e0'],parvals['e1'],parvals['e2']))*np.pi/180.
        x_rot, y_rot, z_rot = spiral_pos(xc,yc,zc,euler)
        return x_rot, y_rot
    else:
        return xc, yc

def spiral_pos(xc,yc,zc,euler):
    """
    Projects a spiral model from the galaxy plane to the plane of the sky

    x : ndarray
        array of cartesian x values corresponding to the spiral model
    y : ndarray
        array of cartesian y values corresponding to the spiral model
    z : ndarray
        empty array of z values to be determined during minimisation
    euler : ndarray
        array of euler angles used for projection to the plane of the sky

    Notes:
    ------
    credit: I-Ting Ho
            (see http://iopscience.iop.org/article/10.3847/1538-4357/aa8460/pdf)
    """
    x_rot = ( np.cos(euler[2]) * np.cos(euler[1]) * np.cos(euler[0]) -
              np.sin(euler[2]) * np.sin(euler[0]) )* xc + \
			( np.cos(euler[2]) * np.cos(euler[1]) * np.sin(euler[0]) +
              np.sin(euler[2]) * np.cos(euler[0]) )* yc + \
			( -1. * np.cos(euler[2]) * np.sin(euler[1]) ) * zc

    y_rot = ( -1. * np.sin(euler[2]) * np.cos(euler[1]) * np.cos(euler[0]) -
              np.cos(euler[2]) * np.sin(euler[0]) )* xc + \
			( -1. * np.sin(euler[2]) * np.cos(euler[1]) * np.sin(euler[0]) +
              np.cos(euler[2]) * np.cos(euler[0]) )* yc + \
			( np.sin(euler[2]) * np.sin(euler[1]) ) * zc

    z_rot = np.sin(euler[1]) * np.cos(euler[0]) * xc + \
			np.sin(euler[1]) * np.sin(euler[0]) * yc + \
			np.cos(euler[1]) * zc

    return x_rot, y_rot, z_rot

def get_spiral_pars(pinit,model,project=True):
    """
    Creates the parameter list for the minimisation

    Parameters
    ----------
    pinit : ndarray
        initial guesses for the minimisation
    model : string
        name of the model you are wanting to fit
    project : string (option)
        project from galactic frame to plane of the sk

    """
    pars=lmfit.Parameters()
    parnames=['x0','y0']
    for i in range(len(parnames)):
        if pinit is None:
            pars.add(parnames[i],value=0.0)
        else:
            pars.add(parnames[i],value=pinit[i])

    #==========================================================================#
    # RM09

    if model == 'RM09':
        parnames.extend(['N','B','A'])
        for i in range(2,len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
            else:
                pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(5,len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
                else:
                    pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

    #==========================================================================#
    # Logspiral

    if model == 'logspiral':
        parnames.extend(['a','b'])
        for i in range(2,len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
            else:
                pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(4,len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=0.0, min =-np.inf ,max=np.inf)
                else:
                    pars.add(parnames[i], value=pinit[i], min =-np.inf ,max=np.inf)

    #==========================================================================#
    # archimedes spiral

    if model == 'archimedesspiral':
        parnames.extend(['a'])
        for i in range(2,len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
            else:
                pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(3,len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=0.0, min =-np.inf ,max=np.inf)
                else:
                    pars.add(parnames[i], value=pinit[i], min =-np.inf ,max=np.inf)

    #==========================================================================#
    # fermat spiral

    if model == 'fermatspiral':
        parnames.extend(['a'])
        for i in range(2,len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
            else:
                pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(3,len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=0.0, min =-np.inf ,max=np.inf)
                else:
                    pars.add(parnames[i], value=pinit[i], min =-np.inf ,max=np.inf)

    #==========================================================================#
    # hyperbolic spiral

    if model == 'hyperbolicspiral':
        parnames.extend(['c'])
        for i in range(2,len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=0.0, min = -np.inf, max=np.inf)
            else:
                pars.add(parnames[i], value=pinit[i], min = -np.inf, max=np.inf)

        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(3,len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=0.0, min =-np.inf ,max=np.inf)
                else:
                    pars.add(parnames[i], value=pinit[i], min =-np.inf ,max=np.inf)

    return pars, parnames

def transform_to_cart(r, theta, x0, y0):
    """
    Transform polar coordinates to Cartesian

    Parameters
    -------
    r : ndarray
        radial position
    theta : ndarry
        angular position
    x0 : float
        origin x position
    y0 : float
        origin y position

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    x = (r * np.cos(theta))+x0
    y = (r * np.sin(theta))+y0
    return x, y

def transform_to_polar(x, y, x0, y0):
    """
    Transform cartesian to polar

    Parameters
    -------
    x : ndarray
        cartesian x locations
    y : ndarray
        cartesian y locations
    x0 : float
        origin x position
    y0 : float
        origin y position

    Returns
    ----------
    r, theta : floats or arrays
        polar coordinates
    """
    r = np.sqrt((x+x0)**2 + (y-y0)**2)
    theta = np.arctan2((x+x0), (y-y0))
    return r, theta
