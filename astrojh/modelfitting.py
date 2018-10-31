#==============================================================================#
# modelfitting.py
#==============================================================================#
import numpy as np
import lmfit
from .functionalforms import polynomial_plane1D, spiral_RM09
from .imagetools import cart2polar, polar2cart
import sys

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
                             fcn_kws={'data':z, 'err':err})

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

def spiralfit(x, y, pinit=None, method='leastsq', model='RM09', project=True,
              report_fit=False):
    """
    Fits a spiral to a series of data points

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    pinit : ndarray (optional)
        initial guesses for fitting. Format = [mx, my, c]
    method : string (optional)
        method used for the minimisation (default = leastsq)
    model : string (optional)
        defines the model for fitting
    project : string (optional)
        project from galactic frame to plane of the sky
    report_fit : bool
        if True prints out best fit results

    """
    # first get the parameters
    pars, parnames = get_spiral_pars(pinit, model, project)
    fitter = lmfit.Minimizer(residual_spiral, pars,
                             fcn_args=(x,y),
                             fcn_kws={'model':model, 'project':project})
    result = fitter.minimize(method=method)

    if report_fit:
        lmfit.report_fit(result)

    popt=[result.params[name].value for name in parnames]
    popt=np.asarray(popt)
    perr=[result.params[name].stderr for name in parnames]
    perr=np.asarray(popt)

    return popt, perr, result

def residual_spiral(pars, x, y, model=None, project=True):
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
    model_x, model_y = make_spiral(pars,model,project)

    dist = np.zeros_like(x)
    for i in range(dist.shape[0]):
        d = np.sqrt((model_x - x[i])**2 + (model_y - y[i])**2 )
        dist[i] = d.min()

    return dist

def make_spiral(pars,model,project=True):
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

	"""
    parvals = pars.valuesdict()
    theta = np.arange(0,165.*np.pi/180,0.01)

    if model=='RM09':
        r = spiral_RM09(parvals['N'], parvals['B'], parvals['A'], theta)

    r = np.hstack((r[::-1],r[1:]))
    theta = np.hstack((theta[::-1], (theta+np.pi)[1:] ))

    x,y = polar2cart(r, theta)
    z = np.zeros_like(x)

    if project:
        euler = np.array((parvals['e0'],parvals['e1'],parvals['e2']))*np.pi/180
        x_rot, y_rot, z_rot = spiral_pos(x,y,z,euler)
        return x_rot, y_rot
    else:
        return x, y

def spiral_pos(x,y,z,euler):
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
              np.sin(euler[2]) * np.sin(euler[0]) )* x + \
			( np.cos(euler[2]) * np.cos(euler[1]) * np.sin(euler[0]) +
              np.sin(euler[2]) * np.cos(euler[0]) )* y + \
			( -1. * np.cos(euler[2]) * np.sin(euler[1]) ) * z

    y_rot = ( -1. * np.sin(euler[2]) * np.cos(euler[1]) * np.cos(euler[0]) -
              np.cos(euler[2]) * np.sin(euler[0]) )* x + \
			( -1. * np.sin(euler[2]) * np.cos(euler[1]) * np.sin(euler[0]) +
              np.cos(euler[2]) * np.cos(euler[0]) )* y + \
			( np.sin(euler[2]) * np.sin(euler[1]) ) * z

    z_rot = np.sin(euler[1]) * np.cos(euler[0]) * x + \
			np.sin(euler[1]) * np.sin(euler[0]) * y + \
			np.cos(euler[1]) * z

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
    if model == 'RM09':
        parnames = ['N','B','A']
        for i in range(len(parnames)):
            if pinit is None:
                pars.add(parnames[i], value=1.0)
            else:
                pars.add(parnames[i], value=pinit[i])
        if project:
            parnames.extend(['e0','e1','e2'])
            for i in range(len(parnames)):
                if pinit is None:
                    pars.add(parnames[i], value=1.0)
                else:
                    pars.add(parnames[i], value=pinit[i])

    return pars, parnames
