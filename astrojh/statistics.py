#==============================================================================#
# statistics.py
#==============================================================================#
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy import signal
from scipy.spatial import distance
from scipy.interpolate import interp1d
from .datatools import interpolate1D
from .masking import *
import itertools
from astropy.utils.console import ProgressBar
from .parallel_map import *
from scipy.ndimage.interpolation import shift

def basic_info( arr, sarr=None ):
    """
    Accepts an array of data and prints out some basic but useful information

    Parameters
    ----------
    arr : numpy array
        An array of values
    sarr : numpy array (optional)
        A corresponding array of the uncertainties

    """
    # get the normalised data
    arr = arr[~np.isnan(arr)]
    mu, std = stats.norm.fit(arr)
    normed_data=(arr-mu)/std
    # tests for deviations from Gaussianity
    skewtest = np.asarray(stats.mstats.skewtest(normed_data, axis=0))
    kurttest = np.asarray(stats.mstats.kurtosistest(normed_data, axis=0))
    kstest = np.asarray(stats.kstest(normed_data, 'norm'))
    adtest = np.asarray(sm.stats.diagnostic.normal_ad(normed_data, axis=0))
    normaltest = np.asarray(stats.normaltest(normed_data))

    if sarr is None:
        print('===============================================================')
        print('                      Basic Statistics                         ')
        print('===============================================================')
        print("")
        print('minimum:  ', np.nanmin(arr))
        print('maximum:  ', np.nanmax(arr))
        print('mean:     ', np.nanmean(arr))
        print('median:   ', np.nanmedian(arr))
        print('std:      ', np.nanstd(arr))
        print('IQR       ', stats.iqr(arr))
        print('25%       ', np.percentile(arr, 25))
        print('75%       ', np.percentile(arr, 75))
        print("")
        print('---------------------------------------------------------------')
        print('                         Moments                               ')
        print('---------------------------------------------------------------')
        print("")
        print('Skew:     ', stats.skew(arr))
        print('Kurtosis: ', stats.kurtosis(arr, fisher=False))
        print("")
        print('---------------------------------------------------------------')
        print('                        Stat Tests                             ')
        print('---------------------------------------------------------------')
        print("")
        print('Skew stat (pvalue):             ', skewtest[0],   '(',
                                                  skewtest[1],   ')',)
        print('Kurtosis stat (pvalue):         ', kurttest[0],   '(',
                                                  kurttest[1],   ')',)
        print('KS stat (pvalue):               ', kstest[0],     '(',
                                                  kstest[1],     ')',)
        print('Anderson-Darling stat (pvalue): ', adtest[0],     '(',
                                                  adtest[1],     ')',)
        print('D Agostino stat (pvalue):       ', normaltest[0], '(',
                                                  normaltest[1],  ')',)
        print("")
        print('===============================================================')
    else:
        print('===============================================================')
        print('                      Basic Statistics                         ')
        print('===============================================================')
        print("")
        print('minimum:  ', np.nanmin(arr), '+/-', sarr[arr==np.nanmin(arr)][0])
        print('maximum:  ', np.nanmax(arr), '+/-', sarr[arr==np.nanmax(arr)][0])
        print('mean:     ', np.nanmean(arr), '+/-', \
                                              stats.sem(arr, axis=None, ddof=0))
        print('median:   ', np.nanmedian(arr))
        print('std:      ', np.nanstd(arr))
        print('IQR       ', stats.iqr(arr))
        print('25%       ', np.percentile(arr, 25))
        print('75%       ', np.percentile(arr, 75))
        print("")
        print('---------------------------------------------------------------')
        print('                         Moments                               ')
        print('---------------------------------------------------------------')
        print("")
        print('Skew:     ', stats.skew(arr))
        print('Kurtosis: ', stats.kurtosis(arr, fisher=False))
        print("")
        print('---------------------------------------------------------------')
        print('                        Stat Tests                             ')
        print('---------------------------------------------------------------')
        print("")
        print('Skew stat (pvalue):             ', skewtest[0],   '(',
                                                  skewtest[1],   ')',)
        print('Kurtosis stat (pvalue):         ', kurttest[0],   '(',
                                                  kurttest[1],   ')',)
        print('KS stat (pvalue):               ', kstest[0],     '(',
                                                  kstest[1],     ')',)
        print('Anderson-Darling stat (pvalue): ', adtest[0],     '(',
                                                  adtest[1],     ')',)
        print('D Agostino stat (pvalue):       ', normaltest[0], '(',
                                                  normaltest[1],  ')',)
        print("")
        print('===============================================================')

def fft1d( xarr, yarr, nsamples=None, sampling=None, irregular=False,
           method='linear' ):
    """
    Accepts two 1D arrays (x, y) a sampling spacing and the number of samples
    and computes one-dimensional discrete Fourier Transform.

    Parameters
    ----------
    xarr : numpy array
        array of x values
    yarr : numpy array
        array of y values
    nsamples : float (optional)
        number of samples (default = len(x))
    sampling : float (optional)
        sampling spacing (frequency on which x values are measured)
    irregular : bool (optional)
        if True and the x axis is irregularly sampled - use scipy interpolate to
        place on a regularly spaced grid
    method : string (optional)
        method for scipy.interp1d - NB: keyword 'kind' (default = linear)

    """
    if nsamples is None:
        nsamples = len(xarr)
    if sampling is None:
        sampling = np.abs(np.min(xarr)-np.max(xarr))/nsamples
    if irregular is True:
        xnew = np.linspace(np.min(xarr), np.max(xarr), nsamples)
        yarr = interpolate1D(xnew, yarr, kind=method)

    xf = np.fft.fftfreq(nsamples, d=sampling)
    yf = np.fft.fft(yarr-yarr.mean(), n=nsamples)
    xfp = xf[:nsamples//2]
    yfp = np.abs(yf[:nsamples//2])

    return xfp, yfp

def peakfinder( xarr, yarr, **kwargs):
    """
    Accepts two 1D arrays and looks for the peaks

    Parameters
    ----------
    xarr : ndarray
        array of x values
    yarr : ndarray
        array of y values within which you want to find peaks

    """
    peak_info = signal.find_peaks( yarr, **kwargs )
    xpospeaks = xarr[peak_info[0]]
    if 'prominences' in peak_info[1]:
        ypospeaks = yarr[peak_info[0]]
    else:
        ypospeaks=[]
    return xpospeaks, ypospeaks

def structurefunction_1d(x, y, order=2, nsamples=None, spacing='linear',
                         irregular=False):
    """
    Computes 1D structure function

    Parameters:
    -----------
    x : ndarray
        array of x values - must be monotonically increasing
    y : ndarray
        array of y values
    order : number (optional)
        order of the structure function (default = the size of the x array)
    nsamples : number (optional)
        Frequency over which to sample the structure function (default=2)
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    irregular : Bool (optional)
        if the spacing of the xaxis is irregular this option will incorporate
        a tolerance into the distance computation. (Default==False)
    """

    if nsamples is None:
        nsamples = np.size(x)

    # Compute spacing between 1 and np.size(x)//2 elements.
    if spacing=='linear':
        structx = np.linspace(1, (np.size(x)-1)//2, num=nsamples)
        structx = np.around(structx, decimals=0)
        structx = np.unique(structx)
    elif spacing=='log':
        structx = np.logspace(np.log10(1), np.log10((np.size(x)-1)//2), num=nsamples)
        structx = np.around(structx, decimals=0)
        structx = np.unique(structx)

    structx=structx.astype('int')
    structy=[]

    # if on an irregularly spaced grid then we need to work with
    # absolute distances and incorporate a tolerance level
    tolerance = compute_tolerance(x)

    # For each distance element over which to compute the SF, compute distance
    # to all pixels - select the relevant ones and compute SF
    for _x in structx:
        diff=[]
        for i in range(len(x)):
            # Find distance between current pixel and all other pixels
            if not irregular:
                distances = compute_distance(i,np.arange(len(x)))
                # select the relevant distance
                id = np.where(distances==_x)[0]
            else:
                distances = compute_distance_irregular(x[i], x)
                id = np.where((distances>=_x-tolerance)&
                              (distances<=_x+tolerance))[0]

            if np.size(id)!=0:
                originval=y[i]
                # SF computation
                diff.extend(np.abs(originval-y[id])**order)

        diff=np.asarray(diff)
        # SF is the average measured on a given size scale
        structy.append(np.mean(diff))

    return structx, np.power(structy, (1./order))

def structurefunction_2d(img, order=2, max_size=None, nsamples=None,
                         spacing='linear', width=1, njobs=1):
    """
    Computes 2D structure function

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    order : number (optional)
        order of the structure function (default=2)
    max_size : number (optional)
        Maximum extent (in pixels) of region over which to compute the structure
        function. Largest scale for SF computation would be max_size/2
    nsamples : number (optional)
        Frequency over which to sample the structure function
        (default = the size of the x array)
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    width : number (optional)
        width of the annulus over which to compute the SF
    njobs : number (optional)
        parallel processing
    """
    # compute the lags for the SF computation
    lags = compute_lags(img,max_size=max_size,nsamples=nsamples,spacing=spacing)
    structy=[]
    errstructy=[]

    # Loop through the lags
    for lagvalue in ProgressBar(lags):
        pi = compute_parameterincrements(img,lagvalue,width=width,njobs=njobs)

        # Estimate number of independent measurements = maparea/_x_area - where
        # map area is given by the number of non NaN pixels in the map and the
        # _x_area is just the area enclosed by the annulus
        n_indep = np.size(img[~np.isnan(img)])/(np.pi * lagvalue**2)

        # SF is the average measured on a given size scale
        structy.append( np.mean( np.abs(pi)**order ) )
        # the uncertainty on the measurement is taken as the standard error on
        # the mean, taking into account independent areas
        std = np.std( np.abs(pi)**order )
        errstructy.append(std/np.sqrt(n_indep))

    return lags,structy,errstructy

def compute_lags(img, max_size=None, nsamples=None, spacing='linear'):
    """
    Computes lags for various computations e.g. structure functionss

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    max_size : number (optional)
        Maximum extent (in pixels) of region over which to compute the structure
        function. Largest scale for SF computation would be max_size/2
    nsamples : number (optional)
        Frequency over which to sample the structure function
        (default = the size of the x array)
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    """

    # Find the maximum half-width of the image excluding any NaNs
    if np.any(np.isnan(img)):
        idx,idy = np.where(~np.isnan(img))
        if (np.size(idx)==0) or (np.size(idy)==0):
            raise ValueError("Image is exclusively made up of NaNs")
        else:
            extentx=np.abs(np.min(idx)-np.max(idx))
            extenty=np.abs(np.min(idy)-np.max(idy))
            extent = np.min([extentx, extenty])
    else:
        extent = np.min(np.shape(img))

    if max_size is not None:
        extent=max_size

    # set the number of samples
    if nsamples is None:
        nsamples = np.min(np.shape(img))

    # Compute spacing between 1 and np.size(x)//2 elements.
    if spacing=='linear':
        lags = np.linspace(1, ((extent//2)-1)//2, num=nsamples)
        lags = np.around(lags, decimals=0)
        lags = np.unique(lags)
    elif spacing=='log':
        lags = np.logspace(np.log10(1),np.log10(((extent//2)-1)//2),num=nsamples)
        lags = np.around(lags, decimals=0)
        lags = np.unique(lags)

    return lags

def compute_parameterincrements( img, lagvalue, width=1, njobs=1 ):
    """
    Parallelised computation of parameter increments. Returns an array of values
    corresponding to the parameter increment given as

    delta P = P(r)-P(r+l)

    where P is the parameter you're interested in, r is some position in the map
    and l is some lag value

    Computation is performed by shifting the map and subtracting the whole
    image which makes for extremely fast computation

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    lagvalue : number
        distance over which to compute structure function
    width : number (optional)
        width of the annulus over which to compute the SF
    njobs : number (optional)
        parallel processing
    """

    reference = [np.shape(img)[0]//2, np.shape(img)[1]//2]
    # create a mask
    mask = annularmask(np.shape(img),centre=reference,radius=lagvalue,width=width)
    maskids = mask_ids(mask, reference=reference, remove_zero=True)

    # Parallel processing of parameter increments
    args = [img,lagvalue]
    inputvalues = [[maskid]+args for maskid in maskids ]
    pivals = parallel_map(parallelpi, inputvalues, numcores=njobs)
    flatpi = [value for pivs in pivals for value in pivs]
    # Convert to an array
    flatpi = np.asarray(flatpi)

    return flatpi

def parallelpi(inputvalues):
    """
    Parallelised computation of parameter increments. Returns an array of values
    corresponding to the parameter increment given as

    delta P = P(r)-P(r+l)

    where P is the parameter you're interested in, r is some position in the map
    and l is some lag value

    Computation is performed by shifting the map and subtracting the whole
    image which makes for extremely fast computation

    Parameters
    ----------
    inputvalues : list
        Combination of img, radius, and ids

    """
    #inputvalues = inputvalues[0]
    dumbvalue = -1e10
    # unpack inputs
    maskid, img, radius = inputvalues
    maskidx = maskid[1]
    maskidy = maskid[0]

    # create a copy of the image and fill it with stupid values instead of NaN
    imgcopy = np.copy(img)
    imgcopy[np.isnan(imgcopy)]=dumbvalue

    #begin by shifting the image
    imgshift = shift(imgcopy, [maskidy,maskidx], cval=dumbvalue)
    # Get rid of the dumb values
    imgshift[((imgshift>=(dumbvalue-100.))&(imgshift<=(dumbvalue+100.)))]=np.NaN
    # compute difference between the two images
    imgdiff = img-imgshift
    # remove zeros and nans values
    imgdiff = imgdiff[(imgdiff != 0.0) & (~np.isnan(imgdiff))]

    return imgdiff

def compute_distance(id, ids):
    """
    Compute the distance between the current value and all other values

    Parameters
    ----------
    id : number
        Current index
    ids : ndarray
        all other ids
    """
    return np.abs(id-ids)

def compute_distance_irregular(xpos, x):
    """
    Compute the distance between the current value and all other values on an
    irregularly spaced grid

    Parameters
    ----------
    xpos : number
        Current x position
    x : ndarray
        array of x values
    """
    return np.abs(xpos-x)

def compute_tolerance(x):
    """
    Computes tolerance level for distances in SF analysis. Tolerance is set to
    half the mean spacing between xvalues

    Parameters
    ----------
    x : ndarray
        array of x values

    """
    diffarr = np.diff(x)
    meandiff = np.mean(diffarr)
    tol = np.abs(meandiff/2.)
    return tol
