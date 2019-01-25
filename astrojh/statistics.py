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
from scipy.ndimage import rotate
import os
from astropy.io import fits
from scipy.stats.kde import gaussian_kde
from scipy.signal import argrelmin
from scipy.signal import argrelmax

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

def compute_pdf(arr, nsamples, bw_method=None ):
    """
    Gaussian kernel density estimation

    Parameters
    ----------
    arr : ndarray
        array of values
    nsamples : number
        number of samples for pdf
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be
        ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar,
        this will be used directly as kde.factor. If a callable, it should
        take a gaussian_kde instance as only parameter and return a scalar.
        If None (default), nothing happens; the current kde.covariance_factor
        method is kept.

    """
    kde = gaussian_kde(arr)
    kde.set_bandwidth(bw_method=kde.factor * bw_method)
    x = np.linspace(np.nanmin(arr),np.nanmax(arr), num=nsamples)
    pdf = kde(x)
    return x, pdf

def fft1d( xarr, yarr, nsamples=None, sampling=None, irregular=False,
           method='linear', subtract_mean=False ):
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
    subtract_mean : bool (optional)
        Removes 'DC' component of FT

    """
    if nsamples is None:
        nsamples = len(xarr)
    if sampling is None:
        sampling = np.abs(np.min(xarr)-np.max(xarr))/(nsamples-1)
    if irregular is True:
        xnew = np.linspace(np.min(xarr), np.max(xarr), nsamples)
        yarr = interpolate1D(xnew, yarr, kind=method)

    xf = np.fft.fftfreq(nsamples, d=sampling)
    if subtract_mean:
        yf = np.fft.fft(yarr-yarr.mean(), n=nsamples)
    else:
        yf = np.fft.fft(yarr, n=nsamples)
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

def sf1d(x, y, order=2, nsamples=None, spacing='linear', irregular=False,
         method='linear'):
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

    nans = np.isnan(y)
    if np.any(nans):
        x = [x[i] for i in range(len(x)) if not nans[i]]
        y = [y[i] for i in range(len(y)) if not nans[i]]
        x = np.asarray(x)
        y = np.asarray(y)
        irregular=True

    if nsamples is None:
        nsamples = np.size(x)
    if irregular is True:
        xnew = np.linspace(np.min(x), np.max(x), nsamples)
        y = interpolate1D(xnew, y, kind=method)

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
    errstructy=[]

    # if on an irregularly spaced grid then we need to work with
    # absolute distances and incorporate a tolerance level
    tolerance = compute_tolerance(x)

    # For each distance element over which to compute the SF, compute distance
    # to all pixels - select the relevant ones and compute SF
    for _x in structx:
        diff=[]
        for i in range(len(x)):
            # Find distance between current pixel and all other pixels
            distances = compute_distance(i,np.arange(len(x)))
            # select the relevant distance
            id = np.where(distances==_x)[0]
            if np.size(id)!=0:
                originval=y[i]
                # SF computation
                diff.extend(np.abs(originval-y[id])**order)

        n_indep = np.size(diff)/_x
        diff=np.asarray(diff)
        # SF is the average measured on a given size scale
        structy.append(np.mean(diff))
        std = np.nanstd(diff)
        errstructy.append(std/np.sqrt(n_indep))

    return np.asarray(structx), np.asarray(structy), np.asarray(errstructy)

def get_minima(x, y, axis=0, order=1, mode='clip'):
    """
    Returns x and y values of local minima in an array

    Parameters:
    -----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    axis : number (optional)
        Axis over which to select from data. Default is 0.
    order : number (optional)
        How many points on each side to use for the comparison to consider
        comparator(n, n+x) to be True.
    mode : str (optional)
        How the edges of the vector are treated. Available options are ‘wrap’
        (wrap around) or ‘clip’ (treat overflow as the same as the last
        (or first) element). Default ‘clip’. See numpy.take

    """
    idminima = argrelmin(y, axis=axis, order=order, mode=mode)
    xminima = x[idminima]
    yminima = y[idminima]
    return xminima, yminima

def get_maxima(x, y, axis=0, order=1, mode='clip'):
    """
    Returns x and y values of local minima in an array

    Parameters:
    -----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    axis : number (optional)
        Axis over which to select from data. Default is 0.
    order : number (optional)
        How many points on each side to use for the comparison to consider
        comparator(n, n+x) to be True.
    mode : str (optional)
        How the edges of the vector are treated. Available options are ‘wrap’
        (wrap around) or ‘clip’ (treat overflow as the same as the last
        (or first) element). Default ‘clip’. See numpy.take

    """
    idmaxima = argrelmax(y, axis=axis, order=order, mode=mode)
    xmaxima = x[idmaxima]
    ymaxima = y[idmaxima]
    return xmaxima, ymaxima

def remove_resonances(x, y, atol=0.1, max_sep=None, **kwargs):
    """
    First attempt at coding something up to remove resonances which appear in
    structure functions. Returns an array of minima cleaned of resonances.

    Further notes
    -------------
    It is possible to set a maximum separation length. Any spacings below this
    length scale will not be evaluated as resonances. Only those above the
    length scale. For a chain of cores in a filament this could be set to the
    maximum observed spacing between adjacent cores. Anything below that
    separation may be a genuine feature in the structure function and everything
    above that separation should be included in the resonance evaluation.

    Parameters
    ----------
    x : ndarray
        array of x values
    y : ndarray
        array of y values
    atol : number (optional)
        tolerance value for acceptance as a resonance
    max_sep : number (optional)
        estimate of the maximum separation between features
    """

    # firstly find the minima and maxima
    xmin, ymin = get_minima(x, y, **kwargs)

    # set up the remove array - this will be gradually populated telling us
    # which minima should be removed
    remove_res = np.zeros(len(xmin), dtype='bool')
    nmeas = len(xmin)

    # We are going to divide each quantity in the xmin array by all quantities
    # in that array to establish if there are resonances within some tolerance
    # value. So set up arrays of size nmeas x nmeas to hold this information
    res_matrix = np.zeros([nmeas, nmeas], dtype='float')
    rem_matrix = np.zeros([nmeas, nmeas], dtype='bool')

    # These are the resonant factors (skip 1 since we want to keep those). The
    # upper value is largely arbitrary, but you want to make sure enough values
    # are selected
    factor_array = np.arange(2, nmeas+1)

    # cycle through the values
    for i in range(nmeas):
        # divide all values by all values
        for j in range(nmeas):
            factor = xmin[j]/xmin[i]
            res_matrix[i,j] = factor

        # create an array for each value of xmin which we will populate
        rem_matrix_indiv = np.zeros([len(factor_array), nmeas], dtype='bool')
        # cycle through the possible factors and test to see if any resonances
        # exist
        for j in range(len(factor_array)):
            # the value we are going to compare
            comparison_value = factor_array[j]
            # the array of values divided by the position of the current minimum
            value_array = res_matrix[i,:]
            # fill this boolean array to say if any resonances exist
            rem_matrix_indiv[j,:] = np.isclose(value_array, comparison_value,
                                               atol=atol)

        # Now update the rem matrix to establish if any of the values should be
        # removed
        for j in range(nmeas):
            if np.any(rem_matrix_indiv[:,j]):
                rem_matrix[i,j]=True

    # Now update the remove_res array
    for i in range(len(xmin)):
        if np.any(rem_matrix[:,i]):
            remove_res[i]=True


    if max_sep is not None:
        # if there are features below this length scale we want to start our
        # evaluation with the largest separation that sits below this limit.
        # Anything smaller than this will be automatically assumed to not be
        # a resonant feature.
        # identify all minima below this length scale
        idlim = np.where(xmin < max_sep)[0]
        remove_res[idlim]=False

    xmin = xmin[~remove_res]
    ymin = ymin[~remove_res]

    return xmin, ymin

def sf1d_rot(img, angle_range=None, stepsize=None, order=2, method='linear'):
    """
    Rotates an image and computes 1D structure functions along x and y
    dimensions. The idea is to look for anisotropy in certain quantities, e.g.
    the velocity field

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    """
    from .datatools import create_table
    headings = ['angle', 'x', 'y']
    outputdir = './sfangle/x/'
    outputprefix = 'sf_'

    # Set a value for masked regions
    dumbvalue=np.NaN
    # copy the image before rotation
    imgcopy = np.copy(img)
    imgcopy[np.isnan(imgcopy)]=dumbvalue

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.0, 10.0))
    # plt.subplots_adjust(bottom=0.2,left=0.15,top=0.9,hspace=0.2, wspace=0.2)

    angles=[]
    angles2=[]
    sfx = []
    sfx2 = []
    sfy = []
    sfy2 = []
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    for i in np.arange(angle_range[0], angle_range[1], stepsize):
        outputname = outputprefix+str(i)+'.dat'
        # rotate the image - do not interpolate otherwise things get funky at
        # at the edges (order = 0)
        imgrot = rotate(imgcopy, i, order=0, cval=np.NaN)
        # start by compressing the information into 1d
        x, img_x, y, img_y = flattento1d(imgrot)
        # Now compute the 1d structure function in both dimensions
        sfx_x, sfx_y = sf1d(x, img_x, order=order)
        sfy_x, sfy_y = sf1d(y, img_y, order=order)

        avals = np.ones(len(sfx_x))*i
        avals2 = np.ones(len(sfy_x))*i

        angles.extend(avals)
        angles2.extend(avals2)
        sfx.extend(sfx_x)
        sfx2.extend(sfy_x)
        sfy.extend(sfx_y)
        sfy2.extend(sfy_y)

        # ax.plot(sfy_x, sfy_y)
        # plt.pause(0.01)

    table = create_table(np.array([angles, sfx, sfy]), headings, outputdir=outputdir,
                        outputfile='testx.dat', overwrite=True)
    table = create_table(np.array([angles2, sfx2, sfy2]), headings, outputdir=outputdir,
                        outputfile='testy.dat', overwrite=True)
    plt.show()
    return 0,0,0,0

def flattento1d(im):
    """
    Average an image along the x and y axis to produce a 1d array. Returns
    x and y values as well as the averaged image values along both dimensions

    Parameters:
    -----------
    img : ndarray
        2D array - an image containing the data
    """
    # import matplotlib.pyplot as plt
    # If any of the image values are NaNs (they should be) then we need to
    # establish the min and max extent in each dimension
    if np.any(np.isnan(im)):

        idy,idx = np.where(~np.isnan(im))
        if (np.size(idx)==0) or (np.size(idy)==0):
            raise ValueError("Image is exclusively made up of NaNs")
        else:
            # generate the x and y values
            x = np.arange(np.min(idx), np.max(idx)+1)
            y = np.arange(np.min(idy), np.max(idy)+1)
            # create empty arrays to hold the mean values
            img_x = np.zeros(len(x))
            img_y = np.zeros(len(y))

            # compute the averages - TODO: careful here? arrays should be
            # regular but there might be holes in the data - should be ignored
            # by nanmean but need to test
            for i in range(len(x)):
                img_x[i]=np.nanmean(im[:,x[i]])
            for i in range(len(y)):
                img_y[i]=np.nanmean(im[y[i],:])

            id = np.where(np.isnan(img_x))
            img_x[id]=np.nanmean(img_x)
            id = np.where(np.isnan(img_y))
            img_y[id]=np.nanmean(img_y)
    else:
        x, y = np.arange(np.shape(im)[1]), np.arange(np.shape(im)[0])
        img_x, img_y= np.nanmean(img, axis=0), np.nanmean(img, axis=1)

    # ax.imshow(im, origin='lower')
    # plt.pause(0.01)

    return x, img_x, y, img_y

def PImaps(img, header, scale_range=None, stepsize=None, spacing='linear', width=1,
           njobs=1, outputdir='./', filenameprefix='param_increments',
           return_map=True, write_fits=True, write_stddev=False,
           write_nmeas=False, sf=False, order=2):
    """
    Computes map of parameter increments

    delta P = P(r)-P(r+l)

    where P is the parameter you're interested in, r is some position in the map
    and l is some lag value

    Will create a series of maps for each lag value as well as standard
    deviation and n measurement maps (if requested). These will be output as
    fits files with the same header as the image.

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    header : FITS header
        FITS header for the image
    scale_range : array like (optional)
        range over which to compute the lags. Should be given as
        scale_range=[lower, upper]
    stepsize : number (optional)
        stepsize over which to compute the lags.
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    width : number (optional)
        width of the annulus over which to compute the SF
    njobs : number (optional)
        parallel processing
    outputdir : string (optional)
        output directory for the images (default = current directory )
    filenameprefix : string (optional)
        prefix for the output file names - will be followed by the lag size and
        '.fits'
    return_map : bool (optional)
        If true, returns a map of the parameter increments. If false - returns
        a 1D array
    write_fits : bool (optional)
        If you wish to write the output to fits format (default = True)
    write_stddev : bool (optional)
        creates a stddev map as well as the PImap (default = False)
    write_nmeas : bool (optional)
        creates an n measurement map as well as the PImap (default = False)
    sf : bool (optional)
        used for creating a structure function map (default = False)
    order : number (optional)
        used for creating a structure function map (default = 2)
    """
    # compute the lags
    lags = compute_lags(img, scale_range=scale_range, stepsize=stepsize,
                        spacing=spacing)

    pilist = []
    # Loop through the lags
    for lagvalue in ProgressBar(lags):
        pi = compute_parameterincrements(img,lagvalue,width=width,njobs=njobs,
                                         return_map=return_map,
                                         sf=sf, order=order)
        # add it to the list
        pilist.append(pi)

        if write_fits:
            # check to see if outdirectory exists and if not, create it
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            hdu = fits.PrimaryHDU(pi[0], header=header)
            hdu.writeto(outputdir+filenameprefix+'_'+str(lagvalue)+'.fits',
                        overwrite=True)

            if write_stddev:
                hdu = fits.PrimaryHDU(pi[1], header=header)
                hdu.writeto(outputdir+filenameprefix+'_'+str(lagvalue)+'_stddev.fits',
                            overwrite=True)
            if write_nmeas:
                hdu = fits.PrimaryHDU(pi[2], header=header)
                hdu.writeto(outputdir+filenameprefix+'_'+str(lagvalue)+'_nmeas.fits',
                            overwrite=True)

    # Return the combined array
    piarr = np.asarray(pilist)

    return lags, piarr

def compute_parameterincrements( img, lagvalue, width=1, njobs=1,
                                 return_map=False, sf=False, order=2 ):
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
    sf : bool (optional)
        used for creating a structure function map (default = False)
    order : number (optional)
        used for creating a structure function map (default = 2)
    """

    reference = [np.shape(img)[0]//2, np.shape(img)[1]//2]

    # create a mask
    mask = annularmask(np.shape(img),centre=reference,radius=lagvalue,width=width)
    maskids = mask_ids(mask, reference=reference, remove_zero=True)

    # Parallel processing of parameter increments
    args = [img,lagvalue,return_map]
    inputvalues = [[maskid]+args for maskid in maskids ]
    pimaps = parallel_map(parallelpi, inputvalues, numcores=njobs)

    if return_map:
        maps = np.asarray(pimaps)
        # create an empty array to hold the maps
        nmap = np.zeros(np.shape(maps[0]), dtype='float')
        # create the average and standard deviation maps
        if not sf:
            avmap = np.nanmean(maps, axis=0)
            stdmap = np.nanstd(maps, axis=0)
        else:
            avmap = np.nanmean((np.abs(maps))**order, axis=0)
            stdmap = np.nanstd((np.abs(maps))**order, axis=0)

        # create nmeasurement map

        # loop over each map
        for i in range(len(maps[:,0,0])):
            # create a map of ones to add to the n map
            nmapindiv=np.ones(np.shape(maps[i]), dtype='float')
            # identify where the maps have infinite values
            idnan = np.where(np.isnan(maps[i]))
            # set those locations = 0.0
            nmapindiv[idnan] = 0.0
            # add maps together
            nmap+=nmapindiv

        # bundle everything up to return
        pi = np.array([avmap, stdmap, nmap])
    else:
        pivals = [value for pivs in pimaps for value in pivs]

        if sf:
            # Estimate number of independent measurements
            outerradius = lagvalue+width
            innerradius = lagvalue-width
            annulusarea = (np.pi * ((outerradius**2)-(innerradius**2)))
            # independent measures
            n_indep = np.size(img[~np.isnan(img)])/annulusarea

            sf = np.nanmean( np.abs(pivals)**order )
            # the uncertainty on the measurement is taken as the standard error on
            # the mean, taking into account independent areas
            std = np.nanstd( np.abs(pivals)**order )
            errsf = std/np.sqrt(n_indep)

            pivals = [sf,std,errsf]

        # Convert to an array
        pi = np.asarray(pivals)

    return pi

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
    maskid, img, radius, return_map = inputvalues
    maskidx = maskid[1]
    maskidy = maskid[0]

    # create a copy of the image and fill it with stupid values instead of NaN
    imgcopy = np.copy(img)
    imgcopy[np.isnan(imgcopy)]=dumbvalue

    # begin by shifting the image
    imgshift = shift(imgcopy, [maskidy,maskidx], cval=dumbvalue)
    # Get rid of the dumb values
    iddumb = np.where((imgshift>=(dumbvalue-200.))&
                      (imgshift<=(dumbvalue+200.)))

    imgshift[iddumb]=np.NaN
    # compute difference between the two images
    imgdiff = img-imgshift

    if not return_map:
        imgdiff=imgdiff[~np.isnan(imgdiff)]

    return imgdiff

def sf2d(img, order=2, scale_range=None, stepsize=None, spacing='linear', width=1,
         njobs=1):
    """
    Computes 2D structure function

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    order : number (optional)
        order of the structure function (default=2)
    scale_range : array like (optional)
        range over which to compute the lags. Should be given as
        scale_range=[lower, upper]
    stepsize : number (optional)
        stepsize over which to compute the lags.
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    width : number (optional)
        width of the annulus over which to compute the SF
    njobs : number (optional)
        parallel processing
    """
    lags, sf = PImaps(img, None, scale_range=scale_range, stepsize=stepsize,
                      spacing=spacing, width=width, njobs=njobs,
                      return_map=False, write_fits=False, sf=True, order=order)

    return lags, sf[:,0], sf[:,2]

def sf2d_maps(img, header, order=2, scale_range=None, stepsize=None,
              spacing='linear', width=1, njobs=1,
              outputdir='./', filenameprefix='sf',
              return_map=True, write_fits=True, write_stddev=False,
              write_nmeas=False):
    """
    Computes 2D structure function

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    order : number (optional)
        order of the structure function (default=2)
    scale_range : array like (optional)
        range over which to compute the lags. Should be given as
        scale_range=[lower, upper]
    stepsize : number (optional)
        stepsize over which to compute the lags.
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    width : number (optional)
        width of the annulus over which to compute the SF
    njobs : number (optional)
        parallel processing
    outputdir : string (optional)
        output directory for the images (default = current directory )
    filenameprefix : string (optional)
        prefix for the output file names - will be followed by the lag size and
        '.fits'
    return_map : bool (optional)
        If true, returns a map of the parameter increments. If false - returns
        a 1D array
    write_fits : bool (optional)
        If you wish to write the output to fits format (default = True)
    write_stddev : bool
        creates a stddev map as well as the PImap (default = False)
    write_nmeas : bool
        creates an n measurement map as well as the PImap (default = False)
    """
    lags, sf = PImaps(img, None, scale_range=scale_range, stepsize=stepsize,
                      spacing=spacing, width=width, njobs=njobs,
                      return_map=True,write_fits=write_fits,outputdir=outputdir,
                      filenameprefix=filenameprefix, write_stddev=write_stddev,
                      write_nmeas=write_nmeas, sf=True, order=order)

    return lags, sf

def compute_lags(img, scale_range=None, stepsize=None, spacing='linear'):
    """
    Computes lags for various computations e.g. structure functionss

    Parameters
    ----------
    img : ndarray
        2D array - an image containing the data for which you would like to
        compute the structure function
    scale_range : array like (optional)
        range over which to compute the lags. Should be given as
        scale_range=[lower, upper]
    stepsize : number (optional)
        stepsize over which to compute the lags.
    spacing : string
        linear or logarithmic spacing of the xdistances over which to compute SF
    """

    # Select the upper and lower limits for fitting
    if scale_range is not None:
        if np.size(scale_range)!=2:
            raise ValueError("Please give both a lower and upper limit in the form scale_range=[low,upp]")
        else:
            low=scale_range[0]
            upp=scale_range[1]
    # If the user does not provide a range - select the maximum according to
    # either the map or data
    else:
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
        # Lags are given as radii so we want the maximum extent possible / 4
        low=1
        upp=extent//4

    # set the number of samples
    if stepsize is None:
        stepsize = 1

    if spacing=='linear':
        lags = np.arange(low, upp, stepsize)
        lags = np.unique(lags)
    elif spacing=='log':
        lags = np.logspace(low, upp, stepsize)
        lags = np.unique(lags)

    return lags

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
