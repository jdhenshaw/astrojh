#==============================================================================#
# statistics.py
#==============================================================================#
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy import signal
from scipy.spatial import distance
from scipy.interpolate import interp1d

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
        interp = interp1d(xarr, yarr, kind=method)
        yarr = interp(xnew)

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
    tol = meandiff/2.
    return tol
