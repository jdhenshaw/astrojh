#==============================================================================#
# statistics.py
#==============================================================================#
import numpy as np
import astropy.units as u
from astropy.units import cds
from astropy.units import astrophys as ap
from .conversions import *
from astropy import constants as const
import scipy.stats as stats
import statsmodels.api as sm
from scipy import signal

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

def fft1d( xarr, yarr, n, d ):
    """
    Accepts two 1D arrays (x, y) a sampling spacing and the number of samples
    and computes one-dimensional discrete Fourier Transform.

    Parameters
    ----------
    xarr : numpy array
        array of x values
    yarr : numpy array
        array of y values
    n : float
        number of samples
    d : float
        sampling spacing (frequency on which x values are measured)

    """
    xf = np.fft.fftfreq(n, d=d)
    yf = np.fft.fft(yarr, n=n)
    xfp = xf[:n//2]
    yfp = np.abs(yf[:n//2])

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
    #print(peak_info)
    xpospeaks = xarr[peak_info[0]]
    if 'prominences' in peak_info[1]:
        ypospeaks = peak_info[1]['prominences']
    return xpospeaks, ypospeaks
