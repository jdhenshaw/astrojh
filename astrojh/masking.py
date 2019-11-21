#==============================================================================#
# masking.py
#==============================================================================#
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude

def circularmask(shape, centre=None, radius=None, wcs=None):
    """
    Accepts a numpy array which includes the shape of the image to be masked,
    additionally accepts a centre location and a radius to help produce the
    mask

    Parameters
    ----------
    shape : numpy array
        numpy array containing the shape of the image (y,x)
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    radius : float (optional)
        radius of the region to mask
    wcs : astropy object
        world coordinate system information

    """
    y = int(shape[0])
    x = int(shape[1])
    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if radius is None:
        radius = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
    mask = dist_from_centre <= radius
    return mask

def circularmask_img(img, centre=None, radius=None, wcs=None):
    """
    Accepts an image to be masked. Additionally accepts a centre location and a
    radius to help produce the mask. Returns a masked image

    Parameters
    ----------
    img : numpy array
        image to be masked
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    radius : float (optional)
        radius of the region to mask
    wcs : astropy object
        world coordinate system information. NB - this is not tested yet

    """
    y = int(np.shape(img)[0])
    x = int(np.shape(img)[1])
    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if radius is None:
        radius = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
    mask = dist_from_centre <= radius
    newimg = np.where(mask==1, img, np.NaN)
    return newimg

def annularmask(shape, centre=None, radius=None, width=None, wcs=None):
    """
    Creates an annular mask of a certain radius and width around a central
    location

    Parameters
    ----------
    shape : numpy array
        numpy array containing the shape of the image (y,x)
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    radius : float (optional)
        radius of the region to mask
    width : float (optional)
        width of the annulus
    wcs : astropy object
        world coordinate system information

    """
    y = int(shape[0])
    x = int(shape[1])
    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if radius is None:
        radius = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    if width is None:
        width=1

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
    mask = (dist_from_centre <= radius+width)&(dist_from_centre >= radius-width)
    return mask

def annularmask_img(img, centre=None, radius=None, width=None, wcs=None):
    """
    Creates an annular mask of a certain radius and width around a central
    location

    Parameters
    ----------
    shape : numpy array
        numpy array containing the shape of the image (y,x)
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    radius : float (optional)
        radius of the region to mask
    width : float (optional)
        width of the annulus
    wcs : astropy object
        world coordinate system information

    """
    y = int(np.shape(img)[0])
    x = int(np.shape(img)[1])
    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if radius is None:
        radius = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    if width is None:
        width=1

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
    mask = (dist_from_centre >= radius-width)&(dist_from_centre <= radius+width)
    newimg = np.where(mask==1, img, np.NaN)
    return newimg

def rectmask(shape, centre=None, width=None, wcs=None):
    """
    Accepts a numpy array which includes the shape of the image to be masked,
    additionally accepts a centre location and a width to help produce the
    mask

    Parameters
    ----------
    shape : numpy array
        numpy array containing the shape of the image (y,x)
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    width : numpy array
        width of the mask - if two dimensions are given a rectangular mask will
        be produced. Should be given in (y,x) if so.
    wcs : astropy object
        world coordinate system information

    """
    y = int(shape[0])
    x = int(shape[1])
    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if width is None:
        width = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.zeros([y,x])

    if np.size(width)==2:
        xp=int(centre[1]+width[1]//2); xn=int(centre[1]-width[1]//2)
        yp=int(centre[0]+width[0]//2); yn=int(centre[0]-width[0]//2)
    else:
        xp=int(centre[1]+width/2); xn=int(centre[1]-width/2)
        yp=int(centre[0]+width/2); yn=int(centre[0]-width/2)

    if xn < 0:
        xn = 0
    if xp > x:
        xp = x
    if yn < 0:
        yn = 0
    if yp > y:
        yp = y

    dist_from_centre[yn:yp,xn:xp]=1.0
    mask = (dist_from_centre == 1.0)

    return mask

def rectmask_img(img, centre=None, width=None, wcs=None):
    """
    Accepts an image to be masked. Additionally accepts a centre location and a
    width to help produce the mask. Returns a masked image

    Parameters
    ----------
    img : numpy array
        image to be masked
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units. If wcs is
        provided the centre value can be given in map units
    width : float (optional)
        width of the square mask
    wcs : astropy object
        world coordinate system information. NB - this is not tested yet

    """
    y = int(np.shape(img)[0])
    x = int(np.shape(img)[1])

    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if width is None:
        width = min(centre[0], centre[1], x-centre[1], y-centre[0])

    if wcs is not None:
        if centre is not None:
            xc, yc = wcs.all_world2pix([centre[1]], [centre[0]], 1)
            centre = [int(yc), int(xc)]

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.zeros([y,x])

    if np.size(width)==2:
        xp=int(centre[1]+width[1]//2); xn=int(centre[1]-width[1]//2)
        yp=int(centre[0]+width[0]//2); yn=int(centre[0]-width[0]//2)
    else:
        xp=int(centre[1]+width//2); xn=int(centre[1]-width//2)
        yp=int(centre[0]+width//2); yn=int(centre[0]-width//2)

    if xn < 0:
        xn = 0
    if xp > x:
        xp = x
    if yn < 0:
        yn = 0
    if yp > y:
        yp = y

    dist_from_centre[yn:yp+1,xn:xp+1]=1.0
    mask = (dist_from_centre == 1.0)
    newimg = np.where(mask==1, img, np.NaN)

    return newimg

def mask_ids(mask, reference=np.array([0,0]), remove_zero=False):
    """
    Reads in a mask and returns the indices where the mask is true relative to
    a reference point

    Parameters
    ----------
    mask : ndarray
        a boolean mask
    reference : array like
        a reference position from which to return the ids. Given as y, x

    """
    idy, idx = np.where(mask==True)
    idx = idx-reference[1]
    idy = idy-reference[0]

    if remove_zero:
        ids = np.array([idy, idx]).T
        zero = np.where((ids[:,0]==0)&(ids[:,1]==0))[0]
        if np.size(zero!=0):
            ids = [ids[i,:] for i in np.arange(len(ids[:,0])) if i != zero]
            ids = np.asarray(ids)
    else:
        ids = np.array([idy, idx]).T

    return ids
