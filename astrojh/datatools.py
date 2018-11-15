#==============================================================================#
# datatools.py
#==============================================================================#
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def create_table(columns, headings, table_name='mytable',
                 outputdir='./', outputfile='mytable.dat',
                 overwrite=False):
    """
    Outputs a simple ascii table given some data and headings

    Parameters
    ----------
    columns : ndarray
        numpy array of data to add to the table
    headings : list
        list of column headings
    table_name : string (optional)
        Name of the table, pretty pointless tbh for the purposes of this code
    outputdir : string (optional)
        name of the output directory
    outputfile : string (optional)
        name of the output file
    overwrite : bool
        Whether or not you want to overwrite a previous table of the same name
    """
    mytable = Table(meta={'name':table_name})

    for i in range(len(headings)):
        mytable[headings[i]] = Column(columns[i,:])

    mytable.write(outputdir+outputfile, format='ascii', overwrite=overwrite)

    return mytable

def interpolate1D(x, y, kind='linear'):
    """
    Perform a simple 1D linear interpolation
    """
    interp = interp1d(x, y, kind=kind)
    interpy = interp(x)

    return interpy

def radial_data_selection(data, model, radius):
    """
    Selects all data within a given distance from a model. Returns array of
    indices that lie within this boundary

    Parameters
    ----------
    data : ndarray
        array of x and y positions of a dataset
    model : ndarray
        array of x and y positions of a model
    radius : float
        search radius for data

    """
    kdtree = cKDTree(data.T)
    idxs=[]
    for i in range(len(model[0,:])):
        coords = model[:,i]
        idx = kdtree.query_ball_point(coords, radius, eps = 0)
        idxs.extend(idx)
    radialselection=np.unique(idxs)
    return radialselection

def map_to_model(data, model):
    """
    The shortest distance between a point and a model is orthogonal. This
    code takes an array of data and finds the closest point in a model curve.
    Returns an array of indices corresponding to the closest matching points in
    the model

    Parameters
    ----------
    data : ndarray
        array of x and y positions of a dataset
    model : ndarray
        array of x and y positions of a model

    """
    maptomodel = np.zeros_like(data[0,:], dtype=int)
    distancetocurve = np.zeros_like(data[0,:], dtype=float)

    for i in range(len(maptomodel)):
        datapoint = np.array([data[:,i]],dtype=float)
        distance_to_curve = distance.cdist(datapoint, model.T)[0]
        distancetocurve[i]=np.min(distance_to_curve)

        idcurve = np.where(distance_to_curve==np.min(distance_to_curve))[0]
        maptomodel[i]=int(idcurve)
    return maptomodel, distancetocurve

def distance_along_curve(model, origin=0):
    """
    Computes the cumulative distance along a curve relative to a position on
    that curve

    Parameters
    ----------
    model : ndarray
        array of x and y positions of a model
    origin : ndarray (optional)
        index of the origin within the model from which all distances will be
        computed (default=0)
    """
    distance=np.zeros(np.size(model[0,:]), dtype=float)

    if origin==-1:
        newmodel=model[:,::-1]
    else:
        newmodel=model

    for i in range(len(model[0,:])):
        if i != 0:
            distance[i]=np.linalg.norm(newmodel[:,i]-newmodel[:,i-1])

    cumulativedistance=np.cumsum(distance)

    if origin==-1:
        cumulativedistance=cumulativedistance[::-1]

    return cumulativedistance

def average_along_curve(distances, data, weights=None):
    """
    Computes the average data quantity for each distance along a model. Returns
    unique distance and mean data

    Parameters
    ----------
    distances : ndarray
        distance along model curve
    data : ndarray
        corresponding data values
    weights : ndarray
        optional weighting for averaging

    """
    uniquedistances = np.unique(distances)
    meandata = []
    for _d in uniquedistances:
        ids = np.where(distances==_d)[0]
        if np.size(ids) != 0.0:
            datasubsample = data[ids]
            if weights is not None:
                weightssubsample = weights[ids]
                meandata.append(np.average(datasubsample, weights=weightssubsample))
            else:
                meandata.append(np.average(datasubsample))
    return uniquedistances,np.asarray(meandata)
