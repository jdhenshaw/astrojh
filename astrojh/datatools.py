#==============================================================================#
# datatools.py
#==============================================================================#
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from scipy.spatial import distance
from scipy.spatial import cKDTree

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
    for i in range(len(maptomodel)):
        datapoint = np.array([data[:,i]])
        distance_to_curve=distance.cdist(datapoint, model.T)[0]
        maptomodel[i]=int(np.where(distance_to_curve==np.min(distance_to_curve))[0])
    return maptomodel

def distance_along_curve(model, origin=0):
    """
    Computes the distance along a curve relative to a position on that curve

    Parameters
    ----------
    model : ndarray
        array of x and y positions of a model
    origin : ndarray (optional)
        index of the origin within the model from which all distances will be
        computed (default=0)
    """
    distance=np.zeros_like(model[0,:], dtype=float)
    originpoint=model[:,origin]
    for i in range(len(model[0,:])):
        distance[i]+=np.linalg.norm(model[:,i]-originpoint)
    return distance
