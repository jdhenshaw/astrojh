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
    data points that lie within this boundary

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
    idxs=np.unique(idxs)

    newdata=data[:,idxs]
    return newdata
