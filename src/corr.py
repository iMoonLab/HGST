import numpy as np
import scipy.io as sio
import os
import nilearn.connectome as connectome
def compute_correlation_matrix(timeseries):
    """
    Compute the correlation matrix from timeseries data.
    :param timeseries: NumPy array of shape (timepoints, regions), 
                       representing the time series data of brain regions.
    :return: correlation matrix of shape (regions, regions)
    """
    timeseries = timeseries.T
    correlation_matrix = np.corrcoef(timeseries)
    #print(correlation_matrix.shape)
    return correlation_matrix

def subject_connectivity(timeseries, kind):
    """
        timeseries : timeseries table for subject (timepoints x regions)
        subject    : the subject ID
        atlas_name : name of the parcellation atlas used
        kind       : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save       : save the connectivity matrix to a file
        save_path  : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """
    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind, standardize='zscore_sample')
        connectivity = conn_measure.fit_transform([timeseries])[0]
    # print(connectivity.shape)
    return connectivity