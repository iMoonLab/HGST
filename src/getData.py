import os
import pandas as pd
from .corr import subject_connectivity
import scipy
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def get_data_adhd():
    labels, all_data, all_timeseries = [],[],[]
    excel_dir = "path to ADHD_labels.csv"  
    excel = pd.read_csv(excel_dir)  
    IDs = excel['ID'].tolist() 
    IDs = [str(x) for x in IDs]  
    base_dir = r'path to ADHD data'
    for i, id in enumerate(IDs):  
        if len(id) == 5:  
            id = "00" + id  
        file_path = os.path.join(base_dir, id, f'sfnwmrda{id}_session_1_rest_1_aal_TCs.1D')  # 构建文件路径
        df = pd.read_csv(file_path, sep='\t', header=None)  
        timeseries = df.iloc[1:, 2:].values  
        timeseries = np.array(timeseries) 
        timeseries = timeseries.astype(float)  
        all_timeseries.append(timeseries)
        this_FC = subject_connectivity(timeseries,'correlation') 
        all_data.append(this_FC) 
        label = excel['DX'][i]
        if label > 1: 
            label = 1  
        labels.append(label)  
    all_data = np.array(all_data)  
    labels = np.array(labels) 
    
    return labels, all_data, all_timeseries  



def get_data_mdd():
    labels, data, all_timeseries = [],[],[]
  
    folder_path='path to /MDD/ROISignals_FunImgARCWF'
    excel_dir = "path to /REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.csv"
    excel = pd.read_csv(excel_dir)
    i = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)
            mat = scipy.io.loadmat(file_path)
            t = mat['ROISignals']
            t = t[:,:90]
            all_timeseries.append(t)
            data.append(subject_connectivity(t,'correlation'))

        pos_num = filename.split('_')[1].split('.')[0]
        if pos_num in excel['ID'].values:
            label = 1
        else:
            label = 0
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return labels, data, all_timeseries




def construct_hyperedges_from_time_series(time_series: np.ndarray, lambda_value: float, scaler=None):
    """
    Construct hyperedges vis Sparse Representation.
    
    Args:
        time_series (np.ndarray): Time series of each brain region, with shape (num_timepoints, num_regions),
                                  where num_regions represents the number of brain regions and num_timepoints represents the number of time points.
        lambda_value (float): L1 regularization parameter λ, used to control sparsity.
        scaler: Scaler object used to standardize the time series. Can choose "standard" or "minmax". Default is None, which means no scaling.
        
    Returns:
        list[list[int]]: List of hyperedges, each hyperedge is a list of node indices, conforming to the input format of `add_hyper

    """
    time_series = time_series.T
    
    num_regions, _ = time_series.shape
    hyperedges = []
    
    if scaler is not None:
        if scaler.lower() == "standard":
            bold_scaler = StandardScaler()
        elif scaler.lower() == "minmax":
            bold_scaler = MinMaxScaler()
        time_series = bold_scaler.fit_transform(time_series.T).T

    for m in range(num_regions):
        target_series = time_series[m]
        other_series = np.delete(time_series, m, axis=0) 
        lasso = Lasso(alpha=lambda_value,  max_iter=5000, tol=1e-4)
        lasso.fit(other_series.T, target_series)
        
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        original_indices = [i if i < m else i + 1 for i in non_zero_indices]
        hyperedge = [m] + original_indices
        hyperedges.append(hyperedge)

    return hyperedges


