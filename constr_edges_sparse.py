from src.getData import get_data_adhd, construct_hyperedges_from_time_series, get_data_abide1,get_kangning_fmri, get_data_mdd
from tqdm import tqdm
from dhg import Hypergraph
import json
import numpy as np
import os

def construct_hyperedges_ADHD(lambda_value: float = 0.2, is_save: bool = False):
    np.random.seed(0)

    labels, features, timeseries_all = get_data_adhd()

    print("Constructing hypergraphs...")
    n_data = len(features)
    hyperedges_each_sample={}

    for i in tqdm(range(n_data)):
        ID=i
        time_series=timeseries_all[ID]
        hyperedges = construct_hyperedges_from_time_series(time_series, lambda_value)
        hyperedges_each_sample[ID]=hyperedges
        
    if is_save: 
        filename="ADHD_sparse_lambda_{}.json".format(lambda_value)
        filename = os.path.join("./src/hyperedges/", filename)
        with open(filename, 'w') as f:
            json.dump(hyperedges_each_sample, f, default=default_converter)
        print("Saved hyperedges to {}".format(filename))
    return hyperedges_each_sample


def construct_hyperedges_MDD(lambda_value: float = 0.2, is_save: bool = False):
    # 设置seed
    np.random.seed(0)

    labels, features, timeseries_all = get_data_mdd()

    print("Constructing hypergraphs...")
    n_data = len(features)
    hyperedges_each_sample={}

    for i in tqdm(range(n_data)):
        ID=i
        time_series=timeseries_all[ID]
        hyperedges = construct_hyperedges_from_time_series(time_series, lambda_value)
        hyperedges_each_sample[ID]=hyperedges
        
    if is_save:
        filename="MDD_sparse_lambda_{}.json".format(lambda_value)
        filename = os.path.join("./src/hyperedges/", filename)
        with open(filename, 'w') as f:
            json.dump(hyperedges_each_sample, f, default=default_converter)
        print("Saved hyperedges to {}".format(filename))
    return hyperedges_each_sample

        
def default_converter(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    
"""
The hyperedge file format:
{
    "0":[
        [0,1,2,5],
        [3,4,5],
        ...
    ],
    
    "1":[
        ...
    ],
    
    ...
}
"""

if __name__ == '__main__':
    construct_hyperedges_ADHD(lambda_value=0.5, is_save=True)   # choose lambda_value
    
