import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


def standardize_data(dataset, _mean=None, _std=None):
    if _mean is None and _std is None:
        all_pcs = np.vstack([trialdata for trialdata in dataset])
        _mean = all_pcs.mean(axis=0)
        _std  = all_pcs.std(axis=0)
    
    dataset_normalized = []
    for trialdata in dataset:
        trialdata_norm = (trialdata - _mean) / (_std + 1e-8)
        dataset_normalized.append(trialdata_norm)
    
    return dataset_normalized, _mean, _std

def minmaxscale_data(dataset, _max=None, _min=None):
    if _max is None and _min is None:
        all_pcs = np.vstack([trialdata for trialdata in dataset])
        _max = np.max(all_pcs, axis=0)
        _min = np.min(all_pcs, axis=0)
    
    dataset_normalized = []
    for trialdata in dataset:
        trialdata_norm = (trialdata - _min) / (_max - _min + 1e-8)
        dataset_normalized.append(trialdata_norm)
    return dataset_normalized, _max, _min
