"""
This module process the MATLAB data into python class that will be used in future analysis procedure. 
"""
import numpy as np

import torch
from torch.utils.data import Dataset   

def load_rnndataset(dataset, frac=1.0):
    """
    INPUT
        [dataset] : a dictionary of single dataset
            - neural_data  : (num_trial, trial_len, num_neurons)
            - behavior_data: (num_trial, trial_len, K)
            - trials_length: (,num_trial)
            - trials_type:   (,num_trial)
    """   
    neural_data = dataset['neural_data']
    behavior_data  = dataset['behavior_data']
    trials_length = dataset['trials_length']
    trials_type   = dataset['trials_type']
    neural_data_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(trial) for trial in neural_data], batch_first=True)
    behavior_data_padded  = torch.nn.utils.rnn.pad_sequence([torch.tensor(trial) for trial in behavior_data], batch_first=True)

    rnndataset = NARNNDataset(neural_data_padded, 
                              behavior_data_padded, 
                              trials_length, 
                              trials_type,
                              frac=frac)
    return rnndataset

class NARNNDataset(Dataset):
    def __init__(self, 
                 dataset_neural_padded, 
                 dataset_behavior_padded, 
                 dataset_tlen, 
                 dataset_ttype,
                 frac=1.0):
        self.traces_trial   = dataset_neural_padded.type(torch.FloatTensor)
        self.behavior_trial = dataset_behavior_padded.type(torch.FloatTensor)
        self.idx_list       = np.arange(len(dataset_tlen))
        if frac < 1.0:
            print(f'[INFO] shuffle the dataset because fraction set {frac} is less than 1.')
            np.random.shuffle(self.idx_list)
        self.idx_list       = self.idx_list[:int(frac*len(dataset_tlen))]
        # print(f'number of trials in the dataset: {len(self.idx_list)}/{len(dataset_tlen)}')

        self.dataset_tlen     = torch.tensor(dataset_tlen, dtype=torch.int)
        self.dataset_ttype    = dataset_ttype

    def get_byTrial(self, data_padded):
        trial_lengths = self.dataset_tlen
        assert data_padded.shape[0] == len(trial_lengths), 'incorrect padded input data shape'
        data_byTrial = []
        for idx in self.idx_list:
            data_byTrial.append(data_padded[idx, :trial_lengths[idx], :])
        return data_byTrial

    def __len__(self):
        return len(self.idx_list)    

    def __getitem__(self, idx):
        idx = self.idx_list[idx]
        return self.traces_trial[idx], self.behavior_trial[idx], self.dataset_tlen[idx], self.dataset_ttype[idx]