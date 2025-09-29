import numpy as np
import pandas as pd
import pickle

import scipy.signal as signal
from sklearn.model_selection import train_test_split

import pynapple as nap

from pynwb import NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.core import MultiContainerInterface

from src.data_loader.data_utils import normalize

def dataloader_perich(fpath_lst, **data_params):
    """
    INPUT
        [data_params]: 
            - seed  : int
            - bin_size : int
            - align_field : string
            - align_range : (int, int)
            - margin : 0
            - ignored_trials: list ['A', 'I', 'F', 'R']
            - behavior_columns : a list 
            - gauss_width : int, in ms
            - train_size
            - val_size
            - test_size
            - neural_normalizor : None / minmax / zscore
            - behavior_normalizor : None / minmax / zscore
    OUTPUT
        [train_dataset] : a dictionary of entry (animal_id, session_id). Each value is another dictionary:
                          {neural_data : a list of matrix,
                           behavior_data: a list of matrix,
                           trials_type: a list of int, which correponds to the degree,
                           trials_length: a list of int, which corresponds to the trial length,
                          }
                          where each matrix is T_i x N and there are num_trial such matrix
        [val_dataset]   
        [test_dataset]
        [dataset_info]: a dictionary of train-val-test split trials 
                        {train_trials : a list of trial ids that belong to the [train_dataset],
                         val_trials   : a list of trial ids that belong to the [val_dataset], 
                         test_trials  : a list of trial ids that belong to the [test_dataset]
                        }
    """
    seed = data_params['seed']
    np.random.seed(seed)

    train_dataset = {}
    val_dataset   = {}
    test_dataset  = {}
    dataset_info  = {}

    for i, fpath in enumerate(fpath_lst):
        print(f'[DEBUG] monkey file path {fpath}')
        # sub-T_ses-CO-20130819_behavior+ecephys_processed_bin0.01_gocue_stop.pkl
        subject_id = fpath.split('sub-')[-1].split('_ses-')[0]
        session_date = fpath.split('_behavior')[0].split('-')[-1]
        print(subject_id, session_date)
        sess_name = f'Animal_{subject_id}-Session_{session_date}'
        
        # load data
        data_dict = pickle.load(open(fpath, 'rb'))
        neural_data_trial_list = data_dict['neural_data']
        n_trials = len(neural_data_trial_list)
        behavior_data_trial_list = list() # [data_dict[behv_name] for behv_name in data_params['behavior_columns']]
        for i_trial in range(n_trials):
            behv_trial = np.concatenate([data_dict[behv_name][i_trial] for behv_name in data_params['behavior_columns']], axis=-1)
            behavior_data_trial_list.append(behv_trial)
        
        trial_type_list = data_dict['trial_info']['target_dir']
        
        ### Get the train-val-test split based on trials ###
        trial_ids = np.arange(len(neural_data_trial_list))
        train_trials, test_trials = train_test_split(trial_ids, test_size=data_params['test_size'])
        if data_params['val_size'] != 0:
            train_trials, val_trials = train_test_split(train_trials, test_size=data_params['val_size'])
        else:
            val_trials = []
            
        print(f'[INFO] Trial data processed!')
        ############################################
        train_dataset[sess_name] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        val_dataset[sess_name]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        test_dataset[sess_name]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        dataset_info[sess_name]  = {'train_trials': [], 'val_trials': [], 'test_trials': [], 'obs_dim': None}
            
        ###################################################
        obs_dim = neural_data_trial_list[0].shape[1]
        print(f'[INFO] obs_dim: {obs_dim}')
        dataset_info[sess_name]['obs_dim'] = obs_dim 
        for trials, dataset, trial_name, in zip([train_trials, val_trials, test_trials], 
                                   [train_dataset, val_dataset, test_dataset],
                                   ['train_trials', 'val_trials', 'test_trials']):
            for t in trials:
                X = neural_data_trial_list[t]
                y = behavior_data_trial_list[t]
                uniq_dir = trial_type_list[t]
                trial_t_len = X.shape[0] # X shape: T*N
                dataset[sess_name]['neural_data'].append(X)
                dataset[sess_name]['behavior_data'].append(y)
                dataset[sess_name]['trials_type'].append(uniq_dir)
                dataset[sess_name]['trials_length'].append(trial_t_len)
                dataset_info[sess_name][trial_name].append(t)
            
        ### Normalize data ###
        # normalize neural data
        train_neural_data = train_dataset[sess_name]['neural_data']
        val_neural_data   = val_dataset[sess_name]['neural_data']
        test_neural_data  = test_dataset[sess_name]['neural_data']
        
        train_neural_data, val_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, val_neural_data, test_neural_data)

        train_dataset[sess_name]['neural_data'] = train_neural_data
        val_dataset[sess_name]['neural_data']   = val_neural_data
        test_dataset[sess_name]['neural_data']  = test_neural_data
        # normalize behavioral data
        train_behv_data = train_dataset[sess_name]['behavior_data']
        val_behv_data   = val_dataset[sess_name]['behavior_data']
        test_behv_data  = test_dataset[sess_name]['behavior_data']

        train_behv_data, val_behv_data, test_behv_data = normalize(data_params['behavior_normalizor'], train_behv_data, val_behv_data, test_behv_data)

        train_dataset[sess_name]['behavior_data'] = train_behv_data
        val_dataset[sess_name]['behavior_data']   = val_behv_data
        test_dataset[sess_name]['behavior_data']  = test_behv_data
        #####################
    
    return train_dataset, val_dataset, test_dataset, dataset_info