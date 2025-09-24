import random 

import numpy as np
import scipy 
from copy import deepcopy

import torch

from src.data_loader.data_utils import _wheelspeed2cmpsec, _get_speedclass, train_val_test_split, normalize, concatenate_behaviors_all
from src.data_loader.datasets import NARNNDataset

class NeuralData4Areas:
    def __init__(self, file_path_lst, **data_params):
        """
        This function load data from [file_path_lst].

        INPUT:
            [file_path_lst] : a list of .mat files, each refer to an agent. 
            [data_params]   : includes 
                - min_trial_len : default 0
                - max_trial_len : default inf
        """
        self.min_trial_len = data_params['min_trial_len']
        self.max_trial_len = data_params['max_trial_len']

        self.num_agents = len(file_path_lst)

        self.neural_data_all = []
        self.speed_data_all  = []
        self.sclass_data_all = []
        self.cursor_data_all = []
        self.lick_data_all   = []
        self.trial_lengths_all = []
        self.trial_types_all   = []
        self.num_trials_all    = []
        for file_path in file_path_lst:
            result = self._load_onefile(file_path=file_path)
            self.neural_data_all.append(result['neural_data_byTrial'])
            self.speed_data_all.append(result['speed_data_byTrial'])
            self.sclass_data_all.append(result['sclass_data_byTrial'])
            self.cursor_data_all.append(result['cursor_data_byTrial'])
            self.lick_data_all.append(result['lick_data_byTrial'])
            self.trial_lengths_all.append(result['trial_lengths'])
            self.trial_types_all.append(result['trial_types'])
            self.num_trials_all.append(result['num_trials'])
    
    def get_behavior_keys(self):
        return ['speed', 'cursor_position', 'speed_class', 'lick']

    def get_train_val_test_split_byagent_indices(self, data_params):
        seed = data_params['seed']
        val_size = data_params['val_size']
        test_size = data_params['test_size']
        train_indices_lst = []
        val_indices_lst   = []
        test_indices_lst  = []
        for i in range(self.num_agents):            
            num_trials = self.num_trials_all[i]
            train_indices, val_indices, test_indices = train_val_test_split(num_trials, val_size, test_size, seed)
            train_indices_lst.append(train_indices)
            val_indices_lst.append(val_indices)
            test_indices_lst.append(test_indices)
            if val_size != 0:
                print(f'[INFO] Mouse {i} has {num_trials} trials ==> split into Training {len(train_indices)} & Validation {len(val_indices)} & Testing {len(test_indices)}')
            else:
                print(f'[INFO] Mouse {i} has {num_trials} trials ==> split into Training {len(train_indices)} & Testing {len(test_indices)}')

        
        data_params['train_indices_lst'] = train_indices_lst
        data_params['val_indices_lst']   = val_indices_lst
        data_params['test_indices_lst']  = test_indices_lst
        return data_params
    
    def load_data_byagent(self, data_params):
        seed = data_params['seed']
        train_indices_lst = data_params['train_indices_lst']
        val_indices_lst   = data_params['val_indices_lst']
        test_indices_lst  = data_params['test_indices_lst']
        neural_normalizor   = data_params['neural_normalizor']
        behavior_normalizor = data_params['behavior_normalizor']
        input_normalizor    = data_params['input_normalizor']
        behavior_keys = data_params['behavior_keys']
        if seed is not None:
            np.random.seed(seed=seed)
        
        train_neural_data_all = []
        train_speed_data_all  = []
        train_sclass_data_all = []
        train_cursor_data_all = []
        train_lick_data_all   = []
        train_trial_len_all   = []
        train_trial_types_all = []

        test_neural_data_all = []
        test_speed_data_all  = []
        test_sclass_data_all = []
        test_cursor_data_all = []
        test_lick_data_all   = []
        test_trial_len_all   = []
        test_trial_types_all = []

        val_neural_data_all = []
        val_speed_data_all  = []
        val_sclass_data_all = []
        val_cursor_data_all = []
        val_lick_data_all   = []
        val_trial_len_all   = []
        val_trial_types_all = []

        obs_dim_lst = []
        for i in range(self.num_agents):
            train_indices = train_indices_lst[i]
            val_indices   = val_indices_lst[i]
            test_indices  = test_indices_lst[i]
            
            # get the data for each agent
            neural_data = self.neural_data_all[i]
            speed_data  = self.speed_data_all[i]
            sclass_data = self.sclass_data_all[i]
            cursor_data = self.cursor_data_all[i]
            lick_data   = self.cursor_data_all[i]
            trial_lengths = self.trial_lengths_all[i]
            trial_types   = self.trial_types_all[i]

            obs_dim_lst.append(neural_data[0].shape[1])

            train_neural_data = [neural_data[k] for k in train_indices]
            train_speed_data  = [speed_data[k] for k in train_indices]
            train_sclass_data = [sclass_data[k] for k in train_indices]
            train_cursor_data = [cursor_data[k] for k in train_indices]
            train_lick_data   = [lick_data[k] for k in train_indices]
            train_trial_len   = [trial_lengths[k] for k in train_indices]
            train_trial_types = [trial_types[k] for k in train_indices]
            

            test_neural_data = [neural_data[k] for k in test_indices]
            test_speed_data  = [speed_data[k] for k in test_indices]
            test_sclass_data = [sclass_data[k] for k in test_indices]
            test_cursor_data = [cursor_data[k] for k in test_indices]
            test_lick_data   = [lick_data[k] for k in test_indices]
            test_trial_len   = [trial_lengths[k] for k in test_indices]
            test_trial_types = [trial_types[k] for k in test_indices]
            
            if val_indices is not None:
                val_neural_data = [neural_data[k] for k in val_indices]
                val_speed_data  = [speed_data[k] for k in val_indices]
                val_sclass_data = [sclass_data[k] for k in val_indices]
                val_cursor_data = [cursor_data[k] for k in val_indices]
                val_lick_data   = [lick_data[k] for k in val_indices]
                val_trial_len   = [trial_lengths[k] for k in val_indices]
                val_trial_types = [trial_types[k] for k in val_indices]
            else:
                val_neural_data = None 
                val_speed_data  = None 
                val_cursor_data = None 
                
            train_neural_data, val_neural_data, test_neural_data = normalize(neural_normalizor, train_neural_data, val_neural_data, test_neural_data)
            train_speed_data,  val_speed_data,  test_speed_data  = normalize(behavior_normalizor, train_speed_data, val_speed_data, test_speed_data)
            train_cursor_data, val_cursor_data, test_cursor_data = normalize(input_normalizor, train_cursor_data, val_cursor_data, test_cursor_data)
            
            train_neural_data_all.append(train_neural_data)
            train_speed_data_all.append(train_speed_data)
            train_sclass_data_all.append(train_sclass_data)
            train_cursor_data_all.append(train_cursor_data)
            train_lick_data_all.append(train_lick_data)
            train_trial_len_all.append(train_trial_len)
            train_trial_types_all.append(train_trial_types)

            test_neural_data_all.append(test_neural_data)
            test_speed_data_all.append(test_speed_data)
            test_sclass_data_all.append(test_sclass_data)
            test_cursor_data_all.append(test_cursor_data)
            test_lick_data_all.append(test_lick_data)
            test_trial_len_all.append(test_trial_len)
            test_trial_types_all.append(test_trial_types)

            if val_indices is not None:
                val_neural_data_all.append(val_neural_data)
                val_speed_data_all.append(val_speed_data)
                val_sclass_data_all.append(val_sclass_data)
                val_cursor_data_all.append(val_cursor_data)
                val_lick_data_all.append(val_lick_data)
                val_trial_len_all.append(val_trial_len)
                val_trial_types_all.append(val_trial_types)
            else:
                val_neural_data_all = None 
                val_speed_data_all  = None
                val_sclass_data_all = None
                val_cursor_data_all = None
                val_lick_data_all   = None
                val_trial_len_all   = None
                val_trial_types_all = None

        # behavior_data_all is the concatenation of all behaviors specified in the behavior_keys
        train_behavior_data_all_noncat = {
                                    'speed': train_speed_data_all, 
                                    'cursor_position': train_cursor_data_all, 
                                    'speed_class': train_sclass_data_all, 
                                    'lick': train_lick_data_all
                                } 
        train_behavior_data_all = concatenate_behaviors_all(train_behavior_data_all_noncat, behavior_keys)

        train_dataset = {
                        'neural_data': train_neural_data_all, # number of agent x number of trial x trial length x number of neurons
                        'behavior_data': train_behavior_data_all,
                        'input_data'   : train_cursor_data_all,
                        'trials_length': train_trial_len_all,
                        'trials_type'  : train_trial_types_all, 
                        'obs_dim': obs_dim_lst
                        }
        
        test_behavior_data_all_noncat = {
                                'speed': test_speed_data_all, 
                                'cursor_position': test_cursor_data_all, 
                                'speed_class': test_sclass_data_all, 
                                'lick': test_lick_data_all
                                } 
        test_behavior_data_all = concatenate_behaviors_all(test_behavior_data_all_noncat, behavior_keys)
        
        test_dataset = {
                        'neural_data' : test_neural_data_all,
                        'behavior_data': test_behavior_data_all,
                        'input_data'   : test_cursor_data_all,
                        'trials_length': test_trial_len_all,
                        'trials_type'  : test_trial_types_all, 
                        'obs_dim': obs_dim_lst
                        }
        if val_indices is not None:
            val_behavior_data_all_noncat = {
                                    'speed': val_speed_data_all, 
                                    'cursor_position': val_cursor_data_all, 
                                    'speed_class': val_sclass_data_all, 
                                    'lick': val_lick_data_all
                                } 
            val_behavior_data_all = concatenate_behaviors_all(val_behavior_data_all_noncat, behavior_keys)
            
            val_dataset = {
                            'neural_data' : val_neural_data_all,
                            'behavior_data': val_behavior_data_all,
                            'input_data'   : val_cursor_data_all,
                            'trials_length': val_trial_len_all,
                            'trials_type'  : val_trial_types_all, 
                            'obs_dim': obs_dim_lst
                            }
        else:
            val_dataset = None 
        
        return train_dataset, val_dataset, test_dataset
    
    def load_data_singleagent(self, data_params):
        train_dataset, val_dataset, test_dataset = self.load_data_byagent(data_params)
        train_dataset = {
                        'neural_data': train_dataset['neural_data'][0],
                        'behavior_data': train_dataset['behavior_data'][0],
                        'trials_length': train_dataset['trials_length'][0], 
                        'trials_type'  : train_dataset['trials_type'][0],
                        'obs_dim'      : train_dataset['obs_dim'][0]                    
                        }
        
        test_dataset = {
                        'neural_data': test_dataset['neural_data'][0],
                        'behavior_data': test_dataset['behavior_data'][0],
                        'trials_length': test_dataset['trials_length'][0], 
                        'trials_type'  : test_dataset['trials_type'][0],
                        'obs_dim'      : test_dataset['obs_dim'][0]                    
                        }
        
        if val_dataset is not None:
            val_dataset = {
                            'neural_data': val_dataset['neural_data'][0],
                            'behavior_data': val_dataset['behavior_data'][0],
                            'trials_length': val_dataset['trials_length'][0], 
                            'trials_type'  : val_dataset['trials_type'][0],
                            'obs_dim'      : val_dataset['obs_dim'][0]                    
                            }
        return train_dataset, val_dataset, test_dataset

    def _load_onefile(self, file_path):
        _data = scipy.io.loadmat(file_path)
        _neuraldata = _data['Ca_trace_byTrial']
        _speeddata  = _data['speed_byTrial']
        _cursordata = _data['cursor_byTrial']
        _lickdata   = _data['lick_byTrial']

        neural_data_byTrial = []
        speed_data_byTrial  = []
        sclass_data_byTrial = []
        cursor_data_byTrial = []
        lick_data_byTrial   = []
        trial_lengths       = []
        trial_types         = []

        for i in range(_speeddata.size):
            x = _neuraldata[i,0]
            y = _speeddata[i,0]
            u = _cursordata[i,0]
            lick = _lickdata[i,0]

            y_cmpsec = _wheelspeed2cmpsec(y)
            y_class  = _get_speedclass(y)

            if self.min_trial_len is not None:
                if len(x) < self.min_trial_len:
                    continue
            
            if self.max_trial_len is not None:
                if len(x) > self.max_trial_len:
                    continue
            neural_data_byTrial.append(x)
            speed_data_byTrial.append(y_cmpsec)
            sclass_data_byTrial.append(y_class)
            cursor_data_byTrial.append(u)
            lick_data_byTrial.append(lick)
            trial_lengths.append(len(x))
            if u[0]>0:
                trial_types.append('left')
            else:
                trial_types.append('right')
        result = {
                    'neural_data_byTrial': neural_data_byTrial,
                    'speed_data_byTrial' : speed_data_byTrial,
                    'sclass_data_byTrial': sclass_data_byTrial,
                    'cursor_data_byTrial': cursor_data_byTrial,
                    'lick_data_byTrial'  : lick_data_byTrial,
                    'trial_lengths'      : trial_lengths,
                    'trial_types'        : trial_types,
                    'num_trials'         : len(trial_lengths)
                }
        return result

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

def load_rnndataset_lfads(dataset, frac=1.0, fixed_length=150):
    """
    INPUT
        [dataset] : a dictionary of single dataset
            - neural_data  : (num_trial, trial_len, num_neurons)
            - behavior_data: (num_trial, trial_len, K)
            - trials_length: (,num_trial)
            - trials_type:   (,num_trial)
        [fixed_length]: Target padded length (default: 150)
    """
    neural_data = dataset['neural_data']
    behavior_data = dataset['behavior_data']
    trials_length = dataset['trials_length']
    trials_type = dataset['trials_type']

    # Pad neural_data to fixed_length
    neural_data_padded = torch.stack([
        torch.cat([
            torch.tensor(trial),
            torch.zeros((fixed_length - trial.shape[0], trial.shape[1]))
        ], dim=0)
        for trial in neural_data
    ])

    # Pad behavior_data to fixed_length (same logic)
    behavior_data_padded = torch.stack([
        torch.cat([
            torch.tensor(trial),
            torch.zeros((fixed_length - trial.shape[0], trial.shape[1]))
        ], dim=0)
        for trial in behavior_data
    ])

    # Create dataset
    rnndataset = NARNNDataset(neural_data_padded, 
                              behavior_data_padded, 
                              trials_length, 
                              trials_type,
                              frac=frac)
    return rnndataset