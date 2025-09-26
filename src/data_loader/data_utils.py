import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader.normalization import standardize_data, minmaxscale_data

def train_val_test_split(num_trials, val_size, test_size, seed):
    """
    Return train, test, validation indices split on [seed]
    """
    np.random.seed(seed)
    # get the train-test split based on trials
    trials = np.arange(num_trials)
    train_trials, test_trials = train_test_split(trials, test_size = test_size)
    if val_size != 0:
        train_trials, val_trials  = train_test_split(train_trials, test_size = val_size)
    else:
        val_trials = []

    return train_trials, val_trials, test_trials 

def normalize(type, train_data, val_data, test_data):
    if type is not None:
        if type == 'zscore':
            train_data, _mean, _std = standardize_data(train_data)
            if len(val_data) != 0:
                val_data, _, _ = standardize_data(val_data, _mean=_mean, _std=_std)        
            test_data, _, _ = standardize_data(test_data, _mean=_mean, _std=_std)
        elif type == 'minmax':
            train_data, _max, _min = minmaxscale_data(train_data)
            if len(val_data) != 0:
                val_data, _, _  = minmaxscale_data(val_data, _max=_max, _min=_min)
            test_data, _, _  = minmaxscale_data(test_data, _max=_max, _min=_min)
    
    return train_data, val_data, test_data

def concatenate_behaviors_all(behavior_data_all_noncat, behavior_keys):
    # Check if all behavior_keys are in train_behavior_data_all_noncat
    assert all(key in behavior_data_all_noncat for key in behavior_keys), \
        "Not all elements in behavior_keys are in train_behavior_data_all_noncat."

    num_agents = len(next(iter(behavior_data_all_noncat.values())))
    
    # Initialize the output list
    behavior_data_all = [[] for _ in range(num_agents)]

    # Concatenate the specified behaviors for each agent and each trial
    for agent_idx in range(num_agents):
        num_trials = len(next(iter(behavior_data_all_noncat.values()))[agent_idx])
        for trial_idx in range(num_trials):
            trial_len = behavior_data_all_noncat['speed'][agent_idx][trial_idx].shape[0]
            concatenated_data = np.concatenate(
                [behavior_data_all_noncat[key][agent_idx][trial_idx].reshape(trial_len, -1) for key in behavior_keys], axis=1
            )
            # print(f'[DEBUG] agent {agent_idx} : trial {trial_idx} has trial length {trial_len}, concat data {concatenated_data.shape}, and behavior data {len(behavior_data_all[0])}, {len(behavior_data_all[1])}')
            behavior_data_all[agent_idx].append(concatenated_data)
    
    return behavior_data_all


def concatenate_behaviors(behavior_data_noncat, num_trials, behavior_keys):
    assert all(key in behavior_data_noncat for key in behavior_keys), \
        f"Not all elements in behavior_keys are in train_behavior_data_noncat: "+\
        f"behavior data key {behavior_data_noncat.keys()} and behavior keys are {behavior_keys}."
    if not isinstance(behavior_keys, list):
        behavior_keys  = list(behavior_keys)
    behavior_data_cat = []
    for trial_idx in range(num_trials):
        trial_len = behavior_data_noncat[behavior_keys[0]][trial_idx].shape[0]
        concatenated_data = np.concatenate(
            [behavior_data_noncat[key][trial_idx].reshape(trial_len, -1) for key in behavior_keys], axis=1
        )
        behavior_data_cat.append(concatenated_data)
    
    return behavior_data_cat