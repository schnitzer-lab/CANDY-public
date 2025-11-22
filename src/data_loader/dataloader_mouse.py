import numpy as np
import scipy 
import pickle
import copy

from sklearn.model_selection import train_test_split
from src.data_loader.data_utils import train_val_test_split, normalize, concatenate_behaviors, parse_data_name

def create_dataset(areas):
    dataset = {
        'neural_data': [], # a list of trials, each is a numpy array of shape (trial_length x num_neurons), length = num_trials
        'behavior_data': [], # a list of trials, each is a numpy array of shape (trial_length x behavior_dim), length = num_trials
        'trials_type': [], # a list of trial types (length = num_trials)
        'trials_length': [], # a list of trial Lengths (length = num_trials)
    }
    return dataset

def concatenate_area_neural_data(neural_data_dict, areas, **data_params):
    """
    Concatenate neural data from multiple areas along the neuron dimension. 
    Filter areas based on provided area list. 
    Filter trials based on min_trial_len and max_trial_len in data_params if provided.
    
    Args:
        neural_data_dict: A dictionary where keys are area names and values are lists of trials.
                          Each trial is a numpy array of shape (trial_length x num_neurons_in_area).
        areas: List of area names to concatenate.
        **data_params: Optional parameters including:
            - min_trial_len: Minimum trial length to include (inclusive). Trials shorter than this are filtered out.
            - max_trial_len: Maximum trial length to include (inclusive). Trials longer than this are filtered out.
    
    Returns:
        concatenated_data: A list of trials, each is a numpy array of shape (trial_length x total_num_neurons).
        valid_trial_indices: A list of trial indices that passed the length filters.
        area_labels: A numpy array of shape (total_num_neurons,) where each element is 0 to (num_areas-1) 
                     indicating which area each neuron belongs to.
        area_mapping: A dictionary mapping area names to their corresponding numeric labels (0 to num_areas-1).
    """
    num_trials = len(next(iter(neural_data_dict.values())))
    min_trial_len = data_params.get('min_trial_len', None)
    max_trial_len = data_params.get('max_trial_len', None)
    
    concatenated_data = []
    valid_trial_indices = []
    
    for trial_idx in range(num_trials):
        trial_len = neural_data_dict[areas[0]][trial_idx].shape[0]
        
        # Apply trial length filters if specified
        if min_trial_len is not None and trial_len < min_trial_len:
            continue  # Skip trials that are too short
        if max_trial_len is not None and trial_len > max_trial_len:
            continue  # Skip trials that are too long
        
        # Concatenate data from all areas for this trial
        concatenated_trial = np.concatenate(
            [neural_data_dict[area][trial_idx].reshape(trial_len, -1) for area in areas], axis=1
        )
        concatenated_data.append(concatenated_trial)
        valid_trial_indices.append(trial_idx)
    
    # Create area mapping: {area_name: numeric_label}
    area_mapping = {area: idx for idx, area in enumerate(areas)}
    
    # Create area labels based on the first valid trial
    if len(concatenated_data) > 0:
        total_neurons = concatenated_data[0].shape[1]
        area_labels = np.zeros(total_neurons, dtype=int)
        
        start_idx = 0
        for area in areas:
            # Use first trial to determine neuron count per area
            area_neuron_count = neural_data_dict[area][valid_trial_indices[0]].shape[1]
            
            # Set numeric labels
            area_labels[start_idx:start_idx + area_neuron_count] = area_mapping[area]
            
            start_idx += area_neuron_count
    else:
        # No valid trials found
        raise Exception("No valid trials found after applying length filters.")

    return concatenated_data, valid_trial_indices, area_labels, area_mapping, num_trials

def concatenate_behaviors(behv_data_dict, behavior_keys):
    """
    Concatenate behavior data from multiple keys along the behavior dimension.
    Args:
        behv_data_dict: A dictionary where keys are behavior names and values are lists of trials.
                        Each trial is a numpy array of shape (trial_length x behavior_dim_for_key).
        behavior_keys: List of behavior keys to concatenate.
    Returns:
        concatenated_data: A list of trials, each is a numpy array of shape (trial_length x total_behavior_dim).
    """
    num_trials = len(next(iter(behv_data_dict.values())))
    concatenated_data = []
    for trial_idx in range(num_trials):
        trial_len = behv_data_dict[behavior_keys[0]][trial_idx].shape[0]
        concatenated_trial = np.concatenate(
            [behv_data_dict[key][trial_idx].reshape(trial_len, -1) for key in behavior_keys], axis=1
        )
        concatenated_data.append(concatenated_trial)
    return concatenated_data

# parse the dataset into neural_data_dict and behavior_data_dict
def parse_neural_data_to_dict(dataset, dataset_info):
    neural_data_dict = {sess_name: {} for sess_name in dataset_info.keys()}
    for sess_name in dataset_info.keys():
        neural_data = copy.deepcopy(dataset[sess_name]['neural_data'])
        area_mapping = dataset_info[sess_name]['area_mapping']
        area_labels = dataset_info[sess_name]['area_labels']
        for trial_idx in range(len(neural_data)):
            trial_data = neural_data[trial_idx]
            # Map area labels to their corresponding trial data
            for area_name, area_idx in area_mapping.items():
                if area_name not in neural_data_dict[sess_name]:
                    neural_data_dict[sess_name][area_name] = []
                neural_data_dict[sess_name][area_name].append(trial_data[:, area_labels == area_idx])
    return neural_data_dict

def to_list_of_areas(neural_data, area_labels, area_mapping):
    neural_data_by_area = {}
    for area in area_mapping.keys():
        label = area_mapping[area]
        area_indices = np.where(area_labels == label)[0]
        neural_data_by_area[area] = [neural_data[trial_id][:, area_indices] for trial_id in range(len(neural_data))]
    return neural_data_by_area

def dataloader_mouse_wheel(fpath_lst, **data_params):
    """
    """
    seed = data_params['seed']
    np.random.seed(seed)
    train_dataset = {}
    valid_dataset = {}
    test_dataset  = {}
    dataset_info  = {}

    for i, fpath in enumerate(fpath_lst):
        subject_id, session_date = parse_data_name(fpath)
        areas = data_params['brain_areas']
        sess_name = f'Animal_{subject_id}-Session_{session_date}'
        
        train_dataset[sess_name] = create_dataset(areas)
        valid_dataset[sess_name]   = create_dataset(areas)
        test_dataset[sess_name]  = create_dataset(areas)

        pickle_data = pickle.load(open(fpath, 'rb'))
        # load all data (neural data include all areas, behv data include all behavior columns)
        neural_data  = pickle_data['neural_data']
        behv_data    = pickle_data['behavior_data']

        # Concatenate neural data and get valid trial indices after filtering
        neural_data, valid_trial_indices, area_labels, area_mapping, orig_num_trials = concatenate_area_neural_data(neural_data, areas=areas, **data_params)
        
        # Concatenate behavior data for all trials first
        behv_data_all = concatenate_behaviors(behv_data, behavior_keys=data_params['behavior_keys'])
        behv_data = [behv_data_all[idx] for idx in valid_trial_indices] # Filter behavior data to only include valid trials (matching neural data)

        # filter trial_types and trial_outcomes based on valid_trial_indices if they exist
        if 'trial_types' in pickle_data.keys():
            trial_types_all = pickle_data['trial_types']
            trial_types = [trial_types_all[idx] for idx in valid_trial_indices]
        else:
            trial_types = [None for _ in range(len(valid_trial_indices))]
            print(f'[WARNING] trial_types not found in data for session {sess_name}, setting all to None.')

        num_trials = len(neural_data)
        obs_dim = neural_data[0].shape[1] if num_trials > 0 else 0

        print(f'[DATALOADER] Loaded session {sess_name} with {num_trials} trials (filtered from original {orig_num_trials} trials), obs_dim: {obs_dim}, num_areas: {len(areas)}')

        train_trials, valid_trials, test_trials = train_val_test_split(num_trials, data_params['valid_size'], data_params['test_size'], seed)
        
        dataset_info[sess_name] = {
            'obs_dim': obs_dim,
            'train_trials': train_trials,
            'valid_trials': valid_trials,
            'test_trials': test_trials,
            'area_labels': area_labels,
            'area_mapping': area_mapping,
            'valid_trial_indices': valid_trial_indices
        }

        # Get the neural data before normalization
        train_neural_data = [neural_data[k] for k in train_trials]
        valid_neural_data = [neural_data[k] for k in valid_trials]
        test_neural_data = [neural_data[k] for k in test_trials]
        # Get the behavior data before normalization
        train_behv_data   = [behv_data[k] for k in train_trials]
        valid_behv_data   = [behv_data[k] for k in valid_trials]
        test_behv_data    = [behv_data[k] for k in test_trials]
        # Get the trials length data
        train_trials_len   = [train_neural_data[k].shape[0] for k in range(len(train_trials))]
        valid_trials_len   = [valid_neural_data[k].shape[0] for k in range(len(valid_trials))]
        test_trials_len    = [test_neural_data[k].shape[0] for k in range(len(test_trials))]
        # Get trials type data (don't have different trial types here, so all None)
        train_dataset[sess_name]['trials_type'] = [trial_types[k] for k in train_trials]
        valid_dataset[sess_name]['trials_type'] = [trial_types[k] for k in valid_trials]
        test_dataset[sess_name]['trials_type']  = [trial_types[k] for k in test_trials]

        #------ Normalization ------#
        train_neural_data, valid_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, valid_neural_data, test_neural_data)
        train_behv_data, valid_behv_data, test_behv_data = normalize(data_params['behavior_normalizor'], train_behv_data, valid_behv_data, test_behv_data)
        #---------------------------#
        train_dataset[sess_name]['neural_data'] = train_neural_data
        valid_dataset[sess_name]['neural_data'] = valid_neural_data
        test_dataset[sess_name]['neural_data']  = test_neural_data

        train_dataset[sess_name]['behavior_data'] = train_behv_data
        valid_dataset[sess_name]['behavior_data'] = valid_behv_data
        test_dataset[sess_name]['behavior_data']  = test_behv_data

        train_dataset[sess_name]['trials_length'] = train_trials_len
        valid_dataset[sess_name]['trials_length'] = valid_trials_len
        test_dataset[sess_name]['trials_length']  = test_trials_len
    return train_dataset, valid_dataset, test_dataset, dataset_info