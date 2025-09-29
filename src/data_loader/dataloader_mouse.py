import numpy as np
import scipy

from src.data_loader.data_utils import train_val_test_split, normalize, concatenate_behaviors

def dataloader_mouse_wheel(fpath_lst, **data_params):
    """
    INPUT:
        [data_params]: 
        - seed : int 
        - behavior_columns : a list of behavior keys
        - train_size 
        - val_size 
        - test_size
    OUTPUT: 
        [train_dataset] : a dictionary of entry (animal_id, session_id). Each value is another dictionary
                          {
                            'neural_data'   : a list of matrix,
                            'behavior_data' : a list of matrix,
                            'trials_type'   : a list of string, which corresponds to left or right,
                            'trials_length' : a list of int, which corresponds to the trial length
                          }
        [val_dataset] : similar as above
        [test_dataset]: similar as above
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
        subject_id  = fpath[-21:-17]
        session_date = fpath[-16:-8] 
        results = _load_onefile(fpath, **data_params)
        print(f'[INFO] subject {subject_id} session {session_date} data LOADED!')
        sess_name = f'Animal_{subject_id}-Session_{session_date}'

        train_dataset[sess_name] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        val_dataset[sess_name]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        test_dataset[sess_name]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        dataset_info[sess_name]  = {'train_trials': None, 'val_trials': None, 'test_trials': None, 'obs_dim': None}
        ### Get the train-val-test split based on trials ###
        num_trials = results['num_trials']
        train_trials, val_trials, test_trials = train_val_test_split(num_trials, data_params['val_size'], data_params['test_size'], seed)

        neural_data   = results['neural_data_byTrial']
        behv_data     = results['behavior_data_byTrial']
        trial_lengths = results['trial_lengths_data']
        trial_types   = results['trial_types_data']

        obs_dim = neural_data[0].shape[1]
        dataset_info[sess_name]['obs_dim'] = obs_dim
        dataset_info[sess_name]['train_trials'] = train_trials 
        dataset_info[sess_name]['val_trials']   = val_trials 
        dataset_info[sess_name]['test_trials']  = test_trials

        # Get the neural data before normalization
        train_neural_data = [neural_data[k] for k in train_trials]
        val_neural_data   = [neural_data[k] for k in val_trials]
        test_neural_data  = [neural_data[k] for k in test_trials]
        # Get the behavior data before normalization
        train_behv_data   = {key: [value[i] for i in train_trials] for key, value in behv_data.items()}
        val_behv_data   = {key: [value[i] for i in val_trials] for key, value in behv_data.items()}
        test_behv_data   = {key: [value[i] for i in test_trials] for key, value in behv_data.items()}
        # Get the trials length data
        train_trials_len   = [trial_lengths[k] for k in train_trials]
        val_trials_len   = [trial_lengths[k] for k in val_trials]
        test_trials_len   = [trial_lengths[k] for k in test_trials]
        # Get the trials type data
        train_trials_type = [trial_types[k] for k in train_trials]
        val_trials_type = [trial_types[k] for k in val_trials]
        test_trials_type = [trial_types[k] for k in test_trials]

        ### Normalize data ###
        # Normalize neural data
        train_neural_data, val_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, val_neural_data, test_neural_data)
        # Nornalize behavior data
        behv_data_train_norm_dic = {}
        behv_data_val_norm_dic   = {}
        behv_data_test_norm_dic = {}
        for j_behv, key in enumerate(data_params['behavior_keys']):
            num_train = len(train_behv_data[key])
            num_test  = len(test_behv_data[key])
            num_val   = len(val_behv_data[key]) if val_behv_data is not None else None
            train_key_data = train_behv_data[key]
            val_key_data   = val_behv_data[key] if val_behv_data is not None else None
            test_key_data  = test_behv_data[key]
            if key not in ['lick', 'speed_class']:
                train_key_data,  val_key_data,  test_key_data  = normalize(data_params['behavior_normalizor'], train_key_data, val_key_data, test_key_data)
            behv_data_train_norm_dic[key] = train_key_data
            if len(val_trials) != 0:
                behv_data_val_norm_dic[key] = val_key_data
            behv_data_test_norm_dic[key] = test_key_data
        # convert behavior dictionary to list of numpy arrays
        train_behv_data = concatenate_behaviors(behv_data_train_norm_dic, num_train, data_params['behavior_keys'])
        val_behv_data   = concatenate_behaviors(behv_data_val_norm_dic, num_val, data_params['behavior_keys'])
        test_behv_data  = concatenate_behaviors(behv_data_test_norm_dic, num_test, data_params['behavior_keys'])
        ######################
        train_dataset[sess_name]['neural_data'] = train_neural_data
        val_dataset[sess_name]['neural_data']   = val_neural_data
        test_dataset[sess_name]['neural_data']  = test_neural_data

        train_dataset[sess_name]['behavior_data'] = train_behv_data
        val_dataset[sess_name]['behavior_data']   = val_behv_data
        test_dataset[sess_name]['behavior_data']  = test_behv_data

        train_dataset[sess_name]['trials_type'] = train_trials_type 
        val_dataset[sess_name]['trials_type']   = val_trials_type
        test_dataset[sess_name]['trials_type']  = test_trials_type

        train_dataset[sess_name]['trials_length'] = train_trials_len
        val_dataset[sess_name]['trials_length']   = val_trials_len
        test_dataset[sess_name]['trials_length']  = test_trials_len
    return train_dataset, val_dataset, test_dataset, dataset_info

########### Helper functions ##########
def _load_onefile(file_path, **data_params):
    behavior_keys = data_params['behavior_keys']
    _data = scipy.io.loadmat(file_path)
    _neuraldata = _data['Ca_trace_byTrial']
    _behaviors  = {}
    _trialL     = _data['Behavior_byTrial']['trialL']
    for key in behavior_keys:
        _behaviors[key] = _data['Behavior_byTrial'][key]

    num_total_trials = _neuraldata.shape[0]
    print(f'number of total trial: {num_total_trials}')

    neural_data_byTrial = []
    behavior_data_byTrial = {key: [] for key in behavior_keys} # the concatenated behaviors for filtered trials
    # if 'speed_class' in behavior_keys:
    #     behavior_keys.add('speed_class')
    #     behavior_data_byTrial['speed_class'] = []
    trial_lengths_data    = []
    trial_types_data      = []

    for i in range(num_total_trials):
        x = _neuraldata[i, 0]
        ys = {key: value[0, 0][i, 0] for key, value in _behaviors.items()} # the behaviors for trial i

        if data_params['min_trial_len'] is not None:
            if len(x) < data_params['min_trial_len']:
                continue
        
        if data_params['max_trial_len'] is not None:
            if len(x) > data_params['max_trial_len']:
                continue
        
        neural_data_byTrial.append(x)
        trial_lengths_data.append(len(x))
        for key, value in ys.items():
            behavior_data_byTrial[key].append(ys[key])
        
        if _trialL[0, 0][i, 0].all() == 1:
            trial_types_data.append('right')
        else:
            trial_types_data.append('left')
    num_trials = len(trial_types_data)
    print(f'filtered trial number is {num_trials}')

    results = {
                'neural_data_byTrial'   : neural_data_byTrial,
                'behavior_data_byTrial' : behavior_data_byTrial,
                'trial_lengths_data'    : trial_lengths_data,
                'trial_types_data'      : trial_types_data,
                'num_trials'            : len(trial_types_data)
                }
    return results