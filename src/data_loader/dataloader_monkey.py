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
        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'train_trials': [], 'val_trials': [], 'test_trials': [], 'obs_dim': None}
            
        ###################################################
        obs_dim = neural_data_trial_list[0].shape[1]
        print(f'[INFO] obs_dim: {obs_dim}')
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['obs_dim'] = obs_dim 
        for trials, dataset, trial_name, in zip([train_trials, val_trials, test_trials], 
                                   [train_dataset, val_dataset, test_dataset],
                                   ['train_trials', 'val_trials', 'test_trials']):
            for t in trials:
                X = neural_data_trial_list[t]
                y = behavior_data_trial_list[t]
                uniq_dir = trial_type_list[t]
                trial_t_len = X.shape[0] # X shape: T*N
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'].append(X)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'].append(y)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_type'].append(uniq_dir)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_length'].append(trial_t_len)
                dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')][trial_name].append(t)
            
        ### Normalize data ###
        # normalize neural data
        train_neural_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        val_neural_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        test_neural_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        
        train_neural_data, val_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, val_neural_data, test_neural_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'] = train_neural_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']   = val_neural_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']  = test_neural_data
        # normalize behavioral data
        train_behv_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        val_behv_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        test_behv_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']

        train_behv_data, val_behv_data, test_behv_data = normalize(data_params['behavior_normalizor'], train_behv_data, val_behv_data, test_behv_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'] = train_behv_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']   = val_behv_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']  = test_behv_data
        #####################
    
    return train_dataset, val_dataset, test_dataset, dataset_info

def dataloader_perich_particular_direction(fpath_lst, **data_params):
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

    held_out_direction_id = data_params['held_out_direction_id']
    held_out_fraction = data_params['held_out_fraction'] # randing from 0 to 1

    for i, fpath in enumerate(fpath_lst):
        print(f'[DEBUG] monkey file path {fpath}')
        # sub-T_ses-CO-20130819_behavior+ecephys_processed_bin0.01_gocue_stop.pkl
        subject_id = fpath.split('sub-')[-1].split('_ses-')[0]
        session_date = fpath.split('_behavior')[0].split('-')[-1]
        print(subject_id, session_date)
        
        # load data
        data_dict = pickle.load(open(fpath, 'rb'))
        neural_data_trial_list = data_dict['neural_data']
        target_direction = data_dict['trial_info']['target_dir']
        print(f'[DEBUG] target_direction: {np.unique(target_direction)}')
        unique_directions = np.array([-2.35619449, -1.57079633, -0.78539816,  0.        ,  0.78539816, 1.57079633,  2.35619449,  3.14159265])
        held_out_direction = unique_directions[held_out_direction_id]
        print(f'[DEBUG] held_out_direction: {held_out_direction}')

        n_trials = len(neural_data_trial_list)
        behavior_data_trial_list = list() # [data_dict[behv_name] for behv_name in data_params['behavior_columns']]
        for i_trial in range(n_trials):
            behv_trial = np.concatenate([data_dict[behv_name][i_trial] for behv_name in data_params['behavior_columns']], axis=-1)
            behavior_data_trial_list.append(behv_trial)
        
        trial_type_list = data_dict['trial_info']['target_dir']
        
        ### Get the train-val-test split based on trials ###
        trial_ids = np.arange(len(neural_data_trial_list))
        test_trials = np.argwhere(np.abs(target_direction - held_out_direction) < 0.1).flatten()
        train_trials = np.setdiff1d(trial_ids, test_trials)
        train_trials_target_direction = np.random.choice(test_trials, size=int(len(test_trials) * (1 - held_out_fraction)), replace=False)
        train_trials = np.concatenate([train_trials, train_trials_target_direction])
        test_trials = np.setdiff1d(test_trials, train_trials_target_direction)
        train_trials, val_trials = train_test_split(train_trials, test_size=data_params['val_size'])
        print(f'[DEBUG] number of train trials: {len(train_trials)}, val trials: {len(val_trials)}, test trials: {len(test_trials)}')
        # train_trials, test_trials = train_test_split(trial_ids, test_size=data_params['test_size'])
        # if data_params['val_size'] != 0:
        #     train_trials, val_trials = train_test_split(train_trials, test_size=data_params['val_size'])
        # else:
        #     val_trials = []
            
        print(f'[INFO] Trial data processed!')
        ############################################
        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'train_trials': [], 'val_trials': [], 'test_trials': [], 'obs_dim': None}
            
        ###################################################
        obs_dim = neural_data_trial_list[0].shape[1]
        print(f'[INFO] obs_dim: {obs_dim}')
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['obs_dim'] = obs_dim 
        for trials, dataset, trial_name, in zip([train_trials, val_trials, test_trials], 
                                   [train_dataset, val_dataset, test_dataset],
                                   ['train_trials', 'val_trials', 'test_trials']):
            for t in trials:
                X = neural_data_trial_list[t]
                y = behavior_data_trial_list[t]
                uniq_dir = trial_type_list[t]
                trial_t_len = X.shape[0] # X shape: T*N
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'].append(X)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'].append(y)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_type'].append(uniq_dir)
                dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_length'].append(trial_t_len)
                dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')][trial_name].append(t)
            
        ### Normalize data ###
        # normalize neural data
        train_neural_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        val_neural_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        test_neural_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        
        train_neural_data, val_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, val_neural_data, test_neural_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'] = train_neural_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']   = val_neural_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']  = test_neural_data
        # normalize behavioral data
        train_behv_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        val_behv_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        test_behv_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']

        train_behv_data, val_behv_data, test_behv_data = normalize(data_params['behavior_normalizor'], train_behv_data, val_behv_data, test_behv_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'] = train_behv_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']   = val_behv_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']  = test_behv_data
        #####################
    
    return train_dataset, val_dataset, test_dataset, dataset_info


def dataloader_perich_deprecated(fpath_lst, **data_params):
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
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read() 
        subject_id = nwbfile.subject.subject_id.upper()
        session_date = f"{nwbfile.session_start_time.year}{nwbfile.session_start_time.month:02}{nwbfile.session_start_time.day:02}"
        print(f'[INFO] subject {subject_id} session {session_date} data LOADED!')
        ### Get By Trial Data Sorted into a Dataframe ###
        trial_info = (
                    nwbfile.trials.to_dataframe()
                    .reset_index()
                    .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
                )
        units = nwbfile.units.to_dataframe()

        # Load descriptions of trial info fields
        descriptions = {}
        for name in nwbfile.trials.colnames:
            if not hasattr(nwbfile.trials, name):
                print(f"Field {name} not found in NWB file trials table")
                continue
            descriptions[name] = getattr(nwbfile.trials, name).description
        
        skip_fields = []
        # Create a dictionary containing DataFrames for all time series
        data_dict = _find_timeseries(nwbfile, skip_fields, descriptions)


        start_time = 0.0
        BIN = 10 # in ms
        rate = round(1000.0 / BIN, 2)  # in Hz

        end_time = round(trial_info["end_time"].iloc[-1] * rate) * BIN
        timestamps = (np.arange(start_time, end_time, BIN) / 1000).round(6)
        timestamps_td = pd.to_timedelta(timestamps, unit="s")

        # Assuming units is the DataFrame and timestamps is the numpy array
        T = len(timestamps)  # Number of time bins (size of timestamps array)
        N = len(units)       # Number of neurons (rows in units DataFrame)
        # Initialize the spike binning matrix (T x N)
        spike_binning_matrix = np.zeros((T, N), dtype=int)

        # Iterate over each neuron
        for neuron_idx,(_, neuron) in enumerate(units.iterrows()):
            # Get the spike times for the current neuron
            spike_times = neuron['spike_times']
            
            # Digitize the spike times into the time bins defined by timestamps
            # Note: bins for digitize should cover the entire range and include an additional upper bound
            time_bins = np.concatenate(([timestamps[0] - (timestamps[1] - timestamps[0])], timestamps))
            spike_indices = np.digitize(spike_times, time_bins) - 1  # `-1` to make it 0-based index
            
            # Count spikes in each time bin
            for idx in spike_indices:
                if 0 <= idx < T:  # Ensure index is within bounds
                    spike_binning_matrix[idx, neuron_idx] += 1

        # Convert to a Pandas DataFrame if needed
        spike_binning_df = pd.DataFrame(spike_binning_matrix, index=timestamps_td)
        data_dict['spikes'] = spike_binning_df
        print(f'[INFO] Spike counts processed!')
        
        def _make_midx(signal_type, chan_names=None, num_channels=None):
            if chan_names is None:
                if "rates" in signal_type:
                    # If merging rates, use the same names as the spikes
                    chan_names = spike_binning_df.columns
                else:
                    # Otherwise, generate names for the channels
                    assert (
                        num_channels is not None
                    ), "`num_channels` must be provided if `chan_names` is not provided"
                    chan_names = [f"{i:04d}" for i in range(num_channels)]
            # Create the MultiIndex for this data
            midx = pd.MultiIndex.from_product(
                [[signal_type], chan_names], names=("signal_type", "channel")
            )
            return midx
        
        data_lst = []

        for key, val in data_dict.items():
            chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
            val.columns = _make_midx(
                        key, chan_names=chan_names, num_channels=val.shape[1]
                    )
            
            # num_channels = val.shape[1]
            # chan_names = [f"{i:04d}" for i in range(num_channels)]
            # midx = pd.MultiIndex.from_product(
            #         [[key], chan_names], names=("signal_type", "channel")
            #     )
            data_lst.append(val)

        data = pd.concat(data_lst, axis=1)
        data.index.name = "clock_time"
        data.sort_index(axis=1, inplace=True)
        data = data[:T]

        trial_info = trial_info.apply(_to_td, axis=0)

        start_field = 'start_time'
        end_field   = 'end_time'
        
        align_field = data_params['align_field'] # None#'target_on_time' 
        align_range = data_params['align_range'] # (None, None)
        margin      = data_params['margin']      # 0
        allow_overlap = False
        allow_nans    = False
        ignored_trials = data_params['ignored_trials'] # None - here you can filter out failed trials
        if ignored_trials is not None:
            ignored_trials = trial_info['result'].isin(ignored_trials)
        
        trial_info["next_start"] = trial_info["start_time"].shift(-1)
        if ignored_trials is not None:
            trial_info = trial_info.loc[~ignored_trials]


        bin_width = pd.to_timedelta(BIN, unit="ms")

        trial_start = trial_info[start_field]
        trial_end   = trial_info[end_field]
        next_start  = trial_info["next_start"]
        if align_field is not None:
            align_left = align_right = trial_info[align_field]
        else:
            align_field = f"{start_field} and {end_field}"  # for logging
            align_left = trial_start
            align_right = trial_end

        start_offset, end_offset = pd.to_timedelta(align_range, unit="ms")
        if not pd.isnull(start_offset) and not pd.isnull(end_offset):
            if not ((end_offset - start_offset) / bin_width).is_integer():
                # Round align offsets if alignment range is not multiple of bin width
                end_offset = start_offset + (end_offset - start_offset).round(bin_width)
                align_range = (
                    int(round(start_offset.total_seconds() * 1000)),
                    int(round(end_offset.total_seconds() * 1000)),
                )

        if pd.isnull(start_offset):
            align_start = trial_start
        else:
            align_start = align_left + start_offset
        if pd.isnull(end_offset):
            # Subtract small interval to prevent inclusive timedelta .loc indexing
            align_end = trial_end - pd.to_timedelta(1, unit="us")
        else:
            align_end = align_right + end_offset - pd.to_timedelta(1, unit="us")

        # Add margins to either end of the data
        margin_delta = pd.to_timedelta(margin, unit="ms")
        margin_start = align_start - margin_delta
        margin_end = align_end + margin_delta

        trial_ids = trial_info["trial_id"]

        # Store the alignment data in a dataframe
        align_data = pd.DataFrame(
            {
                "trial_id": trial_ids,
                "margin_start": margin_start,
                "margin_end": margin_end,
                "align_start": align_start,
                "align_end": align_end,
                "trial_start": trial_start,
                "align_left": align_left,
            }
        ).dropna()
        # Bound the end by the next trial / alignment start
        align_data["end_bound"] = (
            pd.concat([next_start, align_start], axis=1).min(axis=1).shift(-1)
        )
        trial_dfs = []
        num_overlap_trials = 0

        def make_trial_df(args):
            idx, row = args
            # Handle overlap with the start of the next trial
            endpoint = row.margin_end
            trial_id = row.trial_id
            overlap = False
            if not pd.isnull(row.end_bound) and row.align_end > row.end_bound:

                overlap = True
                if not allow_overlap:
                    # Allow overlapping margins, but not aligned data
                    endpoint = (
                        row.end_bound + margin_delta - pd.to_timedelta(1, unit="us")
                    )
            # Take a slice of the continuous data
            trial_idx = pd.Series(
                data.index[
                    data.index.slice_indexer(row.margin_start, endpoint)
                ]
            )
            # Add trial identifiers
            trial_df = pd.DataFrame(
                {
                    ("trial_id", ""): np.repeat(trial_id, len(trial_idx)),
                    ("trial_time", ""): (trial_idx - row.trial_start.ceil(bin_width)),
                    ("align_time", ""): (trial_idx - row.align_left.ceil(bin_width)),
                    ("margin", ""): (
                        (trial_idx < row.align_start) | (row.align_end < trial_idx)
                    ),
                }
            )
            trial_df.index = trial_idx
            return overlap, trial_df

        overlaps, trial_dfs = zip(
            *[make_trial_df(args) for args in align_data.iterrows()]
        )
        num_overlap_trials = sum(overlaps)

        # Report any overlapping trials to the user.
        if num_overlap_trials > 0:
            if allow_overlap:
                print(f"[INFO] Allowed {num_overlap_trials} overlapping trials.")
            else:
                print(
                    f"[INFO] Shortened {num_overlap_trials} trials to prevent overlap."
                )

        trial_data = pd.concat(trial_dfs)
        trial_data.reset_index(inplace=True)
        trial_data = trial_data.merge(
            data, how="left", left_on=[("clock_time", "")], right_index=True
        )
        # Sanity check to make sure there are no duplicated `clock_time`'s
        if not allow_overlap:
            # Duplicated points in the margins are allowed
            td_nonmargin = trial_data[~trial_data.margin]
            assert (
                td_nonmargin.clock_time.duplicated().sum() == 0
            ), "[DEBUG] Duplicated points still found. Double-check overlap code."
        # Make sure NaN's caused by adding trialized data to self.data are ignored
        nans_found = trial_data.isnull().sum().max()
        if nans_found > 0:
            pct_nan = (nans_found / len(trial_data)) * 100
            if allow_nans:
                print(f"[INFO] NaNs found in {pct_nan:.2f}% of `trial_data`.")
            else:
                print(
                    f"[INFO] NaNs found in `self.data`. Dropping {pct_nan:.2f}% "
                    "of points to remove NaNs from `trial_data`."
                )
                trial_data = trial_data.dropna()
        trial_data.sort_index(axis=1, inplace=True)
        print(f'[INFO] Trial data processed!')
        ############################################
        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]  = {'train_trials': [], 'val_trials': [], 'test_trials': [], 'obs_dim': None}
        ### Get the train-val-test split based on trials ###
        trials = trial_data['trial_id'].unique()
        train_trials, test_trials = train_test_split(trials, test_size=data_params['test_size'])
        if data_params['val_size'] != 0:
            train_trials, val_trials = train_test_split(train_trials, test_size=data_params['val_size'])
        else:
            val_trials = []
        ###################################################
        obs_dim = len(trial_data['spikes'].columns)
        dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['obs_dim'] = obs_dim 
        for t in trials:
            df_trial_t = trial_data[(trial_data['trial_id'] == t)]
            trial_t_len = len(df_trial_t)
            if trial_t_len < data_params['min_trial_len'] or trial_t_len > data_params['max_trial_len']:
                continue
            ### Get the trial label ###
            uniq_dir = trial_info[trial_info['trial_id']==t]['target_dir'].values[0]
            uniq_dir = uniq_dir / (2*np.pi) * 360
            ###########################

            ### Get Neural Data ###
            df_trial_t_neurons = df_trial_t['spikes']
            X_spike = df_trial_t_neurons.to_numpy()
            # Smooth the data
            gauss_bin_std = data_params['gauss_width'] / BIN 
            win_len = int(6 * gauss_bin_std)
            # Create Gaussian kernel
            window = signal.windows.gaussian(win_len, gauss_bin_std, sym=True)
            window /= np.sum(window)

            X = np.apply_along_axis(lambda x: smooth_column((x, window, True, 'float64')), 0, X_spike)
            #######################
            
            ### Get Behaviral Data ###
            y = df_trial_t[data_params['behavior_columns']].to_numpy()
            ##########################

            if t in train_trials:
                train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'].append(X)
                train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'].append(y)
                train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_type'].append(uniq_dir)
                train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_length'].append(trial_t_len)
                dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['train_trials'].append(t)
            elif t in val_trials:
                val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'].append(X)
                val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'].append(y)
                val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_type'].append(uniq_dir)
                val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_length'].append(trial_t_len)
                dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['val_trials'].append(t)
            elif t in test_trials:
                test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'].append(X)
                test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'].append(y)
                test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_type'].append(uniq_dir)
                test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['trials_length'].append(trial_t_len)
                dataset_info[(f'Animal-{subject_id}', f'Session-{session_date}')]['test_trials'].append(t)
            else:
                raise Exception(f'trial {t} is not in none of the train-val-test split')
            
        ### Normalize data ###
        # normalize neural data
        train_neural_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        val_neural_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        test_neural_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']
        
        train_neural_data, val_neural_data, test_neural_data = normalize(data_params['neural_normalizor'], train_neural_data, val_neural_data, test_neural_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data'] = train_neural_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']   = val_neural_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['neural_data']  = test_neural_data
        # normalize behavioral data
        train_behv_data = train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        val_behv_data   = val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']
        test_behv_data  = test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']

        train_behv_data, val_behv_data, test_behv_data = normalize(data_params['behavior_normalizor'], train_behv_data, val_behv_data, test_behv_data)

        train_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data'] = train_behv_data
        val_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']   = val_behv_data
        test_dataset[(f'Animal-{subject_id}', f'Session-{session_date}')]['behavior_data']  = test_behv_data
        #####################
    
    return train_dataset, val_dataset, test_dataset, dataset_info

###### Helper Functions #######

def _to_td(x):
    if x.name.endswith("_time"):
        return pd.to_timedelta(x, unit="s")
    else:
        return x

def _make_df(ts):
    """Converts TimeSeries into pandas DataFrame"""
    if ts.timestamps is not None:
        index = ts.timestamps[()]
    else:
        index = np.arange(ts.data.shape[0]) / ts.rate + ts.starting_time
    columns = (
        ts.comments.split("[")[-1].split("]")[0].split(",")
        if "columns=" in ts.comments
        else None
    )
    df = pd.DataFrame(
        ts.data[()], index=pd.to_timedelta(index, unit="s"), columns=columns
    )
    return df

def _find_timeseries(nwbobj, skip_fields, descriptions):
    """Recursively searches the NWB file for time series data"""
    ts_dict = {}
    for child in nwbobj.children:
        if isinstance(child, TimeSeries):
            if child.name in skip_fields:
                continue
            ts_dict[child.name] = _make_df(child)
            descriptions[child.name] = child.description
        elif isinstance(child, ProcessingModule):
            pm_dict = _find_timeseries(child, skip_fields, descriptions)
            ts_dict.update(pm_dict)
        elif isinstance(child, MultiContainerInterface):
            for field in child.children:
                if isinstance(field, TimeSeries):
                    name = child.name + "_" + field.name
                    if name in skip_fields:
                        continue
                    ts_dict[name] = _make_df(field)
                    descriptions[name] = field.description
    return ts_dict

def dataloader_neurotask(fpath, **data_params):
    """
    INPUT
        [data_params]: 
            - seed  : int
            - rebin : bool
            - align : bool
            - bin_size : int
            - align_event : string
            - offset_min / offset_max : int
            - behavior_columns : a list 
            - gauss_width : int, in ms
            - train_size
            - val_size
            - test_size
    OUTPUT
        [train_dataset] : a dictionary of entry (animal_id, session_id). Each value is another dictionary:
                          {neural_data : a list of matrix,
                           behavior_data: a list of matrix,
                           trials_type: a list of int, which correponds to the degree,
                           trials_length: a list of int, which corresponds to the trial length
                          }
                          where each matrix is T_i x N and there are num_trial such matrix
        [val_dataset]   
        [test_dataset]
        [dataset_info]: a dictionary of key animal and value as sessions
    """
    seed = data_params['seed']
    np.random.seed(seed)

    data = nap.load_file(fpath)
    df , BIN = get_dataframe(data,filter_result=[b'R'])
    BIN = int(BIN)
    if data_params['rebin']:
        df = rebin(df, prev_bin_size=BIN, new_bin_size=data_params['bin_size'])
        BIN = data_params['bin_size']
    if data_params['align']:
        df = align_event(df, start_event=data_params['align_event'], bin_size=data_params['bin_size'],\
                         offset_min=data_params['offset_min'],offset_max=data_params['offset_max'])
        BIN = data_params['bin_size']
    
    train_dataset = {}
    val_dataset   = {}
    test_dataset  = {}
    animals = df['animal'].unique() 

    for a in [2]:
        sessions = df[(df['animal'] == a)]['session'] .unique()
        for s in [2]:
            train_dataset[(f'Animal-{a}', f'Session-{s}')] = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
            val_dataset[(f'Animal-{a}', f'Session-{s}')]   = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}
            test_dataset[(f'Animal-{a}', f'Session-{s}')]  = {'neural_data': [], 'behavior_data': [], 'trials_type': [], 'trials_length': []}

            trials = df[(df['animal'] == a) & (df['session'] == s)]['trial_id'].unique()
            ### Get the train-val-test split based on trials ###
            train_trials, test_trials = train_test_split(trials, test_size=data_params['test_size'])
            if data_params['val_size'] != 0:
                train_trials, val_trials = train_test_split(train_trials, test_size=data_params['val_size'])
            else:
                val_trials = []
            ###################################################
            df_ani_sess = df[(df['animal'] == a) & (df['session'] == s)]
            # Extract all Neuron columns
            neurons = [col for col in df_ani_sess.columns if col.startswith('Neuron')]
            # Identify Neuron columns without NaN values
            filtered_neurons = df_ani_sess[neurons].dropna(axis=1).columns
            # Identify all other columns that do not start with 'Neuron'
            non_neuron_columns = [col for col in df_ani_sess.columns if not col.startswith('Neuron')]
            # Keep all non-Neuron columns and the filtered Neuron columns
            df_ani_sess = df_ani_sess[non_neuron_columns + list(filtered_neurons)]
            obs_dim = len(filtered_neurons)
            train_dataset[(f'Animal-{a}', f'Session-{s}')]['obs_dim'] = obs_dim 
            val_dataset[(f'Animal-{a}', f'Session-{s}')]['obs_dim']   = obs_dim 
            test_dataset[(f'Animal-{a}', f'Session-{s}')]['obs_dim']  = obs_dim
            for t in trials:
                df_trial_t = df_ani_sess[(df_ani_sess['trial_id'] == t)]
                trial_t_len = len(df_trial_t)
                if trial_t_len < data_params['min_trial_len'] or trial_t_len > data_params['max_trial_len']:
                    continue
                ### Get the trial label ###
                uniq_dir = df_trial_t['target_dir'].unique()
                assert len(uniq_dir) == 1, 'the target direction for one trial should be consistent'
                uniq_dir = uniq_dir[0]
                uniq_dir = uniq_dir / (2*np.pi) * 360
                ###########################

                ### Get Neural Data ###
                neurons = [col for col in df_trial_t.columns if col.startswith('Neuron')]
                df_trial_t_neurons = df_trial_t[neurons]
                df_trial_t_neurons = df_trial_t_neurons.loc[:,:] #(df_trial_t_neurons != 0).any(axis=0)]
                X_spike = df_trial_t_neurons.to_numpy()
                # Smooth the data
                gauss_bin_std = data_params['gauss_width'] / BIN 
                win_len = int(6 * gauss_bin_std)
                # Create Gaussian kernel
                window = signal.windows.gaussian(win_len, gauss_bin_std, sym=True)
                window /= np.sum(window)

                X = np.apply_along_axis(lambda x: smooth_column((x, window, True, 'float64')), 0, X_spike)
                #######################

                ### Get Behaviral Data ###
                y = np.array(df_trial_t[data_params['behavior_columns']])
                ##########################

                if t in train_trials:
                    train_dataset[(f'Animal-{a}', f'Session-{s}')]['neural_data'].append(X)
                    train_dataset[(f'Animal-{a}', f'Session-{s}')]['behavior_data'].append(y)
                    train_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_type'].append(uniq_dir)
                    train_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_length'].append(trial_t_len)
                elif t in val_trials:
                    val_dataset[(f'Animal-{a}', f'Session-{s}')]['neural_data'].append(X)
                    val_dataset[(f'Animal-{a}', f'Session-{s}')]['behavior_data'].append(y)
                    val_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_type'].append(uniq_dir)
                    val_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_length'].append(trial_t_len)
                elif t in test_trials:
                    test_dataset[(f'Animal-{a}', f'Session-{s}')]['neural_data'].append(X)
                    test_dataset[(f'Animal-{a}', f'Session-{s}')]['behavior_data'].append(y)
                    test_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_type'].append(uniq_dir)
                    test_dataset[(f'Animal-{a}', f'Session-{s}')]['trials_length'].append(trial_t_len)
                else:
                    raise Exception(f'trial {t} is not in none of the train-val-test split')
                
    return train_dataset, val_dataset, test_dataset

def smooth_column(args):
    """Low-level helper function for smoothing single column

    Parameters
    ----------
    args : tuple
        Tuple containing data to smooth in 1d array,
        smoothing kernel in 1d array, whether to
        ignore nans, and data dtype

    Returns
    -------
    np.ndarray
        1d array containing smoothed data
    """
    x, window, ignore_nans, dtype = args
    if ignore_nans and np.any(np.isnan(x)):
        x.astype(dtype)
        # split continuous data into NaN and not-NaN segments
        splits = np.where(np.diff(np.isnan(x)))[0] + 1
        seqs = np.split(x, splits)
        # if signal.convolve uses fftconvolve, there may be small negative values
        def rectify(arr):
            arr[arr < 0] = 0
            return arr

        # smooth only the not-NaN data
        seqs = [
            seq
            if np.any(np.isnan(seq))
            else rectify(signal.convolve(seq, window, "same"))
            for seq in seqs
        ]
        # concatenate to single array
        y = np.concatenate(seqs)
    else:
        y = signal.convolve(x.astype(dtype), window, "same")
    return y
