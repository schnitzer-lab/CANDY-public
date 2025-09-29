import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch 
from torch.utils.data import DataLoader

from src.data_loader.datasets import load_rnndataset
from src.dynamics import *

def init_model(model_type, obs_dim, latent_dim, **kwargs):
    match model_type:
        case 'CANDY':
            model = CANDYDynamics(obs_dim_list=obs_dim, latent_dim=latent_dim, **kwargs) # obs_dim_lst
    return model

def init_dataloaderN(train_dataset, val_dataset, test_dataset, dataset_info, **params):
    model_type = params['model_type']
    model_params = params['model_params']
    train_params = params['train_params']
    data_params = params['data_params']
    obs_dim_lst = []
    data_dict = {'train_loader': [], 'valid_loader': [], 'test_loader': [], 
                 'train_dataset': [], 'valid_dataset': [], 'test_dataset': []}
    for i, sess_name in enumerate(dataset_info.keys()):
        train_dataset_one = train_dataset[sess_name]
        val_dataset_one   = val_dataset[sess_name]
        test_dataset_one  = test_dataset[sess_name]
        
        if model_type in ['CANDY']:
            train_dataset_one = load_rnndataset(train_dataset_one, frac=data_params['train_frac'] if 'train_frac' in data_params else 1.0)
            val_dataset_one   = load_rnndataset(val_dataset_one, frac=1.0) 
            test_dataset_one  = load_rnndataset(test_dataset_one, frac=1.0)
            batch_size = model_params['batch_size']
            shuffle_train = train_params['shuffle_train']
            assert batch_size is not None, 'batch size need to be specified'
            train_loader_one = DataLoader(train_dataset_one, batch_size=batch_size, shuffle=shuffle_train)
            val_loader_one   = DataLoader(val_dataset_one, batch_size=batch_size, shuffle=False)
            test_loader_one  = DataLoader(test_dataset_one, batch_size=batch_size, shuffle=False)
        else:
            raise NotImplementedError('model_type is not implemented')
        data_dict['train_loader'].append(train_loader_one)
        data_dict['valid_loader'].append(val_loader_one)
        data_dict['test_loader'].append(test_loader_one)
        data_dict['train_dataset'].append(train_dataset_one)
        data_dict['valid_dataset'].append(val_dataset_one)
        data_dict['test_dataset'].append(test_dataset_one)
        obs_dim_lst.append(dataset_info[sess_name]['obs_dim'])
    return data_dict, obs_dim_lst

def get_latest_checkpoint_number(ckpt_save_dir):
    # Step 1: Check if the folder exists
    if not os.path.exists(ckpt_save_dir):
        print(f"The directory {ckpt_save_dir} does not exist.")
        return None

    # Step 2: List all files in the directory
    files = os.listdir(ckpt_save_dir)

    # Step 3: Filter files that match the pattern {number}_ckpt.pth
    checkpoint_files = [f for f in files if f.endswith('_ckpt.pth') and f.split('_')[0].isdigit()]

    if not checkpoint_files:
        print(f"No checkpoint files found in {ckpt_save_dir}.")
        return None

    # Step 4: Extract the numbers and find the largest one
    numbers = [int(f.split('_')[0]) for f in checkpoint_files]
    latest_number = max(numbers)

    return latest_number

def init_csv(dataset_info): 
    columns = ["model_name", "data_seed", "model_seed"]
    for i, sess_name in enumerate(dataset_info.keys()):
        columns += [f'{sess_name}']
    # Define sub-columns for each dataset column
    sub_columns = [
        "recon_MSE", "recon_MAE", "recon_R2", "recon_Corr",
        "decoder_MSE", "decoder_MAE", "decoder_R2", "decoder_Corr",
        "unidecoder_MSE", "unidecoder_MAE", "unidecoder_R2", "unidecoder_Corr",
        "supdecoder_MSE", "supdecoder_MAE", "supdecoder_R2", "supdecoder_Corr"
    ]

    # Create a multi-index for the dataframe
    multi_columns = pd.MultiIndex.from_tuples(
        [("model_name", ""), ("data_seed", ""), ("model_seed", "")] + [
            (col, sub) for col in columns[3:] for sub in sub_columns
        ],
        names=["Data", "Metrics"]
    )

    # Initialize the DataFrame
    df = pd.DataFrame(columns=multi_columns)
    return df

def get_model_name(params):
    latent_dim  = params['latent_dim']
    data_params = params['data_params']
    model_type  = params['model_type']
    model_args  = params['model_params']
    train_args  = params['train_params']

    data_seed  = data_params['seed']
    model_seed = model_args['seed']
    match model_type:
        case 'CANDY':
            lr_init           = model_args['lr_init']
            scale_l2          = model_args['scale_l2']
            supervise_behv    = model_args['supervise_behv']
            scale_behv_recons = model_args['scale_behv_recons']
            steps_ahead       = model_args['steps_ahead']
            hidden_layer_lst  = model_args['hidden_layer_lst']
            activation        = model_args['activation']
            contrastive       = model_args['contrastive']
            contrastive_temp  = model_args['contrastive_temp']
            contrastive_label_diff  = model_args['contrastive_label_diff']
            contrastive_feature_sim = model_args['contrastive_feature_sim']
            contrastive_scale       = model_args['contrastive_scale']
            contrastive_time_scaler = model_args['contrastive_time_scaler']
            contrastive_num_batch   = model_args['contrastive_num_batch']
            batch_size              = model_args['batch_size']
            model_name = f"{model_type}-latent_dim{latent_dim}-seed{model_seed}" +\
                        f"-contrastive{contrastive}-supbev{supervise_behv}-batch{batch_size}-cbatch{contrastive_num_batch}" #+\
                        # f"-lr_init{lr_init}-scalel2-{scale_l2}-activation{activation}-scalebehvrecons{scale_behv_recons}" +\
                        # f"-stepsahead{'_'.join([str(s) for s in steps_ahead])}-hiddenlayers{'_'.join([str(l) for l in hidden_layer_lst])}"
    return model_name

def get_save_path_name(data_path_lst):
    name = ''
    for file in data_path_lst:
        name += file[-19:-8]
        name += '-'
    name = name[:-1]
    return name
