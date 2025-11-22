import os 
import pickle
import sys
sys.path.insert(0, '../../') 
import itertools
import wandb

import numpy as np
import pandas as pd 

import torch

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from src.data_loader.dataloader_mouse import dataloader_mouse_wheel
from src.data_loader.dataloader_monkey import dataloader_perich
from src.runner.utils import init_model, init_dataloaderN, init_csv
from src.runner.utils import get_model_name, get_latest_checkpoint_number
from src.decoding.utils import train_behv_decoder, update_csv_behv
from src.plotting.utils import plot_losses, get_vis_data, plot_latent_vis

def runnerN(data_path_lst, params):
    num_agents = len(data_path_lst)

    # Initialize all the training parameters
    latent_dim  = params['latent_dim']
    data_params = params['data_params']
    model_type   = params['model_type']
    model_params = params['model_params']
    train_params = params['train_params']
    decoder_params = params['decoder_params']
    decoder_train_params = params['decoder_train_params']
    save_params = params['save_params']
    data_type = params['data_type']
    brain_areas = params['brain_areas']
    area = "_".join(brain_areas)

    model_name = get_model_name(params)

    parent_folder = f"{save_params['save_path']}/{area}/{model_name}/frac_{data_params['train_frac']}"

    # Set seed
    np.random.seed(seed=data_params['seed'])
    torch.manual_seed(data_params['seed'])

    # Initialize dataset (dictionary with key [Animal, Session] and value as dictionary of neural/behv/trialtype data)
    if data_type in ['mouse_wheel']:
        train_dataset, valid_dataset, test_dataset, dataset_info = dataloader_mouse_wheel(data_path_lst, **data_params)
    elif data_type in ['perich_monkey']:
        train_dataset, valid_dataset, test_dataset, dataset_info = dataloader_perich(data_path_lst, **data_params)

    session_names = "+".join(f"{sess_name}" for sess_name in dataset_info.keys()) + ".pkl"

    # Initialize the csv files
    df_train = init_csv(dataset_info)
    df_test  = init_csv(dataset_info)
    df_index = 0

    df_train.loc[df_index, 'model_name'] = model_name
    df_train.loc[df_index, 'data_seed']  = data_params['seed']
    df_test.loc[df_index, 'model_name'] = model_name
    df_test.loc[df_index, 'data_seed']  = data_params['seed']

    model_params['seed'] = model_params['seed']
    params['model_params'] = model_params
    # Inidialize the csv file 
    df_train.loc[df_index, 'model_seed'] = model_params['seed'] 
    df_test.loc[df_index, 'model_seed']  = model_params['seed']
    
    # Initialize the data buffer
    z_hat_train_dict = {}
    z_hat_valid_dict = {}
    z_hat_test_dict  = {}
    a_hat_train_dict = {}
    a_hat_valid_dict = {}
    a_hat_test_dict  = {}
    behv_train_dict = {}
    behv_valid_dict = {}
    behv_test_dict  = {}
    behv_hat_train_dict = {}
    behv_hat_valid_dict = {}
    behv_hat_test_dict  = {}
    trials_type_train_dict = {}
    trials_type_valid_dict = {}
    trials_type_test_dict  = {}

    # Initilize ckpt saving buffer
    os.makedirs(f"{parent_folder}/", exist_ok=True)

    try:
        with open(f"{parent_folder}/results.pkl", 'rb') as f:
            res = pickle.load(f)
            print(f'[INFO] {session_names} pickle file exist!')
        with open(f"{parent_folder}/flag.pkl", 'rb') as f:
            res_flag = pickle.load(f)
            print(f'[INFO] {session_names} pickle file saved correctly!')
        
        if res_flag['flag']:
            print(f'[INFO] Loading {session_names} dataset.')
            model = res['model']
            z_hat_train_dict = res['z_hat_train_dict']
            z_hat_valid_dict = res['z_hat_valid_dict']
            z_hat_test_dict  = res['z_hat_test_dict']
            a_hat_train_dict = res['a_hat_train_dict']
            a_hat_valid_dict = res['a_hat_valid_dict']
            a_hat_test_dict  = res['a_hat_test_dict']
            behv_train_dict  = res['behv_train_dict']
            behv_valid_dict  = res['behv_valid_dict']
            behv_test_dict   = res['behv_test_dict']
            behv_hat_train_dict = res['behv_hat_train_dict']
            behv_hat_valid_dict = res['behv_hat_valid_dict']
            behv_hat_test_dict  = res['behv_hat_test_dict']
            trials_type_train_dict = res['trials_type_train_dict']
            trials_type_valid_dict = res['trials_type_valid_dict']
            trials_type_test_dict  = res['trials_type_test_dict']
            train_recon = res['train_recon']
            valid_recon = res['valid_recon']
            test_recon  = res['test_recon']
            train_dataset = res['train_dataset']
            valid_dataset = res['valid_dataset']
            test_dataset  = res['test_dataset']
            df_train = res['df_train']
            df_test  = res['df_test']
            df_index = res['df_index']
        else:
            raise FileNotFoundError
    except (FileNotFoundError, FileExistsError, KeyError):
        # Step 1: Create dataloader
        data_dict, obs_dim_lst = init_dataloaderN(train_dataset, valid_dataset, test_dataset, dataset_info, **params)
        train_loader = data_dict['train_loader']
        val_loader   = data_dict['valid_loader']
        test_loader  = data_dict['test_loader']

        # Step 2: Train the Model
        if train_params['wandb']:
            wandb.init(
                project="shared-dynamics",   # Project name
                name=f"{area}/{model_name}", # Experiment name
                config=params,
            )

        model_params['ckpt_save_dir'] = f'{parent_folder}/'
        decoder_params['ckpt_save_dir'] = f'{parent_folder}/'
        model = init_model(model_type, obs_dim_lst, latent_dim, **model_params)
        
        ckpt_dir =  model_params['ckpt_save_dir'] + '/ckpts'
        ckpt_num = get_latest_checkpoint_number(ckpt_dir)
        if ckpt_num is not None:
            model.config['load']['ckpt'] = ckpt_num
            print(f'[INFO] checkpoint {ckpt_num} exist. Start from this epoch and continue training.')
            model.config.load.resume_train = True
            model._load_ckpt(model.candy_list, model.ldm, model.mapper, model.optimizer)

        model.fit(train_loader, val_loader, **train_params)
        model.config['load']['ckpt'] = 'best_loss'
        model._load_ckpt(model.candy_list, model.ldm, model.mapper, model.optimizer)
        data_orig_train, train_recon, z_hat_train_lst, a_hat_train_lst, behv_train_lst, behv_hat_train_lst, trials_type_train_lst = model.transform(train_loader)
        data_orig_valid, valid_recon, z_hat_valid_lst, a_hat_valid_lst, behv_valid_lst, behv_hat_valid_lst, trials_type_valid_lst = model.transform(val_loader)
        data_orig_test, test_recon, z_hat_test_lst, a_hat_test_lst, behv_test_lst, behv_hat_test_lst, trials_type_test_lst = model.transform(test_loader)

        recon_score_train_lst = model.scoring(data_orig_train, train_recon)
        recon_score_test_lst  = model.scoring(data_orig_test, test_recon)

        for i, sess_name in enumerate(dataset_info):   
            # update the original dataset given that the Torch Dataset might be shuffled 
            train_dataset[sess_name]['behavior_data'] = behv_train_lst[i]
            valid_dataset[sess_name]['behavior_data'] = behv_valid_lst[i]
            test_dataset[sess_name]['behavior_data']  = behv_test_lst[i]
            train_dataset[sess_name]['neural_data'] = data_orig_train[i]
            valid_dataset[sess_name]['neural_data'] = data_orig_valid[i]
            test_dataset[sess_name]['neural_data']  = data_orig_test[i]
            train_dataset[sess_name]['trials_type'] = trials_type_train_lst[i]
            valid_dataset[sess_name]['trials_type'] = trials_type_valid_lst[i]
            test_dataset[sess_name]['trials_type']  = trials_type_test_lst[i]
            # update the results buffer
            z_hat_train_dict[sess_name] = z_hat_train_lst[i]
            z_hat_valid_dict[sess_name] = z_hat_valid_lst[i]
            z_hat_test_dict[sess_name]  = z_hat_test_lst[i]
            a_hat_train_dict[sess_name] = a_hat_train_lst[i]
            a_hat_valid_dict[sess_name] = a_hat_valid_lst[i]
            a_hat_test_dict[sess_name]  = a_hat_test_lst[i]
            behv_train_dict[sess_name] = behv_train_lst[i]
            behv_valid_dict[sess_name] = behv_valid_lst[i]
            behv_test_dict[sess_name]  = behv_test_lst[i]
            behv_hat_train_dict[sess_name] = behv_hat_train_lst[i]
            behv_hat_valid_dict[sess_name] = behv_hat_valid_lst[i]
            behv_hat_test_dict[sess_name]  = behv_hat_test_lst[i]
            trials_type_train_dict[sess_name] = trials_type_train_lst[i]
            trials_type_valid_dict[sess_name] = trials_type_valid_lst[i]
            trials_type_test_dict[sess_name]  = trials_type_test_lst[i]
            # update the dataframe for metrics recording
            df_train.loc[df_index, (f'{sess_name}', "recon_MSE")] = recon_score_train_lst[i]['MSE']
            df_train.loc[df_index, (f'{sess_name}', "recon_MAE")] = recon_score_train_lst[i]['MAE']
            df_train.loc[df_index, (f'{sess_name}', "recon_R2")]  = recon_score_train_lst[i]['R2']
            df_train.loc[df_index, (f'{sess_name}', "recon_Corr")]= recon_score_train_lst[i]['Corr']

            df_test.loc[df_index, (f'{sess_name}', "recon_MSE")] = recon_score_test_lst[i]['MSE']
            df_test.loc[df_index, (f'{sess_name}', "recon_MAE")] = recon_score_test_lst[i]['MAE']
            df_test.loc[df_index, (f'{sess_name}', "recon_R2")]  = recon_score_test_lst[i]["R2"]
            df_test.loc[df_index, (f'{sess_name}', "recon_Corr")]= recon_score_test_lst[i]['Corr']
        
        # Save pickle files
        with open(f"{parent_folder}/results.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'dataset_info': dataset_info,
                'data_path_lst': data_path_lst,
                'params': params,
                'z_hat_train_dict': z_hat_train_dict, 
                'z_hat_valid_dict': z_hat_valid_dict,
                'z_hat_test_dict' : z_hat_test_dict,
                'a_hat_train_dict': a_hat_train_dict,
                'a_hat_valid_dict': a_hat_valid_dict,
                'a_hat_test_dict' : a_hat_test_dict,
                'behv_train_dict': behv_train_dict, 
                'behv_valid_dict': behv_valid_dict,
                'behv_test_dict' : behv_test_dict,
                'behv_hat_train_dict': behv_hat_train_dict,
                'behv_hat_valid_dict': behv_hat_valid_dict,
                'behv_hat_test_dict' :  behv_hat_test_dict,
                'trials_type_train_dict': trials_type_train_dict, 
                'trials_type_valid_dict': trials_type_valid_dict,
                'trials_type_test_dict' : trials_type_test_dict,
                'train_recon': train_recon,
                'valid_recon': valid_recon,
                'test_recon': test_recon,
                'train_dataset': train_dataset,
                'valid_dataset': valid_dataset,
                'test_dataset': test_dataset,
                'df_train': df_train, 
                'df_test': df_test,
                'df_index': df_index
            }, f)
        with open(f"{parent_folder}/flag.pkl", 'wb') as f:
            pickle.dump({'flag': True}, f)
            print(f'[INFO] {session_names} saved successfully!')
        wandb.finish()

    # Step 3: Behavior decoding results
    a_hat_dict = {'train': a_hat_train_dict, 'valid': a_hat_valid_dict, 'test': a_hat_test_dict}
    behv_dict  = {'train': behv_train_dict, 'valid': behv_valid_dict, 'test': behv_test_dict}
    behv_hat_dict = {'train': behv_hat_train_dict, 'valid': behv_hat_valid_dict, 'test': behv_hat_test_dict}
    decoding_results_dict = train_behv_decoder(dataset_info, a_hat_dict, behv_dict, behv_hat_dict, params)
    update_csv_behv(df_train, df_test, df_index, dataset_info, decoding_results_dict)

    # Step 3: Plot loss functions
    if save_params['save_losses']:
        plot_losses(model, parent_folder)

    # Step 4: Plot latent plots
    if save_params['save_latent']:
        # Get Visualization dataset
        vis_dataset = get_vis_data(a_hat_train_dict, a_hat_test_dict, train_dataset, test_dataset, dataset_info)
        plot_latent_vis(vis_dataset, parent_folder)
    
    df_train_file_path = os.path.join(f"{parent_folder}/", "df_train.csv")
    df_test_file_path  = os.path.join(f"{parent_folder}/", "df_test.csv")
    df_train.to_csv(df_train_file_path, index=True)
    df_test.to_csv(df_test_file_path, index=True)

    df_index += 1
    return df_train, df_test