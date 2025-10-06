import os
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
import argparse

import torch 
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from src.data_loader.data_loader import NeuralData4Areas, load_rnndataset
from src.dynamics import *
from config.load_config import load_model_config, load_data_config

from src.runner.runner import runnerN
from src.data_loader.dataloader_mouse import dataloader_mouse_wheel
from src.runner.utils import init_model, init_dataloaderN, init_csv
from src.decoding.utils import train_behv_decoder, update_csv_behv
from src.runner.utils import get_model_name, get_latest_checkpoint_number
from src.plotting.utils import plot_losses, get_vis_data, plot_latent_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing script.')
    parser.add_argument('--brain_area', type=str, default=None,
                       help='Brain areas [mouse wheel] or Task type [monkey] (default: None).')
    parser.add_argument('--data_folder', type=str, default='G:\My Drive\Research',
                        help='Data parent folder (default: G:\My Drive\Research).')
    parser.add_argument('--data_config', type=str, default='../config/data_config/mouse_wheel.yaml',
                        help='Dataset configuration path (default: mouse_wheel.yaml)')
    parser.add_argument('--latent_dim', type=int, default=8,
                       help='Latent subspace dimension (default: 8).')
    parser.add_argument('--model_seed', type=int, default=0,
                        help='Model seed (default: 0)')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type (default: SHAREDDFINE).')
    parser.add_argument('--model_config',type=str, required=True,
                        help='Model type corresponding yaml configuration file path.')
    parser.add_argument('--save_path', type=str, default='./test',
                       help='Saving parent folder path (default: ./test).')
    parser.add_argument('--behv_sup_off', action='store_true', 
                        help='turn off the behavior supervision.')
    parser.add_argument('--contrastive_off', action='store_true',
                        help='turn off contrastive learning.')
    parser.add_argument('--dim_x_behv', type=int, default=None, 
                        help='Optional dimension for behavior (default: optional).')
    parser.add_argument('--train_frac', type=float, default=1.0, 
                        help='Actual fraction of training data to use (default: 1.0). Useful for comparison of performance as a function of training data size while fixing the testing data.')
    # Multi-GPU support arguments
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Enable multi-GPU training.')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). If not specified, all available GPUs will be used.')
    parser.add_argument('--data_parallel_type', type=str, default='DataParallel', choices=['DataParallel', 'DistributedDataParallel'],
                        help='Type of data parallelism to use (default: DataParallel).')
    args = parser.parse_args()
    
    # args.model_type = 'SHAREDDFINE'
    # args.latent_dim = 16
    # args.model_config = f'./config/model_config/{args.model_type.lower()}.yaml'
    # args.save_path = '../results/results_282_sa/' # str/SHAREDDFINE-latent_dim16-seed0-contrastiveTrue-supbevTrue-lr_init0.002-scalel2-0.005-activationleakyrelu-scalebehvrecons1-stepsahead1_2_3_4-hiddenlayers64_16/
    # args.data_config = './config/data_config/mouse_wheel.yaml'
    # args.brain_area = 'str'
    # args.data_folder = '../'
    # args.model_seed = 0

    data_config, data_params = load_data_config(args)
    data_path_lst = data_config['data_path_lst']
    brain_area    = data_config['brain_area']

    if 'mouse_wheel' in args.data_config:
        data_type = 'mouse_wheel'
    elif 'perich_monkey' in args.data_config:
        data_type = 'perich_monkey'
    else:
        raise Exception(f'{args.data_config} need to include valid data type.')

    model_params, train_params = load_model_config(args)
    model_params['ckpt_save_dir'] = args.save_path
    model_params['seed'] = args.model_seed
    model_params['dim_l'] = args.latent_dim

    # Handle multi-GPU arguments
    if args.multi_gpu:
        model_params['multi_gpu'] = True
        if args.gpu_ids:
            # Parse comma-separated GPU IDs
            model_params['gpu_ids'] = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        model_params['data_parallel_type'] = args.data_parallel_type
        
        # Adjust batch size for multi-GPU training if needed
        if args.gpu_ids:
            num_gpus = len([int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')])
        else:
            num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1:
            print(f"[INFO] Fine-tuning with multi-GPU support on {num_gpus} GPUs")

    decoder_params = {'normalize': False, 'tol': 1e-4, 'decoder_type': 'lasso'}
    decoder_train_params = {
                    'params_grid': {'alpha': np.arange(1e-5, 10, 0.1)},
                    'scoring' : 'neg_mean_squared_error',
                    'n_split' : 10,
                    'n_repeats' : 3,
                    'seed' : 1,
                    'train_together': False
                    }
    save_params = {
                    'grid_start'    : 0,
                    'grid_end'      : None,
                    'save_latent': True,
                    'save_losses': True,
                    'save_path': args.save_path
                    }

    data_params['train_frac'] = args.train_frac

    params = {
        'area': brain_area,
        'latent_dim' : args.latent_dim,
        'model_type': args.model_type,
        'data_type': data_type,
        'data_params': data_params,
        'model_params' : model_params,
        'train_params' : train_params,
        'decoder_params': decoder_params,
        'decoder_train_params': decoder_train_params,
        'save_params': save_params
    }

    fine_tuning_data_path_lst = [
        f'{args.data_folder}/data/shikano/aim282_nodeconv/sw25_20231129_ctx-str-cbl-rsc/train-byTrial-sw25_20231129_{brain_area}.mat',
        f'{args.data_folder}/data/shikano/aim282_nodeconv/sw25_20231130_ctx-str-cbl-rsc/train-byTrial-sw25_20231130_{brain_area}.mat',
        f'{args.data_folder}/data/shikano/aim282_nodeconv/sw25_20231201_ctx-str-cbl-rsc/train-byTrial-sw25_20231201_{brain_area}.mat',
        f'{args.data_folder}/data/shikano/aim282_nodeconv/sw25_20231202_ctx-str-cbl-rsc/train-byTrial-sw25_20231202_{brain_area}.mat',
        f'{args.data_folder}/data/shikano/aim282_nodeconv/sw25_20231203_ctx-str-cbl-rsc/train-byTrial-sw25_20231203_{brain_area}.mat',
    ]

    # ## Load pretrained model

    model_type   = params['model_type']
    model_params = params['model_params']
    train_params = params['train_params']

    latent_dim  = params['latent_dim']
    data_type = params['data_type']
    area = params['area']

    model_params['dim_x_behv'] = latent_dim
    model_name = get_model_name(params)
    parent_folder = f"{save_params['save_path']}/{area}/{model_name}/frac_1.0"
    model_params['ckpt_save_dir'] = f'{parent_folder}/'

    train_dataset, valid_dataset, test_dataset, dataset_info = dataloader_mouse_wheel(data_path_lst, **data_params)
    data_dict, obs_dim_lst = init_dataloaderN(train_dataset, valid_dataset, test_dataset, dataset_info, **params)
    model = init_model(model_type, obs_dim_lst, latent_dim, **model_params)

    # ckpt_dir =  model_params['ckpt_save_dir'] + '/ckpts'
    # ckpt_num = get_latest_checkpoint_number(ckpt_dir)

    # if ckpt_num is not None:
    #     model.config.load.resume_train = False
    #     
    #     model.config['load']['ckpt'] = ckpt_num 
    #     model._load_ckpt(model.dfine_list, model.ldm, model.ldm_individual_list, model.mapper, model.subject_discriminator, model.optimizer)
    model.config['load']['ckpt'] = 'best_loss'
    model._load_ckpt(model.dfine_list, model.ldm, model.ldm_individual_list, model.mapper, model.subject_discriminator, model.optimizer)

    # ## Create a new model for fine tuning

    parent_folder = f"{save_params['save_path']}/{area}/{model_name}/fine_tuning/frac_{data_params['train_frac']}"
    os.makedirs(parent_folder, exist_ok=True)
    model_params['ckpt_save_dir'] = f'{parent_folder}/'
    model_params['dim_x_behv'] = latent_dim

    model_params['supervise_behv'] = True

    train_dataset, valid_dataset, test_dataset, dataset_info = dataloader_mouse_wheel(fine_tuning_data_path_lst, **data_params)
    data_dict, obs_dim_lst = init_dataloaderN(train_dataset, valid_dataset, test_dataset, dataset_info, **params)
    fine_tuning_model = init_model(model_type, obs_dim_lst, latent_dim, **model_params)

    fine_tuning_model.ldm = deepcopy(model.ldm)
    fine_tuning_model.ldm_individual_list = deepcopy(model.ldm_individual_list)
    fine_tuning_model.mapper = deepcopy(model.mapper)

    # fine_tuning_model.ldm.requires_grad_ = False
    fine_tuning_model.ldm.eval()
    # fine_tuning_model.mapper.requires_grad_ = False
    fine_tuning_model.mapper.eval()
    for i in range(len(fine_tuning_model.ldm_individual_list)):
        # fine_tuning_model.ldm_individual_list[i].requires_grad_ = False
        fine_tuning_model.ldm_individual_list[i].eval()

    data_dict, obs_dim_lst = init_dataloaderN(train_dataset, valid_dataset, test_dataset, dataset_info, **params)
    train_loader = data_dict['train_loader']
    val_loader   = data_dict['valid_loader']
    test_loader  = data_dict['test_loader']

    # Initialize the csv files
    df_train = init_csv(dataset_info)
    df_valid = init_csv(dataset_info)
    df_test  = init_csv(dataset_info)
    df_index = 0

    df_train.loc[df_index, 'model_name'] = model_name
    df_train.loc[df_index, 'data_seed']  = data_params['seed']
    df_valid.loc[df_index, 'model_name'] = model_name
    df_valid.loc[df_index, 'data_seed']  = data_params['seed']
    df_test.loc[df_index, 'model_name'] = model_name
    df_test.loc[df_index, 'data_seed']  = data_params['seed']

    model_params['seed'] = model_params['seed']
    params['model_params'] = model_params
    # Inidialize the csv file 
    df_train.loc[df_index, 'model_seed'] = model_params['seed']
    df_valid.loc[df_index, 'model_seed'] = model_params['seed']
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

    fine_tuning_model.optimizer = torch.optim.Adam([param for sublist in fine_tuning_model.dfine_list for param in sublist.parameters()], lr=fine_tuning_model.config.lr.init, eps=fine_tuning_model.config.optim.eps)

    ckpt_dir =  model_params['ckpt_save_dir'] + '/ckpts'
    ckpt_num = get_latest_checkpoint_number(ckpt_dir)

    if ckpt_num is not None:
        fine_tuning_model.config.load.resume_train = True
        
        fine_tuning_model.config['load']['ckpt'] = ckpt_num
        fine_tuning_model._load_ckpt(fine_tuning_model.dfine_list, fine_tuning_model.ldm, fine_tuning_model.ldm_individual_list, fine_tuning_model.mapper, fine_tuning_model.subject_discriminator, fine_tuning_model.optimizer)
        print(f'Checkpoint loaded successfully! Restart from epoch {fine_tuning_model.start_epoch}')
    fine_tuning_model.fit(train_loader, val_loader, **train_params)

    fine_tuning_model.config['load']['ckpt'] = 'best_loss'
    fine_tuning_model._load_ckpt(fine_tuning_model.dfine_list, 
                                fine_tuning_model.ldm, 
                                fine_tuning_model.ldm_individual_list, 
                                fine_tuning_model.mapper, 
                                fine_tuning_model.subject_discriminator, 
                                fine_tuning_model.optimizer)

    data_orig_train, train_recon, z_hat_train_lst, a_hat_train_lst, behv_train_lst, behv_hat_train_lst, trials_type_train_lst = model.transform(train_loader, data_type=train_params['data_type'], do_full_inference=train_params['do_full_inference'])
    data_orig_valid, valid_recon, z_hat_valid_lst, a_hat_valid_lst, behv_valid_lst, behv_hat_valid_lst, trials_type_valid_lst = model.transform(val_loader, data_type=train_params['data_type'], do_full_inference=train_params['do_full_inference'])
    data_orig_test,  test_recon,  z_hat_test_lst,  a_hat_test_lst, behv_test_lst,   behv_hat_test_lst,  trials_type_test_lst  = model.transform(test_loader, data_type=train_params['data_type'], do_full_inference=train_params['do_full_inference'])

    recon_score_train_lst = model.scoring(data_orig_train, train_recon)
    recon_score_valid_lst = model.scoring(data_orig_valid, valid_recon)
    recon_score_test_lst  = model.scoring(data_orig_test,  test_recon)

    for i, (subject_id, session_id) in enumerate(dataset_info):   
        # update the original dataset given that the Torch Dataset might be shuffled 
        train_dataset[(subject_id, session_id)]['behavior_data'] = behv_train_lst[i]
        valid_dataset[(subject_id, session_id)]['behavior_data'] = behv_valid_lst[i]
        test_dataset[(subject_id, session_id)]['behavior_data']  = behv_test_lst[i]
        train_dataset[(subject_id, session_id)]['neural_data'] = data_orig_train[i]
        valid_dataset[(subject_id, session_id)]['neural_data'] = data_orig_valid[i]
        test_dataset[(subject_id, session_id)]['neural_data']  = data_orig_test[i]
        train_dataset[(subject_id, session_id)]['trials_type'] = trials_type_train_lst[i]
        valid_dataset[(subject_id, session_id)]['trials_type'] = trials_type_valid_lst[i]
        test_dataset[(subject_id, session_id)]['trials_type']  = trials_type_test_lst[i]
        # update the results buffer
        z_hat_train_dict[(subject_id, session_id)] = z_hat_train_lst[i]
        z_hat_valid_dict[(subject_id, session_id)] = z_hat_valid_lst[i]
        z_hat_test_dict[(subject_id, session_id)]  = z_hat_test_lst[i]
        a_hat_train_dict[(subject_id, session_id)] = a_hat_train_lst[i]
        a_hat_valid_dict[(subject_id, session_id)] = a_hat_valid_lst[i]
        a_hat_test_dict[(subject_id, session_id)]  = a_hat_test_lst[i]
        behv_train_dict[(subject_id, session_id)] = behv_train_lst[i]
        behv_valid_dict[(subject_id, session_id)] = behv_valid_lst[i]
        behv_test_dict[(subject_id, session_id)]  = behv_test_lst[i]
        behv_hat_train_dict[(subject_id, session_id)] = behv_hat_train_lst[i]
        behv_hat_valid_dict[(subject_id, session_id)] = behv_hat_valid_lst[i]
        behv_hat_test_dict[(subject_id, session_id)]  = behv_hat_test_lst[i]
        trials_type_train_dict[(subject_id, session_id)] = trials_type_train_lst[i]
        trials_type_valid_dict[(subject_id, session_id)] = trials_type_valid_lst[i]
        trials_type_test_dict[(subject_id, session_id)]  = trials_type_test_lst[i]
        # update the dataframe for metrics recording
        df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_MSE")] = recon_score_train_lst[i]['MSE']
        df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_MAE")] = recon_score_train_lst[i]['MAE']
        df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_R2")]  = recon_score_train_lst[i]['R2']
        df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_Corr")]= recon_score_train_lst[i]['Corr']

        df_valid.loc[df_index, (f'{subject_id}_{session_id}', "recon_MSE")] = recon_score_valid_lst[i]['MSE']
        df_valid.loc[df_index, (f'{subject_id}_{session_id}', "recon_MAE")] = recon_score_valid_lst[i]['MAE']
        df_valid.loc[df_index, (f'{subject_id}_{session_id}', "recon_R2")]  = recon_score_valid_lst[i]['R2']
        df_valid.loc[df_index, (f'{subject_id}_{session_id}', "recon_Corr")]= recon_score_valid_lst[i]['Corr']

        df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_MSE")] = recon_score_test_lst[i]['MSE']
        df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_MAE")] = recon_score_test_lst[i]['MAE']
        df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_R2")]  = recon_score_test_lst[i]["R2"]
        df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_Corr")]= recon_score_test_lst[i]['Corr']

    a_hat_dict = {'train': a_hat_train_dict, 'valid': a_hat_valid_dict, 'test': a_hat_test_dict}
    behv_dict  = {'train': behv_train_dict, 'valid': behv_valid_dict, 'test': behv_test_dict}
    behv_hat_dict = {'train': behv_hat_train_dict, 'valid': behv_hat_valid_dict, 'test': behv_hat_test_dict}

    decoding_results_dict = train_behv_decoder(dataset_info,  
                                               a_hat_dict,
                                               behv_dict,
                                               behv_hat_dict,
                                               params)

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
            'df_valid': df_valid,
            'df_test': df_test,
            'df_index': df_index
        }, f)
        
    with open(f"{parent_folder}/decoding_results.pkl", 'wb') as f:
        pickle.dump(decoding_results_dict, f)
        
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

        
    with open(f"{parent_folder}/flag.pkl", 'wb') as f:
        pickle.dump({'flag': True}, f) 
