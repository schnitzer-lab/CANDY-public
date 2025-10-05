import os
import argparse

import numpy as np
import pandas as pd
import torch

from src.runner.runner import runnerN
from src.runner.utils import get_model_name
from config.load_config import load_model_config, load_data_config, load_decoder_config

def main(args):
    data_config, data_params = load_data_config(args)
    data_path_lst = data_config['data_path_lst']
    brain_area    = data_config['brain_area']
    data_params['train_frac'] = args.train_frac

    if 'mouse_wheel' in args.data_config:
        data_type = 'mouse_wheel'
    elif 'perich_monkey' in args.data_config:
        data_type = 'perich_monkey'
    else:
        raise Exception(f'{args.data_config} need to include valid data type.')
    
    latent_dim  = args.latent_dim
    model_type  = args.model_type
    save_parent = args.save_path

    model_params, train_params = load_model_config(args)
    model_params['ckpt_save_dir'] = save_parent
    model_params['seed'] = args.model_seed

    decoder_params, decoder_train_params = load_decoder_config(args)
    if 'linear' not in args.decoder_config:
        decoder_params['seed'] = args.model_seed
        decoder_train_params['seed'] = args.model_seed
        decoder_params['hidden_layer_list_mapper'].append(int(latent_dim/2))
        model_params['activation_mapper'] = decoder_params['activation_mapper']
        model_params['hidden_layer_list_mapper'] = decoder_params['hidden_layer_list_mapper']
        decoder_type = 'nonlinear'
    else:
        decoder_type = 'linear'
    
    save_params = {
                    'grid_start'    : 0,
                    'grid_end'      : None,
                    'save_latent': not args.not_save_latent,
                    'save_losses': True,
                    'save_path': save_parent
                    }
    params = {
        'area': brain_area,
        'latent_dim' : latent_dim,
        'model_type': model_type,
        'decoder_type': decoder_type,
        'data_type': data_type,
        'data_params': data_params,
        'model_params' : model_params,
        'train_params' : train_params,
        'decoder_params': decoder_params,
        'decoder_train_params': decoder_train_params,
        'save_params': save_params
    }
    os.makedirs(save_params['save_path'], exist_ok=True)

    if args.behv_sup_off:
        params['model_params']['supervise_behv'] = False 
    if args.contrastive_off:
        params['model_params']['contrastive'] = False
    
    df_train, df_test = runnerN(data_path_lst, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing script.')
    parser.add_argument('--brain_area', type=str, default=None,
                       help='Brain areas [mouse wheel] or Task type [monkey] (default: None).')
    parser.add_argument('--data_folder', type=str, default='G:\My Drive\Research',
                        help='Data parent folder (default: G:\My Drive\Research).')
    parser.add_argument('--data_config', type=str, default='./config/data_config/mouse_wheel.yaml',
                        help='Dataset configuration path (default: mouse_wheel.yaml)')
    parser.add_argument('--latent_dim', type=int, default=8,
                       help='Latent subspace dimension (default: 8).')
    parser.add_argument('--model_seed', type=int, default=0,
                        help='Model seed (default: 0)')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type (default: CANDY).')
    parser.add_argument('--model_config',type=str, required=True,
                        help='Model type corresponding yaml configuration file path.')
    parser.add_argument('--decoder_config', type=str, default='./config/decoder_config/linear.yaml', 
                        help='behavior decoder yaml configuration file path.')
    parser.add_argument('--save_path', type=str, default='./test',
                       help='Saving parent folder path (default: ./test).')
    parser.add_argument('--behv_sup_off', action='store_true', 
                        help='turn off the behavior supervision.')
    parser.add_argument('--contrastive_off', action='store_true',
                        help='turn off contrastive learning.')
    parser.add_argument('--train_frac', type=float, default=1.0, 
                        help='Actual fraction of training data to use (default: 1.0). Useful for comparison of performance as a function of training data size while fixing the testing data.')
    parser.add_argument('--not_save_latent', action='store_true',
                        help='turn off latent save.')

    args = parser.parse_args()
    main(args)