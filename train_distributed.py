#!/usr/bin/env python3
"""
Distributed training script for CANDY using PyTorch DistributedDataParallel.

This script can be used for both single-GPU and multi-GPU training:
- Single GPU: python train_distributed.py [args]
- Multi-GPU: torchrun --nproc_per_node=2 train_distributed.py [args]
"""

import os
import argparse
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd

from src.runner.runner import runnerN
from src.runner.utils import get_model_name
from config.load_config import load_model_config, load_data_config, load_decoder_config


def setup_distributed():
    """Setup distributed training environment variables."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Distributed training
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Choose backend based on CUDA availability
        if torch.cuda.is_available():
            # Initialize the process group with NCCL for GPU
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(local_rank)
        else:
            # Fallback to CPU with Gloo backend
            print("[WARNING] CUDA not available, falling back to CPU training with Gloo backend")
            dist.init_process_group(backend='gloo', init_method='env://')
            local_rank = -1  # CPU training
        
        return rank, world_size, local_rank, True
    else:
        # Single GPU/CPU training
        local_rank = 0 if torch.cuda.is_available() else -1
        return 0, 1, local_rank, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    # Only print from main process to avoid cluttered output
    if rank == 0:
        print(f"Starting training with {world_size} process(es)")
        if is_distributed:
            print(f"Using DistributedDataParallel with {world_size} GPUs")
        else:
            print("Using single GPU/CPU training")
    
    try:
        # Load configurations
        data_config, data_params = load_data_config(args)
        data_path_lst = data_config['data_path_lst']
        brain_area = data_config['brain_area']
        data_params['train_frac'] = args.train_frac

        if 'mouse_wheel' in args.data_config:
            data_type = 'mouse_wheel'
        elif 'perich_monkey' in args.data_config:
            data_type = 'perich_monkey'
        else:
            raise Exception(f'{args.data_config} need to include valid data type.')
        
        latent_dim = args.latent_dim
        model_type = args.model_type
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
            'grid_start': 0,
            'grid_end': None,
            'save_latent': not args.not_save_latent,
            'save_losses': True,
            'save_path': save_parent
        }
        
        params = {
            'area': brain_area,
            'latent_dim': latent_dim,
            'model_type': model_type,
            'decoder_type': decoder_type,
            'data_type': data_type,
            'data_params': data_params,
            'model_params': model_params,
            'train_params': train_params,
            'decoder_params': decoder_params,
            'decoder_train_params': decoder_train_params,
            'save_params': save_params
        }
        
        # Create save directory only on main process
        if rank == 0:
            os.makedirs(save_params['save_path'], exist_ok=True)
        
        # Synchronize all processes
        if is_distributed:
            dist.barrier()

        if args.behv_sup_off:
            params['model_params']['supervise_behv'] = False 
        if args.contrastive_off:
            params['model_params']['contrastive'] = False
        
        # Configure distributed training parameters
        if is_distributed:
            params['model_params']['multi_gpu'] = True
            params['model_params']['data_parallel_type'] = 'DistributedDataParallel'
            if rank == 0:
                print(f"[INFO] Distributed training enabled with {world_size} processes")
        else:
            # Single GPU or CPU training
            params['model_params']['multi_gpu'] = False
            params['model_params']['data_parallel_type'] = 'DataParallel'
            if rank == 0:
                print("[INFO] Single process training")
        
        # Adjust batch size for distributed training
        if is_distributed and world_size > 1:
            # Each process handles a portion of the batch
            # The effective global batch size will be batch_size * world_size
            if rank == 0:
                print(f"[INFO] Effective global batch size: {train_params.get('batch_size', 'default')} * {world_size}")
                
        # Run training
        df_train, df_test = runnerN(data_path_lst, params)
        
        # Only save results from main process
        if rank == 0:
            print("Training completed successfully")
            
    except Exception as e:
        if rank == 0:
            print(f"Training failed with error: {e}")
        raise e
    finally:
        # Clean up distributed training
        cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed training script for CANDY.')
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
    parser.add_argument('--model_config', type=str, required=True,
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
                        help='Actual fraction of training data to use (default: 1.0).')
    parser.add_argument('--not_save_latent', action='store_true',
                        help='turn off latent save.')

    args = parser.parse_args()
    main(args)