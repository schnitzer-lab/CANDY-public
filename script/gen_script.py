import argparse
import os
import subprocess
import pickle
import sys
sys.path.append('..')

import numpy as np

from src.runner.utils import get_model_name
from config.load_config import load_model_config, load_data_config

def main(args):
    model_type = 'CANDY'
    model_config_name = args.model_config_name 
    data_config_name    = args.data_config_name
    args.data_config = f'../config/data_config/{args.data_config_name}.yaml'
    args.model_config = f'../config/model_config/{model_config_name}.yaml'
    decoder_config_name = args.decoder_config_name
    args.data_folder = '../../shared-dynamics/' # FIXME change data folder
    latent_dims = [2, 4, 8, 16, 32, 64] # FIXME
    model_seeds = [0, 1, 2, 3, 4] # FIXME

    mem_size = 180

    hyperparam_name = "cbatch64_ctemp0.2_cs0.1_cts0" # "cbatch64_ctemp0.5_cs0.5_cts0"
    save_path = f"../results/{hyperparam_name}_behvOFF" # FIXME change save path
    os.makedirs(save_path, exist_ok=True)
    N_cpu = 2

    for latent_dim in latent_dims:
        for model_seed in model_seeds:
            args.brain_area = 'str'
            area = args.brain_area

            args.model_seed = model_seed
            args.latent_dim = latent_dim
            
            data_config, data_params = load_data_config(args)
            model_params, train_params = load_model_config(args)
            model_params['seed'] = args.model_seed

            model_params['supervise_behv'] = False # FIXME remove 
            # model_params['contrastive'] = False # FIXME remove

            params = {
                        'area': args.brain_area,
                        'latent_dim': latent_dim,
                        'model_type': model_type,
                        'data_type': 'mouse_wheel',
                        'data_params': data_params,
                        'model_params': model_params,
                        'train_params': train_params,
                        'decoder_params': {'normalize': False, 'tol': 1e-4, 'decoder_type': 'lasso'},
                        'decoder_train_params': {
                            'params_grid': {'alpha': np.arange(1e-5, 10, 0.1)},
                            'scoring': 'neg_mean_squared_error',
                            'n_split': 10,
                            'n_repeats': 3,
                            'seed': 1,
                            'train_together': False
                        },
                        'save_params': {
                            'grid_start': 0,
                            'grid_end': None,
                            'save_latent': True,
                            'save_losses': True,
                            'save_path': save_path
                        }
                    }

            model_name = get_model_name(params)
            parent_folder = f"{save_path}/{area}/{model_name}/"

            try:
                with open(f"{parent_folder}/frac_1.0/flag.pkl", 'rb') as f:
                    res_flag = pickle.load(f)

                    if (res_flag['flag']) and (os.path.exists(f"{parent_folder}/frac_1.0/df_test.csv")):
                        print(f'[INFO] {args.model_type}_{area}_latent{latent_dim}_seed{model_seed} Finished!')
                        continue
            except Exception as e:
                pass

            script_dest = 'script.sh'
            job_name = f"{data_config_name}-{model_config_name}-latent_dim{latent_dim}-seed{model_seed}" # FIXME change the naming
            output   = f"{hyperparam_name}/{data_config_name}-{model_config_name}-latent_dim{latent_dim}-seed{model_seed}_job.%j.out"
            error    = f"{hyperparam_name}/{data_config_name}-{model_config_name}-latent_dim{latent_dim}-seed{model_seed}_job.%j.err"

            with open(script_dest, 'w') as file: # FIXME change the partition and time
                file.write(f"""#!/usr/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --time=1-00:00:00
#SBATCH --partition=owners
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={N_cpu}
#SBATCH --mem={mem_size}GB

export OMP_NUM_THREADS={N_cpu}
export MKL_NUM_THREADS={N_cpu}
export OPENBLAS_NUM_THREADS={N_cpu}
export NUMEXPR_NUM_THREADS={N_cpu}

source ~/.bashrc
conda activate py3.10
python3 ../train.py --behv_sup_off --save_path {save_path} --latent_dim {latent_dim} --model_seed {model_seed} --model_config ../config/model_config/{model_config_name}.yaml --data_folder {args.data_folder} --data_config ../config/data_config/{data_config_name}.yaml --decoder_config ../config/decoder_config/{decoder_config_name}.yaml""")
# FIXME change the command line ESPECIALLY results_282_ NAME
            cmd = f'sbatch {script_dest}'
            job = subprocess.Popen(cmd, shell=True)
            job.wait()
            # sys.exit(0) # FIXME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running the shared dynamics script")
    parser.add_argument('--data_config_name', type=str, default='mouse_wheel',
                        help='data configuration name (default: mouse_wheel)')
    parser.add_argument('--decoder_config_name', type=str, default='linear',
                        help='decoder configuration name (default: linear)')
    parser.add_argument('--model_config_name', type=str, default=None,
                        help='model configuration name, specify if different from model_type (default: None)')
    args = parser.parse_args()
    main(args)
