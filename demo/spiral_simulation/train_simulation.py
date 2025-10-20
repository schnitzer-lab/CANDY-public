import os 
import pickle
import argparse
import sys
sys.path.insert(0, '../') 
sys.path.insert(0, '../../') 
sys.path.insert(0, '../../../') 

import itertools
import wandb

import numpy as np
import pandas as pd 

from sklearn.decomposition import PCA
from experiments.spiral_simulation.cca import CCA

from src.decoding.utils import train_behv_decoder, update_csv_behv
from src.plotting.utils import plot_losses, get_vis_data

import torch

from matplotlib import pyplot as plt

from src.dynamics import *
from src.runner.utils import init_dataloaderN, init_csv
from src.runner.utils import get_latest_checkpoint_number

def _plot_latent(z_hat_pca, trials_length, trials_type, files_names, ax):
    start = 0
    end   = 0
    color_trial_type_dict = {'left': 'skyblue', 'right': 'pink'}
    cmap = plt.get_cmap('Spectral')
    n_colors = len(files_names)
    if n_colors > 1:
        color_trial_indi_dict = {i: cmap(i / (n_colors - 1)) for i in range(n_colors)}  # Generate color mapping
    else:
        color_trial_indi_dict = {0: 'purple'}

    for i, trial_length in enumerate(trials_length):
        end = start + trial_length
        if trials_type[i] in color_trial_type_dict.keys():
            c = color_trial_type_dict[trials_type[i]]
        else:
            c = color_trial_indi_dict[trials_type[i]]
        ax.plot(z_hat_pca[start:end,0], z_hat_pca[start:end,1], c=c)
        ax.scatter(z_hat_pca[start,0], z_hat_pca[start,1], c='red', alpha=0.5, s=50, zorder=10)
        ax.scatter(z_hat_pca[end-1,0], z_hat_pca[end-1,1], c='black', alpha=0.2, s=50, zorder=10)
        start = end

def plot_latent_vis(vis_dataset, save_path):
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    
    for r, traintest in enumerate(['train', 'test']):
        z_hat_pca = vis_dataset[f'z_hat_{traintest}_lst_all_pca']
        behv          = vis_dataset[f'dataset_{traintest}_behvs']
        trials_length = vis_dataset[f'dataset_{traintest}_trials_length']
        trials_type   = vis_dataset[f'dataset_{traintest}_trials_type']
        trials_ind    = vis_dataset[f'dataset_{traintest}_trials_indi']
        files_names   = vis_dataset['files_names']
        _plot_latent(z_hat_pca, trials_length, trials_type, files_names, axs[r, 0])
        _plot_latent(z_hat_pca, trials_length, trials_ind,  files_names, axs[r, 1])
    plt.savefig(f"{save_path}/latent_vis.png")
    plt.close(fig)

def plot_dynamics_2d(dynamics_matrix,
                        bias_vector,
                        mins=(-40, -40),
                        maxs=(40, 40),
                        npts=20,
                        axis=None,
                        z_arr=None,  # Added parameter for trajectories
                        cca=None,
                        **kwargs):
        """
        Visualize a high-dimensional dynamical system in a 2D PCA space via a quiver plot,
        and overlay trajectories from z_hat_dict if provided.
        
        Args:
            dynamics_matrix (ndarray, shape (n, n)): "A" matrix of the system.
            bias_vector     (ndarray, shape (n,)):   "b" vector of the system.
            mins, maxs (2-tuple): Min/max in PCA space for the quiver plot grid.
            npts (int):        Number of arrow grid points (per axis).
            axis (matplotlib axis): Optional axis to draw on. Creates new if None.
            pca (sklearn.decomposition.PCA): Fitted PCA that reduces to 2D.
            z_arr (ndarray, shape (n_subj, n_trials, T, 2)): Trajectories to overlay.
            kwargs:            Passed through to matplotlib.pyplot.quiver.
        Returns:
            q: The quiver object.
        """
        # Create a grid in the 2D PCA space
        x_grid, y_grid = np.meshgrid(
            np.linspace(mins[0], maxs[0], npts),
            np.linspace(mins[1], maxs[1], npts)
        )
        xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        
        # Convert each (x,y) in the 2D PCA space back to original dimension
        x_orig = xy_grid  # shape: (npts*npts, n)
        
        # --- DISCRETE-TIME version: dx = A*x + b - x  ---
        # shape: (npts*npts, n)
        if bias_vector is None:
            bias_vector = np.zeros(dynamics_matrix.shape[0])
        dx_orig = (dynamics_matrix @ x_orig.T).T + bias_vector[None] - x_orig

        if cca is not None:
            # ---------- helper matrices ----------
            # (n_feat × n_pc)  – columns are PC loading vectors
            P1_T = cca.pca_1.components_.T
            P2_T = cca.pca_2.components_.T

            # handy inverses
            M1_inv = np.linalg.inv(cca.M_1)
            M2_inv = np.linalg.inv(cca.M_2)

            # ---------- map grid points to model (ẑ) space ----------
            # xy_grid ≡ x_orig are points in aligned/original space
            z_align      = x_orig                                # (N × 2)
            z_align_pc   = cca.pca_1.transform(z_align)          # (N × n_pc)
            z_canon      = z_align_pc @ cca.M_1                  # canonical (N × d)
            z_hat_pc     = z_canon @ M2_inv                      # (N × n_pc)
            z_hat_orig   = cca.pca_2.inverse_transform(z_hat_pc) # (N × 2)

            # ---------- compute flow in model space ----------
            dx_hat_orig  = (dynamics_matrix @ z_hat_orig.T).T - z_hat_orig

            # ---------- map vectors back to aligned space ----------
            dx_hat_pc    = cca.pca_2.transform(dx_hat_orig)                    # (N × n_pc)
            dx_canon     = dx_hat_pc @ cca.M_2                   # (N × d)
            dx_align_pc  = dx_canon @ M1_inv                     # (N × n_pc)
            dx_pca       = cca.pca_1.inverse_transform(dx_align_pc)   # (N × 2)
        else:
            # no CCA → the grid is already in the native coordinates
            dx_pca = dx_orig
        
        # Reshape for quiver
        dx_pca_x = dx_pca[:, 0].reshape(x_grid.shape)
        dx_pca_y = dx_pca[:, 1].reshape(y_grid.shape)
        
        # Plot
        if axis is None:
            axis = plt.gca()
        
        # Plot the vector field
        q = axis.quiver(x_grid, y_grid, dx_pca_x, dx_pca_y, **kwargs)
        
        # Plot trajectories if z_hat_dict is provided
        if z_arr is not None:
            for i_subj in range(z_arr.shape[0]):
                for i_trial in range(z_arr.shape[1]):
                    z = z_arr[i_subj, i_trial]
                    # Plot the trajectory
                    axis.plot(z[:, 0], z[:, 1], alpha=0.5)
            # axis.legend(loc='upper right', fontsize='small')
        return q


def main(args):
    data_pickle = pickle.load(open(args.data_path, 'rb'))
    train_loader = data_pickle['train_loader']
    val_loader   = data_pickle['valid_loader']
    test_loader  = data_pickle['test_loader']
    train_dataset = data_pickle['train_dataset']
    valid_dataset = data_pickle['valid_dataset']
    test_dataset  = data_pickle['test_dataset']
    dataset_info = data_pickle['dataset_info']
    obs_dim_lst = [dataset_info[key]['obs_dim'] for key in dataset_info.keys()]

    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_type = 'CANDY'
    latent_dim = args.latent_dim
    ckpt_save_dir = args.ckpt_save_dir

    data_params = { 
                    'seed' : 0,
                    'val_size': 0.1,
                    'test_size': 0.3
                }

    model_params = {
        'seed': seed,
        'lr_init': 2e-3,
        'scale_l2': 5e-3,
        'supervise_behv': False,
        'scale_behv_recons': 1,
        'steps_ahead': [1, 2, 3, 4],
        'hidden_layer_lst': [3,2],
        'num_epochs': 500,
        'activation': 'leakyrelu',
        'which_behv_dims': [0, 1],
        'ckpt_save_dir': ckpt_save_dir,
        'device': 'cpu',
        'contrastive': True,
        'contrastive_temp': 0.1,
        'contrastive_label_diff': 'l1',
        'contrastive_feature_sim': 'l2',
        'contrastive_scale': 0.1,
        'contrastive_time_scaler': 0.0,
        'contrastive_num_batch': 64,
        'dim_x': latent_dim,
        'dim_a': latent_dim, 
        'batch_size': 32,
    }

    train_params = {
        'batch_size': 32,
        'do_full_inference': True,
        'data_type': 'filter',
        'shuffle_train': True,
        'wandb': False
    }

    decoder_params = {'normalize': False, 'tol': 1e-4, 'decoder_type': 'lasso'}
    decoder_train_params = {
                    'params_grid': {'alpha': np.arange(1e-5, 10, 0.1)},
                    'scoring' : 'neg_mean_squared_error',
                    'n_split' : 10,
                    'n_repeats' : 3,
                    'seed' : 1,
                    'train_together': False
                    }

    params = {
        'decoder_type': 'linear',
        'latent_dim' : latent_dim,
        'model_type': model_type,
        'data_params': data_params,
        'model_params' : model_params,
        'train_params' : train_params,
        'decoder_params': decoder_params,
        'decoder_train_params': decoder_train_params,
    }

    parent_folder = model_params['ckpt_save_dir']


    df_train = init_csv(dataset_info)
    df_test  = init_csv(dataset_info)
    df_index = 0

    model_name = f'{model_type}-latent_dim{latent_dim}'
    df_train.loc[df_index, 'model_name'] = model_name
    df_train.loc[df_index, 'data_seed']  = data_params['seed']
    df_test.loc[df_index, 'model_name'] = model_name
    df_test.loc[df_index, 'data_seed']  = data_params['seed']

    df_train.loc[df_index, 'model_seed'] = model_params['seed']
    df_test.loc[df_index, 'model_seed']  = model_params['seed']

    z_hat_train_dict = {}
    z_hat_test_dict  = {}
    a_hat_train_dict = {}
    a_hat_test_dict  = {}
    behv_train_dict = {}
    behv_test_dict  = {}
    behv_hat_train_dict = {}
    behv_hat_test_dict  = {}
    trials_type_train_dict = {}
    trials_type_test_dict  = {}

    os.makedirs(parent_folder, exist_ok=True)
    try:
        with open(f"{parent_folder}/results.pkl", 'rb') as f:
            res = pickle.load(f)
            print(f'[INFO] pickle file exist!')
        with open(f"{parent_folder}/flag.pkl", 'rb') as f:
            res_flag = pickle.load(f)
            print(f'[INFO] pickle file saved correctly!')
        
        if res_flag['flag']:
            print(f'[INFO] Loading dataset.')
            model = res['model']
            z_hat_train_dict = res['z_hat_train_dict']
            z_hat_test_dict  = res['z_hat_test_dict']
            a_hat_train_dict = res['a_hat_train_dict']
            a_hat_test_dict  = res['a_hat_test_dict']
            behv_train_dict  = res['behv_train_dict']
            behv_test_dict   = res['behv_test_dict']
            behv_hat_train_dict = res['behv_hat_train_dict']
            behv_hat_test_dict  = res['behv_hat_test_dict']
            trials_type_train_dict = res['trials_type_train_dict']
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
        # Initialize model
        model = CANDYDynamics(obs_dim_lst, latent_dim, **model_params)
        ckpt_dir = model_params['ckpt_save_dir'] + '/ckpts'
        ckpt_num = get_latest_checkpoint_number(ckpt_dir)
        if ckpt_num is not None:
            model.config['load']['ckpt'] = ckpt_num
            print(f'[INFO] checkpoint {ckpt_num} exist. Start from this epoch and continue training.')
            model.config.load.resume_train = True
            model._load_ckpt(model.candy_list, model.ldm, model.mapper, model.optimizer)

        model.fit(train_loader, val_loader, **train_params)
        model.config['load']['ckpt'] = 'best_loss'
        model._load_ckpt(model.candy_list, model.ldm, model.mapper, model.optimizer)
        data_orig_train_list, train_recon_list, z_hat_train_list, a_hat_train_list, behv_train_list, behv_hat_train_list, trials_type_train_list = model.transform(train_loader, data_type=train_params['data_type'], do_full_inference=train_params['do_full_inference'])
        data_orig_test_list, test_recon_list, z_hat_test_list,  a_hat_test_list, behv_test_list, behv_hat_test_list, trials_type_test_list = model.transform(test_loader, data_type=train_params['data_type'], do_full_inference=train_params['do_full_inference'])

        recon_score_train_lst = model.scoring(data_orig_train_list, train_recon_list)
        recon_score_test_lst  = model.scoring(data_orig_test_list, test_recon_list)

        for i, (subject_id, session_id) in enumerate(dataset_info.keys()):   
            # update the original dataset given that the Torch Dataset might be shuffled 
            train_dataset[(subject_id, session_id)]['behavior_data'] = behv_train_list[i]
            test_dataset[(subject_id, session_id)]['behavior_data']  = behv_test_list[i]
            train_dataset[(subject_id, session_id)]['neural_data'] = data_orig_train_list[i]
            test_dataset[(subject_id, session_id)]['neural_data']  = data_orig_test_list[i]
            train_dataset[(subject_id, session_id)]['trials_type'] = trials_type_train_list[i]
            test_dataset[(subject_id, session_id)]['trials_type']  = trials_type_test_list[i]
            # update the results buffer
            z_hat_train_dict[(subject_id, session_id)] = z_hat_train_list[i]
            z_hat_test_dict[(subject_id, session_id)]  = z_hat_test_list[i]
            a_hat_train_dict[(subject_id, session_id)] = a_hat_train_list[i]
            a_hat_test_dict[(subject_id, session_id)]  = a_hat_test_list[i]
            behv_train_dict[(subject_id, session_id)] = behv_train_list[i]
            behv_test_dict[(subject_id, session_id)]  = behv_test_list[i]
            behv_hat_train_dict[(subject_id, session_id)] = behv_hat_train_list[i]
            behv_hat_test_dict[(subject_id, session_id)]  = behv_hat_test_list[i]
            trials_type_train_dict[(subject_id, session_id)] = trials_type_train_list[i]
            trials_type_test_dict[(subject_id, session_id)]  = trials_type_test_list[i]
            # update the dataframe for metrics recording
            df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_MSE")] = recon_score_train_lst[i]['MSE']
            df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_MAE")] = recon_score_train_lst[i]['MAE']
            df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_R2")]  = recon_score_train_lst[i]['R2']
            df_train.loc[df_index, (f'{subject_id}_{session_id}', "recon_Corr")]= recon_score_train_lst[i]['Corr']

            df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_MSE")] = recon_score_test_lst[i]['MSE']
            df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_MAE")] = recon_score_test_lst[i]['MAE']
            df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_R2")]  = recon_score_test_lst[i]["R2"]
            df_test.loc[df_index, (f'{subject_id}_{session_id}', "recon_Corr")]= recon_score_test_lst[i]['Corr']

        with open(f"{model_params['ckpt_save_dir']}/results.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'dataset_info': dataset_info,
                'params': params,
                'z_hat_train_dict': z_hat_train_dict, 
                'z_hat_test_dict' : z_hat_test_dict,
                'a_hat_train_dict': a_hat_train_dict,
                'a_hat_test_dict' : a_hat_test_dict,
                'behv_train_dict': behv_train_dict, 
                'behv_test_dict' : behv_test_dict,
                'behv_hat_train_dict': behv_hat_train_dict,
                'behv_hat_test_dict' :  behv_hat_test_dict,
                'trials_type_train_dict': trials_type_train_dict,
                'trials_type_test_dict' : trials_type_test_dict,
                'train_recon': train_recon_list,
                'test_recon': test_recon_list,
                'train_dataset': train_dataset,
                'valid_dataset': valid_dataset,
                'test_dataset': test_dataset,
                'train_loader': train_loader,
                'valid_loader': val_loader,
                'test_loader': test_loader,
            }, f)

    # Step 2: Behavior decoding results
    a_hat_dict = {'train': z_hat_train_dict, 'valid': z_hat_test_dict, 'test': z_hat_test_dict}
    behv_dict  = {'train': behv_train_dict, 'valid': behv_test_dict, 'test': behv_test_dict}
    behv_hat_dict = {'train': behv_hat_train_dict, 'valid': behv_hat_test_dict, 'test': behv_hat_test_dict}
    decoding_results_dict = train_behv_decoder(dataset_info, a_hat_dict, behv_dict, behv_hat_dict, params)
    update_csv_behv(df_train, df_test, df_index, dataset_info, decoding_results_dict)

    # Step 3: Plot loss functions
    print('plotting loss', flush=True)
    plot_losses(model, model_params['ckpt_save_dir'])

    # Step 4: Plot latent plots
    print('plotting latent', flush=True)
    # Get Visualization dataset
    vis_dataset = get_vis_data(z_hat_train_dict, z_hat_test_dict, train_dataset, test_dataset, dataset_info)
    plot_latent_vis(vis_dataset, model_params['ckpt_save_dir'])
    
    print('saving csv files', flush=True)
    df_train_file_path = os.path.join(f"{model_params['ckpt_save_dir']}/", "df_train.csv")
    df_test_file_path  = os.path.join(f"{model_params['ckpt_save_dir']}/", "df_test.csv")
    df_train.to_csv(df_train_file_path, index=True)
    df_test.to_csv(df_test_file_path, index=True)
    
    ##### Plotting #####
    if args.latent_dim == 2:
        theta = np.pi / 15 
        r = 0.99
        A = r * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        model_A = model.ldm.A.detach().numpy()
        W_log_diag = model.ldm.W_log_diag.detach().cpu().numpy()
        avg_z_hat_oversess = []
        z_hat_test_list = []
        colors = ['royalblue', 'goldenrod', 'mediumseagreen']
        for i,z_hat_test_list_sess in enumerate(z_hat_test_dict.values()):
            z_hat_test_list_sess = np.array(z_hat_test_list_sess) 
            avg_inferred_latent_trajs_sess = z_hat_test_list_sess.mean(axis=0)
            z_hat_test_list.append(z_hat_test_list_sess)
            avg_z_hat_oversess.append(avg_inferred_latent_trajs_sess)
        avg_z_hat_oversess = np.array(avg_z_hat_oversess).mean(axis=0)

        z_test_list = []
        avg_z_oversess = []
        for sess_name in test_dataset.keys():
            z_test_list_sess = test_dataset[sess_name]['dynamics_data']
            z_test_list_sess = [z_test_list_sess_trial for z_test_list_sess_trial in z_test_list_sess]
            z_test_list.append(z_test_list_sess)
            avg_z_test_matrix_sess = np.array(z_test_list_sess).mean(axis=0)
            avg_z_oversess.append(avg_z_test_matrix_sess)
        avg_z_oversess = np.array(avg_z_oversess).mean(axis=0)

        z_hat_test_arr = np.array(z_hat_test_list)
        z_test_arr = np.array(z_test_list)
        print(f'A.shape: {A.shape}, model_A.shape: {model_A.shape}')
        print(f'[INFO] z_hat_test_arr.shape: {z_hat_test_arr.shape}, z_test_arr.shape: {z_test_arr.shape}')

        n_subj, n_trials, T, n_dim = z_hat_test_arr.shape
        z_hat_test_arr = z_hat_test_arr.reshape(n_subj, n_trials*T, n_dim)
        z_test_arr = z_test_arr.reshape(n_subj, n_trials*T, n_dim)
        cca = CCA(2)
        anchor_subj = 0
        _, _, S = cca.fit(z_test_arr[anchor_subj].T.copy(), z_hat_test_arr[anchor_subj].T.copy())
        zero_mean_z_hat_test_arr = z_hat_test_arr - np.mean(z_hat_test_arr, axis=1, keepdims=True)
        pca_z_hat_test_arr = np.array([cca.pca_2.transform(zero_mean_z_hat_test_arr[i].copy()) for i in range(n_subj)])
        align_z_hat_test_arr = np.array([cca.pca_1.inverse_transform(pca_z_hat_test_arr[i].copy() @ cca.M_2 @ np.linalg.inv(cca.M_1)) for i in range(n_subj)])

        align_z_hat_test_arr = align_z_hat_test_arr.reshape(n_subj, n_trials, T, 2)
        z_test_arr = z_test_arr.reshape(n_subj, n_trials, T, 2)
        z_hat_test_arr = z_hat_test_arr.reshape(n_subj, n_trials, T, 2)
        print(f'align_z_hat_test_arr.shape: {align_z_hat_test_arr.shape}')

        # select_trials = [0]
        mins = (-1.25, -1.25)
        maxs = (1.25, 1.25)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_dynamics_2d(A.copy(), bias_vector=None, mins=mins, maxs=maxs, npts=16, axis=axs[0], z_arr=z_test_arr.mean(1, keepdims=True), color='tab:gray', alpha=0.75, cca=None)
        axs[0].set_title('Original')
        plot_dynamics_2d(model_A.copy(), bias_vector=np.exp(W_log_diag), mins=mins, maxs=maxs, npts=16, axis=axs[1], z_arr=align_z_hat_test_arr.mean(1, keepdims=True), color='dimgray', alpha=0.75, cca=cca)
        axs[1].set_title('Aligned')
        fig.tight_layout()

        plt.savefig(f"./{model_params['ckpt_save_dir']}/spiral.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing script.')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='latent dimension.')
    parser.add_argument('--ckpt_save_dir', type=str, required=True,
                        help='ckpt save direction name. Use hyperparameters.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='the spiral dynamics data file path.')
    args = parser.parse_args()
    main(args)