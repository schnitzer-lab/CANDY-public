import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def _get_trial_len_all(z_hat_lst):
    """
    INPUT
        [z_hat_lst] : a list of latent factors. Each element represents a trial.
    OUTPUT
        [trial_length] : a numpy array of trial length
    """
    trial_length = []
    for z_hat in z_hat_lst:
        trial_length.append(z_hat.shape[0])

    return trial_length

def get_vis_data(z_hat_train_dict, z_hat_test_dict, train_dataset, test_dataset, dataset_info):
    z_hat_train_lst_all = []
    z_hat_test_lst_all = []

    dataset_train_trials_type = []
    dataset_train_trials_indi = []
    dataset_train_behvs = []
    dataset_test_trials_type = []
    dataset_test_trials_indi = []
    dataset_test_behvs = []
    files_names = []
    for i, sess_name in enumerate(dataset_info):
        z_hat_train = z_hat_train_dict[sess_name]
        z_hat_test  = z_hat_test_dict[sess_name]
        behv_train  = train_dataset[sess_name]['behavior_data']
        behv_test   = test_dataset[sess_name]['behavior_data']
        ttype_train = train_dataset[sess_name]['trials_type']
        ttype_test  = test_dataset[sess_name]['trials_type']

        z_hat_train_lst_all += z_hat_train
        z_hat_test_lst_all  += z_hat_test

        dataset_train_trials_type += ttype_train 
        dataset_train_trials_indi += [i] * len(ttype_train)
        dataset_train_behvs += behv_train
        dataset_test_trials_type += ttype_test
        dataset_test_trials_indi += [i] * len(ttype_test)
        dataset_test_behvs += behv_test

        files_names += [sess_name]

    # Step 1: stack the z_hat_lst all
    z_hat_train_lst_all_stack = np.vstack(z_hat_train_lst_all)
    z_hat_test_lst_all_stack  = np.vstack(z_hat_test_lst_all)

    pca = PCA(n_components=3)
    z_hat_train_lst_all_pca = pca.fit_transform(z_hat_train_lst_all_stack)
    z_hat_test_lst_all_pca  = pca.transform(z_hat_test_lst_all_stack)

    dataset_train_trials_length = _get_trial_len_all(z_hat_train_lst_all)
    dataset_test_trials_length  = _get_trial_len_all(z_hat_test_lst_all)

    vis_dataset = {
                    'z_hat_train_lst_all_pca' : z_hat_train_lst_all_pca,
                    'z_hat_test_lst_all_pca'  : z_hat_test_lst_all_pca,
                    'dataset_train_trials_length' : dataset_train_trials_length,
                    'dataset_test_trials_length'  : dataset_test_trials_length,
                    'dataset_train_trials_indi' : dataset_train_trials_indi,
                    'dataset_test_trials_indi'  : dataset_test_trials_indi,
                    'dataset_train_trials_type' : dataset_train_trials_type,
                    'dataset_test_trials_type'  : dataset_test_trials_type,
                    'dataset_train_behvs' : dataset_train_behvs,
                    'dataset_test_behvs'  : dataset_test_behvs,
                    'files_names': files_names
                  }
    return vis_dataset

def plot_losses(model, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for i, t in enumerate(['train', 'valid']):
        for j, lt in enumerate(['total_losses', 'model_losses', 'behv_losses', 'contrastive_losses']):
            try:
                axes[j].plot(model.losses[t][lt], label=t)
                axes[j].legend()
            except:
                pass
            axes[j].set_title(lt)
            axes[0].set_ylabel(t)
    plt.savefig(f"{save_path}/losses.png")
    plt.close(fig)

def plot_bar_with_scatter(avg_dict, metrics=["decoder_R2", "unidecoder_R2", "supdecoder_R2"]):
    """
    Plots a bar plot with scatter points on top for selected metrics.

    Parameters:
        avg_dict (dict): Dictionary where keys are metric names and values are lists of values (one per dataset).
        metrics (list): List of metric names to plot.
    """
    num_metrics = len(metrics)
    
    # Compute means for bar heights
    means = [np.mean(avg_dict[metric]) for metric in metrics]

    # X positions for bars and scatter points
    x_positions = np.arange(num_metrics)

    plt.figure(figsize=(8, 5))
    plt.bar(x_positions, means, color='lightblue', alpha=0.6, label="Mean Value")

    # Scatter plot for individual dataset values
    for i, metric in enumerate(metrics):
        y_values = avg_dict[metric]
        x_jitter = np.random.uniform(-0.1, 0.1, size=len(y_values))  # Add slight jitter for better visibility
        plt.scatter(x_positions[i] + x_jitter, y_values, color='blue', alpha=0.8, label="Dataset Values" if i == 0 else "")

    # Formatting
    plt.xticks(x_positions, metrics)
    plt.ylabel("Metric Value")
    plt.title("Bar Plot with Scatter Overlay")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()

def _plot_latent(z_hat_pca, trials_length, trials_type, files_names, ax):
    start = 0
    end   = 0
    if 'left' in trials_type and 'right' in trials_type:
        color_trial_type_dict = {'left': 'skyblue', 'right': 'pink'}
    else: # monkey data
        unique_trial_type = list(set(trials_type))
        color_trial_type_dict = {trial_type: plt.get_cmap('tab10')(i) for i, trial_type in enumerate(unique_trial_type)}
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

def _plot_latent_bybehv(z_hat_pca, trials_length, behvs, ax):
    start = 0
    end   = 0
    cmap = plt.get_cmap('rainbow')
    vmin = np.min(np.concatenate(behvs))
    vmax = np.max(np.concatenate(behvs))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i, trial_length in enumerate(trials_length):
        behv = behvs[i]
        c = cmap(norm(behv))
        end = start + trial_length
        ax.scatter(z_hat_pca[start:end,0], z_hat_pca[start:end,1], c=c, alpha=0.5, s=50, zorder=10)
        start = end
    # Add the color bar corresponding to the behv values
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Speed')

def plot_latent_vis(vis_dataset, save_path, filter=False):
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
        _plot_latent_bybehv(z_hat_pca, trials_length, behv, axs[r, 2])
    if not filter:
        plt.savefig(f"{save_path}/latent_vis.png")
    else:
        plt.savefig(f"{save_path}/latent_vis_zfilter.png")

    plt.close(fig)