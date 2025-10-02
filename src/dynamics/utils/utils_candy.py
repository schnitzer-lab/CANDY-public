from yacs.config import CfgNode as CN
import torch
import torch.nn as nn
import numpy as np


#### Initialization of default and recommended (except dimensions and hidden layer lists, set them suitable for data to fit) config 
_config = CN() 

## Set device and seed
_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
_config.seed = int(torch.randint(low=0, high=100000, size=(1,)))

## Dump model related settings
_config.model = CN() 

# Hidden layer list where each element is the number of neurons for that hidden layer of DFINE encoder/decoder. Please use [20,20,20,20] for nonlinear manifold simulations.
_config.model.hidden_layer_list = [32,32,32] 
# Activation function used in encoder and decoder layers
# leakyrelu pretty good: 0.50 (100 epochs); 0.60 (200 epochs); 0.612 (300 epochs); 0.622 (500 epochs)
# relu: 0.467 (100 epochs)
# tanhshrink: 0.50 (100 epochs); 0.61 (500 epochs)
# sigmoid: 0.444 (100 epochs)
_config.model.activation = 'leakyrelu'
# Dimensionality of neural observations
_config.model.dim_y = 30
# Dimensionality of manifold latent factor, a choice higher than dim_y (above) may lead to overfitting
_config.model.dim_a = 16
# Dimensionality of dynamic latent factor, it's recommended to set it same as dim_a (above), please see Extended Data Fig. 8
_config.model.dim_x = 16
# Initialization scale of LDM state transition matrix
_config.model.init_A_scale = 1
# Initialization scale of LDM observation matrix
_config.model.init_C_scale = 1
# Initialization scale of LDM process noise covariance matrix
_config.model.init_W_scale = 0.5
# Initialization scale of LDM observation noise covariance matrix
_config.model.init_R_scale = 0.5
# Initialization scale of dynamic latent factor estimation error covariance matrix
_config.model.init_cov = 1
# Boolean for whether process noise covariance matrix W is learnable or not
_config.model.is_W_trainable = True
# Boolean for whether observation noise covariance matrix R is learnable or not
_config.model.is_R_trainable = True
# Initialization type of LDM parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.ldm_kernel_initializer = 'default'
# Initialization type of DFINE encoder and decoder parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.nn_kernel_initializer = 'xavier_normal'
# Boolean for whether to learn a behavior-supervised model or not. It must be set to True if supervised model will be trained.  
_config.model.supervise_behv = False
# Hidden layer list for the behavior mapper where each element is the number of neurons for that hidden layer of the mapper
_config.model.hidden_layer_list_mapper = []
# Activation function used in mapper layers
_config.model.activation_mapper = 'linear'
# List of dimensions of behavior data to be decoded by mapper, check for any dimensionality mismatch 
_config.model.which_behv_dims = [0, 1]
# Boolean for whether to decode behavior from a_smooth
_config.model.behv_from_smooth = False
# Main save directory for DFINE results, plots and checkpoints
_config.model.save_dir = 'D:/DATA/DFINE_results'
# Number of steps to save DFINE checkpoints
_config.model.save_steps = 10

# Subject discriminator
_config.model.use_subject_discriminator = False
_config.model.num_subjects = 1
_config.model.subject_discriminator_hidden_layer_list = [32]
_config.model.subject_discriminator_activation = 'leakyrelu'
_config.model.subject_discriminator_kernel_initializer = 'xavier_normal'
_config.model.subject_discriminator_scale = 1.0

# Contrastive learning
_config.model.contrastive = True
_config.model.contrastive_temp = 2.0
_config.model.contrastive_label_diff = 'l1'
_config.model.contrastive_feature_sim = 'l2'
_config.model.contrastive_scale = 10.0
_config.model.contrastive_num_batch = 64
_config.model.contrastive_time_scaler = 1

## Dump loss related settings
_config.loss = CN()

# L2 regularization loss scale (we recommend a grid-search for the best value, i.e., a grid of [1e-4, 5e-4, 1e-3, 2e-3]). Please use 0 for nonlinear manifold simulations as it leads to a better performance. 
_config.loss.scale_l2 = 2e-3
# List of number of steps ahead for which DFINE is optimized. For unsupervised and supervised versions, default values are [1,2,3,4] and [1,2], respectively. 
_config.loss.steps_ahead = [1,2,3,4]
# If _config.model.supervise_behv is True, scale for MSE of behavior reconstruction (We recommend a grid-search for the best value. It should be set to a large value).
_config.loss.scale_behv_recons = 20

## Dump training related settings
_config.train = CN()

# Batch size 
_config.train.batch_size = 32
# Number of epochs for which DFINE is trained
_config.train.num_epochs = 200
# Number of steps to check validation data performance
_config.train.valid_step = 1 
# Number of steps to save training/validation plots
_config.train.plot_save_steps = 50
# Number of steps to print training/validation logs
_config.train.print_log_steps = 10

## Dump loading settings
_config.load = CN()

# Number of checkpoint to load
_config.load.ckpt = -1
# Boolean for whether to resume training from the epoch where checkpoint is saved
_config.load.resume_train = False

## Dump learning rate related settings
_config.lr = CN()

# Learning rate scheduler type, options are explr (StepLR, purely exponential if explr.step_size == 1), cyclic (CyclicLR) or constantlr (constant learning rate, no scheduling)
_config.lr.scheduler = 'explr'
# Initial learning rate
_config.lr.init = 0.02

# Dump cyclic LR scheduler related settings, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html for details
_config.lr.cyclic = CN()
# Minimum learning rate for cyclic LR scheduler
_config.lr.cyclic.base_lr = 0.005
# Maximum learning rate for cyclic LR scheduler
_config.lr.cyclic.max_lr = 0.02
# Envelope scale for exponential cyclic LR scheduler mode
_config.lr.cyclic.gamma = 1
# Mode for cyclic LR scheduler
_config.lr.cyclic.mode = 'triangular'
# Number of iterations in the increasing half of the cycle
_config.lr.cyclic.step_size_up = 10

# Dump exponential LR scheduler related settings, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html for details
_config.lr.explr = CN()
# Multiplicative factor of learning rate decay
_config.lr.explr.gamma = 0.9
# Steps to decay the learning rate, becomes purely exponential if step is 1
_config.lr.explr.step_size = 15

## Dump optimizer related settings
_config.optim = CN()

# Epsilon for Adam optimizer
_config.optim.eps = 1e-8
# Gradient clipping norm    
_config.optim.grad_clip = 0.1


def get_default_config():
    '''
    Creates the default config

    Returns: 
    ------------
    - config: yacs.config.CfgNode, default DFINE config
    '''

    return _config.clone()


def carry_to_device(data, device, dtype=torch.float32):    
    '''
    Carries dict/list of torch Tensors/numpy arrays to desired device recursively

    Parameters: 
    ------------
    - data: torch.Tensor/np.ndarray/dict/list: Dictionary/list of torch Tensors/numpy arrays or torch Tensor/numpy array to be carried to desired device
    - device: str, Device name to carry the torch Tensors/numpy arrays to
    - dtype: torch.dtype, Data type for torch.Tensor to be returned, torch.float32 by default

    Returns: 
    ------------
    - data: torch.Tensor/dict/list: Dictionary/list of torch.Tensors or torch Tensor carried to desired device
    '''

    if torch.is_tensor(data):
        return data.to(device)
    
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype).to(device)

    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = carry_to_device(data[key], device)
        return data
    
    elif isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = carry_to_device(d, device)
        return data

    else:
        return data

    
def get_nrmse_error(y, y_hat, version_calculation='modified'):
    '''
    Computes normalized root-mean-squared error between two 3D tensors. Note that this operation is not symmetric. 

    Parameters: 
    ------------
    - y: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with true observations
    - y_hat: torch.Tensor/np.ndarray, shape: (num_seq, num_steps, dim_y), Tensor with reconstructed/estimated observations
    - version_calculation: str, Version to calculate the variance. If 'regular', variance of each sequence is computed separately, 
                                which may result in unstable nrmse value since some sequences may be constant or close to being constant and
                                results in ~0 variance, so high/unreasonable nrmse. To prevent that, variance is computed across flattened sequence in 
                                'modified' mode. 'modified' by default.

    Returns: 
    ------------
    - normalized_error: torch.Tensor, shape: (dim_y,), Normalized root-mean-squared error for each data dimension
    - normalized_error_mean: torch.Tensor, shape: (), Average normalized root-mean-squared error for each data dimension
    '''

    # Check if dimensions are consistent
    assert y.shape == y_hat.shape, f'dimensions of y {y.shape} and y_hat {y_hat.shape} do not match'
    assert len(y.shape) == 3, 'mismatch in x dimension: x should be in the format of (num_seq, num_steps, dim_x)'

    y = convert_to_tensor(y).detach().cpu()
    y_hat = convert_to_tensor(y_hat).detach().cpu()
    
    # carry time to first dimension
    y = torch.permute(y, (1,0,2)) # (num_steps, num_seq, dim_x)
    y_hat = torch.permute(y_hat, (1,0,2)) # (num_steps, num_seq, dim_x)

    recons_error = torch.mean(torch.square(y - y_hat), dim=0)

    # way 1 to calculate variance
    if version_calculation == 'regular':
        var_y = torch.mean(torch.square(y - torch.mean(y, dim=0)), dim=0)

    # way 2 to calculate variance (sometime data in a batch is flat, it's more robust to calculate variance globally)
    elif version_calculation == 'modified':
        y_resh = torch.reshape(y, (-1, y.shape[2]))
        var_y = torch.mean(torch.square(y_resh - torch.mean(y_resh, dim=0)), dim=0)
        var_y = torch.tile(var_y.unsqueeze(dim=0), (y.shape[1], 1))
    normalized_error = torch.mean((torch.sqrt(recons_error) / torch.sqrt(var_y)), dim=0) # mean across batches 
    normalized_error_mean = torch.mean(normalized_error)

    return normalized_error, normalized_error_mean


def convert_to_tensor(x, dtype=torch.float32):
    '''
    Converts numpy.ndarray to torch.Tensor

    Parameters: 
    ------------
    - x: np.ndarray, Numpy array to convert to torch.Tensor (if it's of type torch.Tensor already, it's returned without conversion)
    - dtype: torch.dtype, Data type for torch.Tensor to be returned, torch.float32 by default

    Returns: 
    ------------
    - y: torch.Tensor, Converted tensor
    '''

    if isinstance(x, torch.Tensor):
        y = x
    elif isinstance(x, np.ndarray):
        y = torch.tensor(x, dtype=dtype) # use np.ndarray as middle step so that function works with tf tensors as well
    else:
        assert False, 'Only Numpy array can be converted to tensor'
    return y

    
def compute_mse(y_flat, y_hat_flat, mask_flat=None):
    '''
    Returns average Mean Square Error (MSE) 

    Parameters:
    ------------
    - y_flat: torch.Tensor, shape: (num_samp, dim_y), True data to compute MSE of  
    - y_hat_flat: torch.Tensor, shape: (num_samp, dim_y), Predicted/Reconstructed data to compute MSE of  
    - mask_flat: torch.Tensor, shape: (num_samp, 1), Mask to compute MSE loss which shows whether 
                                                     observations at each timestep exists (1) or are missing (0)

    Returns:
    ------------    
    - mse: torch.Tensor, Average MSE 
    '''

    if mask_flat is None: 
        mask_flat = torch.ones(y_flat.shape[:-1], dtype=torch.float32)

    # Make sure mask is 2D
    if len(mask_flat.shape) != len(y_flat.shape):
        mask_flat = mask_flat.unsqueeze(dim=-1)

    # Compute the MSEs and mask the timesteps where observations are missing
    mse = (y_flat - y_hat_flat) ** 2
    mse = torch.mul(mask_flat, mse)
    
    # Return the mean of the mse (over available observations)
    if mask_flat.shape[-1] != y_flat.shape[-1]: # which means shape of mask_flat is of dimension 1
        num_el = mask_flat.sum() * y_flat.shape[-1]
    else:
        num_el = mask_flat.sum()

    mse = mse.sum() / num_el
    return mse

def identity_func(x):
    return x

def get_activation_function(activation_str):
    '''
    Returns activation function given the activation function's name

    Parameters:
    ----------------------
    - activation_str: str, Activation function's name

    Returns:
    ----------------------
    - activation_fn: torch.nn, Activation function  
    '''

    if activation_str.lower() == 'elu':
        return nn.ELU()
    elif activation_str.lower() == 'hardtanh':
        return nn.Hardtanh()
    elif activation_str.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_str.lower() == 'relu':
        return nn.ReLU()
    elif activation_str.lower() == 'rrelu':
        return nn.RReLU()
    elif activation_str.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str.lower() == 'mish':
        return nn.Mish()
    elif activation_str.lower() == 'tanh':
        return nn.Tanh()
    elif activation_str.lower() == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation_str.lower() == 'linear':
        return identity_func

def get_kernel_initializer_function(kernel_initializer_str):
    '''
    Returns kernel initialization function given the kernel initialization function's name

    Parameters:
    ----------------------
    - kernel_initializer_str: str, Kernel initialization function's name

    Returns:
    ----------------------
    - kernel_initializer_fn: torch.nn.init, Kernel initialization function  
    '''

    if kernel_initializer_str.lower() == 'uniform':
        return nn.init.uniform_
    elif kernel_initializer_str.lower() == 'normal':
        return nn.init.normal_
    elif kernel_initializer_str.lower() == 'xavier_uniform':
        return nn.init.xavier_uniform_
    elif kernel_initializer_str.lower() == 'xavier_normal':
        return nn.init.xavier_normal_
    elif kernel_initializer_str.lower() == 'kaiming_uniform':
        return nn.init.kaiming_uniform_
    elif kernel_initializer_str.lower() == 'kaiming_normal':
        return nn.init.kaiming_normal_
    elif kernel_initializer_str.lower() == 'orthogonal':
        return nn.init.orthogonal_
    elif kernel_initializer_str.lower() == 'default':
        return lambda x:x

def softplus(x):
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.relu(x)

def transform2bytrial(results, data_type='hat'):
    res_bytrial = results['batch_inference']

    mask   = res_bytrial['mask']
    if data_type == 'filter':
        X_recon = res_bytrial['y_filter']
        Z_pred  = res_bytrial['x_filter']
        A_pred  = res_bytrial['a_filter']
    elif data_type == 'smooth':
        X_recon = res_bytrial['y_smooth']
        Z_pred  = res_bytrial['x_smooth']
        A_pred  = res_bytrial['a_smooth']
    elif data_type == 'hat':
        X_recon = res_bytrial['y_hat']
        try:
            Z_pred  = res_bytrial['x_smooth']
        except KeyError:
            Z_pred  = res_bytrial['a_hat']
        A_pred  = res_bytrial['a_hat']
    else:
        raise Exception('incorrect type')
    beh = res_bytrial['behv']
    beh_pred = res_bytrial['behv_hat']
    trials_type = res_bytrial['ttype']
    X_orig = res_bytrial['y']
    obs_dim = X_orig.shape[-1]
    latent_dim = A_pred.shape[-1]
    beh_dim = beh_pred.shape[-1]
    # print('behv_dimension', beh_dim)
    # print(f'behv shape: {beh.shape}; behv_hat shape: {beh_pred.shape}')

    num_trials = X_orig.shape[0]
    data_orig  = []
    data_recon = []
    Z_bytrial  = []
    A_bytrial  = []
    beh_bytrial = []
    beh_hat_bytrial = []
    for i in range(num_trials):
        obs_dim = X_orig.shape[-1]
        latent_dim = A_pred.shape[-1]
        mask_i = mask[i].reshape(-1).detach().cpu().numpy().astype(np.bool_)
        x_orig_i  = X_orig[i].reshape(-1, obs_dim).detach().cpu().numpy()
        x_recon_i = X_recon[i].reshape(-1, obs_dim).detach().cpu().numpy()
        z_pred_i  = Z_pred[i].reshape(-1, latent_dim).detach().cpu().numpy()
        a_pred_i  = A_pred[i].reshape(-1, latent_dim).detach().cpu().numpy()
        beh_i     = beh[i].reshape(-1, beh_dim).detach().cpu().numpy()
        beh_pred_i = beh_pred[i].reshape(-1, beh_dim).detach().cpu().numpy()
        data_orig.append(x_orig_i[mask_i,:])
        data_recon.append(x_recon_i[mask_i, :])
        Z_bytrial.append(z_pred_i[mask_i, :])
        A_bytrial.append(a_pred_i[mask_i, :])
        beh_bytrial.append(beh_i[mask_i, :])
        beh_hat_bytrial.append(beh_pred_i[mask_i, :])
    return data_orig, data_recon, Z_bytrial, A_bytrial, beh_bytrial, beh_hat_bytrial, trials_type

def transform2bytrial_byagent(results_lst, data_type='hat'):
    num_agents = len(results_lst)
    data_orig_lst  = []
    data_recon_lst = []
    Z_bytrial_lst  = []
    A_bytrial_lst  = []
    beh_bytrial_lst = []
    beh_hat_bytrial_lst = []
    trials_type_lst = []

    for i in range(num_agents):
        results = results_lst[i]
        data_orig, data_recon, Z_bytrial, A_bytrial, beh_bytrial, beh_hat_bytrial, ttype = transform2bytrial(results, data_type)
        data_orig_lst.append(data_orig)
        data_recon_lst.append(data_recon)
        Z_bytrial_lst.append(Z_bytrial)
        A_bytrial_lst.append(A_bytrial)
        beh_bytrial_lst.append(beh_bytrial)
        beh_hat_bytrial_lst.append(beh_hat_bytrial)
        trials_type_lst.append(ttype)
    return data_orig_lst, data_recon_lst, Z_bytrial_lst, A_bytrial_lst, beh_bytrial_lst, beh_hat_bytrial_lst, trials_type_lst
