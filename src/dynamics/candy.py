import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.autograd import Function
from torchmetrics import Metric
import os
import datetime
import logging

from src.dynamics.base_dynamics import sharedDynamics
from src.dynamics.utils.utils_candy import get_default_config, carry_to_device, get_kernel_initializer_function, compute_mse, get_activation_function, transform2bytrial_byagent
from src.dynamics.utils.rnc_loss import RnCLoss

class CANDYDynamics(sharedDynamics):
    def __init__(self, obs_dim_list, latent_dim, **kwargs):
        """
        Initializes the CANDYDynamics class.

        Parameters:
        ------------
        - n_subjects: int, Number of subjects in the dataset
        - obs_dim_list: list, List of observation dimensions for each subject
        - latent_dim: int, Dimensionality of the latent space
        - **kwargs: Additional keyword arguments
            - hidden_size_enc: int, Hidden layer dimension of the encoder
            - hidden_size_dec: int, Hidden layer dimension of the decoder
            - dropout: float, Dropout ratio (between 0 and 1)
            - output_size: int, Size of the output dimension (optional, default = obs_dim)
            - clip: float, Clip value for gradient (optional, default = 5.0)
            - lag_con: int, Lag to the controller input (optional, default = 1)
            - device: str, Device to run the model on ('cpu' or 'cuda0') (optional, default = 'cpu')
        """
        super().__init__(obs_dim_list, latent_dim, **kwargs)
        self.n_subjects = len(obs_dim_list)

        self.config = get_default_config()
        self.config.seed = kwargs.pop('seed', 0)
        self.config.model.dim_a = latent_dim
        self.config.model.dim_x = latent_dim
        self.config.model.which_behv_dims =  kwargs.pop('which_behv_dims', self.config.model.which_behv_dims)
        self.config.model.hidden_layer_list = kwargs.pop('hidden_layer_list', self.config.model.hidden_layer_list)
        self.config.model.supervise_behv = kwargs.pop('supervise_behv', self.config.model.supervise_behv)
        self.config.model.contrastive = kwargs.pop('contrastive', self.config.model.contrastive)
        self.config.model.contrastive_temp = kwargs.pop('contrastive_temp', self.config.model.contrastive_temp)
        self.config.model.contrastive_label_diff = kwargs.pop('contrastive_label_diff', self.config.model.contrastive_label_diff)
        self.config.model.contrastive_feature_sim = kwargs.pop('contrastive_feature_sim', self.config.model.contrastive_feature_sim)
        self.config.model.contrastive_scale = kwargs.pop('contrastive_scale', self.config.model.contrastive_scale)
        self.config.model.contrastive_time_scaler = kwargs.pop('contrastive_time_scaler', self.config.model.contrastive_time_scaler)
        self.config.model.contrastive_num_batch = kwargs.pop('contrastive_num_batch', self.config.model.contrastive_num_batch)

        self.config.model.activation_mapper  = kwargs.pop('activation_mapper', self.config.model.activation_mapper)
        self.config.hidden_layer_list_mapper = kwargs.pop('hidden_layer_list_mapper', self.config.model.hidden_layer_list_mapper)

        self.config.lr.init = kwargs.pop('lr_init', self.config.lr.init)
        self.config.loss.scale_l2 = kwargs.pop('scale_l2', self.config.loss.scale_l2)
        self.config.loss.steps_ahead = kwargs.pop('steps_ahead', self.config.loss.steps_ahead)
        self.config.loss.scale_behv_recons = kwargs.pop('scale_behv_recons', self.config.loss.scale_behv_recons)
        self.config.model.save_dir = kwargs['ckpt_save_dir']
        self.config.device = kwargs.pop('device', 'cpu')
        self.config.train.num_epochs = kwargs.pop('num_epochs', self.config.train.num_epochs)
        self.device = 'cpu' if self.config.device == 'cpu' or not torch.cuda.is_available() else 'cuda:0' 

        self._set_dims_and_scales()

        # Checkpoint and plot save directories, create directories if they don't exist
        self.ckpt_save_dir = os.path.join(self.config.model.save_dir, 'ckpts'); os.makedirs(self.ckpt_save_dir, exist_ok=True)

        self.start_epoch = kwargs.pop('start_epoch', 0)

        # Define shared LDM
        A, C, W_log_diag, R_log_diag, mu_0, Lambda_0 = self._init_ldm_parameters(self.dim_x, self.dim_a)
        self.ldm = LDM(dim_x=self.dim_x, dim_a=self.dim_a, 
                       A=A, C=C, 
                       W_log_diag=W_log_diag, R_log_diag=R_log_diag,
                       mu_0=mu_0, Lambda_0=Lambda_0,
                       is_W_trainable=self.config.model.is_W_trainable,
                       is_R_trainable=self.config.model.is_R_trainable)
        self.ldm.to(self.device)

        self.candy_list = list()
        for i in range(self.n_subjects):
            self.config.model.dim_y = obs_dim_list[i]
            candy = CANDY(self.config)
            candy.to(self.device) # carry the model to the desired device
            self.candy_list.append(candy)

        # Define shared behavioral decoder
        if self.config.model.supervise_behv:
            self.mapper = self._get_MLP(input_dim=self.dim_x, 
                                        output_dim=self.dim_behv, 
                                        layer_list=self.config.model.hidden_layer_list_mapper, 
                                        activation_str=self.config.model.activation_mapper)
            self.mapper.to(self.device)
        else:
            self.mapper = None
        self.dim_behv = len(self.config.model.which_behv_dims)
        self.scale_behv_recons = self.config.loss.scale_behv_recons

        # Define contrastive learning loss
        if self.config.model.contrastive:
            self.contrastive_loss = RnCLoss(temperature=self.config.model.contrastive_temp, 
                                            label_diff=self.config.model.contrastive_label_diff, 
                                            feature_sim=self.config.model.contrastive_feature_sim)

        self.best_val_loss = torch.inf
        self.best_val_behv_loss = torch.inf
        
        self.batch_size = kwargs['batch_size']

        candy_params = sum([list(candy.parameters()) for candy in self.candy_list], list())
        ldm_params = list(self.ldm.parameters())
        mapper_params = list(self.mapper.parameters()) if self.mapper is not None else []
        params = candy_params + ldm_params + mapper_params
        self.optimizer = self._get_optimizer(params)
        self.lr_scheduler = self._get_lr_scheduler()

        # Get the metrics 
        self.metric_names, self.metrics = self._get_metrics()
        self.losses = {
            'train': {
                'behv_losses' : [],
                'contrastive_losses': [],
                'model_losses': [],
                'reg_losses'  : [],
                'total_losses': []
            },
            'valid': {
                'behv_losses' : [],
                'contrastive_losses': [],
                'model_losses': [],
                'reg_losses'  : [],
                'total_losses': []
            }
        }

        # Initialize logger 
        self.logger = self._get_logger(prefix='candy')


    def _set_dims_and_scales(self):
        '''
        Sets the observation (y), manifold latent factor (a) and dynamic latent factor (x)
        (and behavior data dimension if supervised model is to be trained) dimensions,
        as well as behavior reconstruction loss and regularization loss scales from config. 
        '''

        # Set the dimensions
        self.dim_y = self.config.model.dim_y
        self.dim_a = self.config.model.dim_a
        self.dim_x = self.config.model.dim_x

        if self.config.model.supervise_behv:
            self.dim_behv = len(self.config.model.which_behv_dims)
        
        # Set the loss scales for behavior component and for the regularization
        if self.config.model.supervise_behv:
            self.scale_behv_recons = self.config.loss.scale_behv_recons
        self.scale_l2 = self.config.loss.scale_l2


    def _init_ldm_parameters(self, dim_x, dim_a): 
        '''
        Initializes the LDM Module parameters

        Returns:
        ------------
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of dynamics noise covariance matrix (W, therefore it is diagonal and PSD)
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD)
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor prediction initial condition (x_{0|-1}) for Kalman filtering
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman filtering

        * We learn the log-diagonal of matrix W and R to satisfy the PSD constraint for cov matrices. Diagnoal W and R are used for the stability of learning 
        similar to prior latent LDM works, see (Kao et al., Nature Communications, 2015) & (Abbaspourazad et al., IEEE TNSRE, 2019) for further info
        '''

        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.ldm_kernel_initializer)
        A = kernel_initializer_fn(self.config.model.init_A_scale * torch.eye(dim_x, dtype=torch.float32)) 
        C = kernel_initializer_fn(self.config.model.init_C_scale * torch.randn(dim_a, dim_x, dtype=torch.float32)) 

        W_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_W_scale * torch.eye(dim_x, dtype=torch.float32))))
        R_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_R_scale * torch.eye(dim_a, dtype=torch.float32))))
        
        mu_0 = kernel_initializer_fn(torch.zeros(dim_x, dtype=torch.float32))
        Lambda_0 = kernel_initializer_fn(self.config.model.init_cov * torch.eye(dim_x, dtype=torch.float32))

        return A, C, W_log_diag, R_log_diag, mu_0, Lambda_0


    def _get_MLP(self, input_dim, output_dim, layer_list, activation_str='tanh'):
        '''
        Creates an MLP object

        Parameters:
        ------------
        - input_dim: int, Dimensionality of the input to the MLP network
        - output_dim: int, Dimensionality of the output of the MLP network
        - layer_list: list, List of number of neurons in each hidden layer
        - activation_str: str, Activation function's name, 'tanh' by default

        Returns: 
        ------------
        - mlp_network: an instance of MLP class with desired architecture
        '''

        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.nn_kernel_initializer)
    
        mlp_network = MLP(input_dim=input_dim,
                          output_dim=output_dim,
                          layer_list=layer_list,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )
        return mlp_network


    def _get_metrics(self):
        '''
        Creates the metric names and nested metrics dictionary. 

        Returns: 
        ------------
        - metric_names: list, Metric names to log in Tensorboard, which are the keys of train/valid defined below
        - metrics_dictionary: dict, nested metrics dictionary. Keys (and metric_names) are (e.g. for config.loss.steps_ahead = [1,2]): 
            - train: 
                - steps_{k}_mse: metrics.Mean, Training {k}-step ahead predicted MSE         
                - model_loss: metrics.Mean, Training negative sum of {k}-step ahead predicted MSEs (e.g. steps_1_mse + steps_2_mse)
                - reg_loss: metrics.Mean, L2 regularization loss for CANDY encoder and decoder weights
                - behv_mse: metrics.Mean, Exists if config.model.supervise_behv is True, Training behavior MSE
                - behv_loss: metrics.Mean, Exists if config.model.supervise_behv is True, Training behavior reconstruction loss
                - total_loss: metrics.Mean, Sum of training model_loss, reg_loss and behv_loss (if config.model.supervise_behv is True)
            - valid: 
                - steps_{k}_mse: metrics.Mean, Validation {k}-step ahead predicted MSE
                - model_loss: metrics.Mean, Validation negative sum of {k}-step ahead predicted MSEs (e.g. steps_1_mse + steps_2_mse)
                - reg_loss: metrics.Mean, L2 regularization loss for CANDY encoder and decoder weights
                - behv_mse: metrics.Mean, Exists if config.model.supervise_behv is True, Validation behavior MSE
                - behv_loss: metrics.Mean, Exists if config.model.supervise_behv is True, Validation behavior reconstruction loss
                - total_loss: metrics.Mean, Sum of validation model_loss, reg_loss and behv_loss (if config.model.supervise_behv is True)
        '''

        metric_names = []
        for k in self.config.loss.steps_ahead:
            metric_names.append(f'steps_{k}_mse')
            
        if self.config.model.supervise_behv:
            metric_names.append('behv_mse')
            metric_names.append('behv_loss')
        if self.config.model.contrastive:
            metric_names.append('contrastive_loss')
        metric_names.append('model_loss')
        metric_names.append('reg_loss')
        metric_names.append('total_loss')
        
        metrics = {}
        metrics['train'] = {}
        metrics['valid'] = {}

        for key in metric_names:
            metrics['train'][key] = Mean()
            metrics['valid'][key] = Mean()
        
        return metric_names, metrics


    def _get_optimizer(self, params):
        '''
        Creates the Adam optimizer with initial learning rate and epsilon specified inside config by config.lr.init and config.optim.eps, respectively
        
        Parameters:
        ------------
        - params: Parameters to be optimized by the optimizer

        Returns:
        ------------
        - optimizer: Adam optimizer with desired learning rate, epsilon to optimize parameters specified by params
        '''
        
        optimizer = torch.optim.Adam(params=params, 
                                     lr=self.config.lr.init, 
                                     eps=self.config.optim.eps)
        return optimizer
    

    def _get_lr_scheduler(self):
        '''
        Creates the learning rate scheduler based on scheduler type specified in config.lr.scheduler. Options are constrained by StepLR (explr), CyclicLR (cyclic) and LambdaLR (which is used as constantlr).
        '''

        if self.config.lr.scheduler.lower() == 'explr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.config.lr.explr.gamma, step_size=self.config.lr.explr.step_size)
        elif self.config.lr.scheduler.lower() == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.lr.cyclic.base_lr, max_lr=self.config.lr.cyclic.max_lr, mode=self.config.lr.cyclic.mode, gamma=self.config.lr.cyclic.gamma, step_size_up=self.config.lr.cyclic.step_size_up, cycle_momentum=False)
        elif self.config.lr.scheduler.lower() == 'constantlr':
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1)
        else:
            assert False, 'Only these learning rate schedulers are available: StepLR (explr), CyclicLR (cyclic) and LambdaLR (which is constantlr)!'
        return scheduler


    def _load_ckpt(self, candy_list, ldm, mapper, optimizer, lr_scheduler=None):
        '''
        Loads the checkpoint specified in the config by config.load.ckpt.

        Parameters:
        ------------
        - candy_list: list, List of initialized CANDY models to load the parameters to
        - ldm: torch.nn.Module, Initialized LDM model to load the parameters to
        - mapper: torch.nn.Module, Initialized Mapper model to load the parameters to (None by default)
        - optimizer: torch.optim.Adam, Initialized Adam optimizer to load optimizer parameters to (loading is skipped if config.load.resume_train is False)
        - lr_scheduler: torch.optim.lr_scheduler, Initialized learning rate scheduler to load learning rate scheduler parameters to, None by default (loading is skipped if config.load.resume_train is False)

        Returns:
        ------------
        - candy_list: list, Loaded CANDY models
        - ldm: torch.nn.Module, Loaded LDM model
        - mapper: torch.nn.Module, Loaded Mapper model (if provided, otherwise, None)
        - optimizer: torch.optim.Adam, Loaded Adam optimizer (if config.load.resume_train is True, otherwise, initialized optimizer is returned)
        - lr_scheduler: torch.optim.lr_scheduler, Loaded learning rate scheduler (if config.load.resume_train is True, otherwise, initialized learning rate scheduler is returned)
        '''

        self.logger.warning('Optimizer and LR scheduler can be loaded only in resume_train mode, else they are re-initialized')
        load_path = os.path.join(self.config.model.save_dir, 'ckpts', f'{self.config.load.ckpt}_ckpt.pth')
        self.logger.info(f'Loading model from: {load_path}...')

        # Load the checkpoint 
        try: 
            ckpt = torch.load(load_path)
        except:
            self.logger.error('Ckpt path does not exist!')
            assert False, ''

        # If config.load.resume_train is True, load optimizer and learning rate scheduler
        if self.config.load.resume_train:
            self.start_epoch = ckpt['epoch'] + 1 if isinstance(ckpt['epoch'], int) else 1
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                self.logger.error('Optimizer cannot be loaded!, check if optimizer type is consistent!')
                assert False, ''
            
            if lr_scheduler is not None:
                try:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                except:
                    self.logger.error('LR scheduler cannot be loaded, check if scheduler type is consistent!')
                    assert False, ''

        try:
            for i in range(len(candy_list)):
                candy_list[i].load_state_dict(ckpt['candy_state_dict_list'][i])
            ldm.load_state_dict(ckpt['ldm_state_dict'])
            if mapper is not None:
                mapper.load_state_dict(ckpt['mapper_state_dict'])
        except:
            self.logger.error('Given architecture in config does not match the architecture of given checkpoint!')
            assert False, ''

        self.logger.info(f'Checkpoint succesfully loaded from {load_path}!')
        return candy_list, ldm, mapper, optimizer, lr_scheduler


    def _save_ckpt(self, epoch, candy_list, ldm, mapper, optimizer, lr_scheduler=None):
        '''
        Saves the checkpoint under ckpt_save_dir (see __init__) with filename {epoch}_ckpt.pth

        Parameters:
        ------------
        - epoch: int, Epoch number for which the checkpoint is to be saved for
        - candy_list: list, List of initialized CANDY models to be saved
        - ldm: torch.nn.Module, Initialized LDM model to be saved
        - mapper: torch.nn.Module, Initialized Mapper model to be saved (None by default)
        - optimizer: torch.optim.Adam, Initialized Adam optimizer to be saved (loading is skipped if config.load.resume_train is False)
        - lr_scheduler: torch.optim.lr_scheduler, Initialized learning rate scheduler to be saved, None by default (loading is skipped if config.load.resume_train is False)
        '''

        save_path = os.path.join(self.ckpt_save_dir, f'{epoch}_ckpt.pth')
        if lr_scheduler is not None:
            torch.save({
                        'candy_state_dict_list': [candy.state_dict() for candy in candy_list],
                        'ldm_state_dict': ldm.state_dict(),
                        'mapper_state_dict': mapper.state_dict() if mapper is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch
                        }, save_path)
        else:
            torch.save({
                        'candy_state_dict_list': [candy.state_dict() for candy in candy_list],
                        'ldm_state_dict': ldm.state_dict(),
                        'mapper_state_dict': mapper.state_dict() if mapper is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                        }, save_path)


    def _update_metrics(self, loss_dict_list, batch_size, train_valid='train', verbose=True):
        '''
        Updates the metrics based on the provided loss values.

        Parameters:
        ------------
        - loss_dict_list: list of dicts, List of dictionaries containing loss values.
        - batch_size: int, Number of trials for which the metrics are computed.
        - train_valid: str, Indicates whether to update train or validation metrics. Default is 'train'.
        - verbose: bool, Whether to print a warning if a key in metric_names doesn't exist in loss_dict.
        '''
        avg_loss_dict = {}
        for key in self.metric_names:
            loss_sum = 0
            count = 0
            for loss_dict in loss_dict_list:
                if key in loss_dict:
                    loss_sum += loss_dict[key]
                    count += 1
            if count > 0:
                avg_loss = loss_sum / count
                self.metrics[train_valid][key].update(avg_loss, batch_size)
                avg_loss_dict[key] = avg_loss
            else:
                if verbose:
                    self.logger.warning(f'{key} does not exist in any loss_dict, metric cannot be updated!')
                else:
                    pass
        # FIXME
        if self.config.model.supervise_behv:
            self.losses[train_valid]['behv_losses'].append(avg_loss_dict['behv_loss'].detach().cpu().numpy())
        if self.config.model.contrastive:
            self.losses[train_valid]['contrastive_losses'].append(avg_loss_dict['contrastive_loss'].detach().cpu().numpy())
        self.losses[train_valid]['model_losses'].append(avg_loss_dict['model_loss'].detach().cpu().numpy())
        self.losses[train_valid]['reg_losses'].append(avg_loss_dict['reg_loss'].detach().cpu().numpy())
        self.losses[train_valid]['total_losses'].append(avg_loss_dict['total_loss'].detach().cpu().numpy())

        return avg_loss_dict

    def _reset_metrics(self, train_valid='train'):
        '''
        Resets the metrics 

        Parameters:
        ------------
        - train_valid: str, Which metrics to reset, 'train' by default
        '''

        for _, metric in self.metrics[train_valid].items():
            metric.reset()


    def _get_logger(self, prefix='candy'):
        '''
        Creates the logger which is saved as .log file under config.model.save_dir

        Parameters:
        ------------
        - prefix: str, Prefix which is used as logger's name and .log file's name, 'candy' by default 

        Returns:
        ------------
        - logger: logging.Logger, Logger object to write logs into .log file
        '''

        os.makedirs(self.config.model.save_dir, exist_ok=True)
        date_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        log_path = os.path.join(self.config.model.save_dir, f'{prefix}_{date_time}.log')

        # from: https://stackoverflow.com/a/56689445/16228104
        logger = logging.getLogger(f'{prefix.upper()} Logger')
        logger.setLevel(logging.DEBUG)

        # Remove old handlers from logger (since logger is static object) so that in several calls, it doesn't overwrite to previous log files
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger


    def _get_log_str(self, epoch, train_valid='train'):
        '''
        Creates the logging/printing string of training/validation statistics at each epoch

        Parameters: 
        ------------
        - epoch: int, Number of epoch to log the statistics for 
        - train_valid: str, Training or validation prefix to log the statistics, 'train' by default

        Returns: 
        ------------
        - log_str: str, Logging string 
        '''

        log_str = f'Epoch {epoch}, {train_valid.upper()}\n'

        # Logging k-step ahead predicted MSEs
        for k in self.config.loss.steps_ahead:
            if k == 1:
                log_str += f"{k}_step_mse: {self.metrics[train_valid][f'steps_{k}_mse'].compute():.5f}\n"
            else:
                log_str += f"{k}_steps_mse: {self.metrics[train_valid][f'steps_{k}_mse'].compute():.5f}\n"

        # Logging L2 regularization loss and L2 scale 
        log_str += f"reg_loss: {self.metrics[train_valid]['reg_loss'].compute():.5f}, scale_l2: {self.candy_list[0].scale_l2:.5f}\n"

        # If model is behavior-supervised, log behavior reconstruction loss
        if self.config.model.supervise_behv:
            log_str += f"behv_loss: {self.metrics[train_valid]['behv_loss'].compute():.5f}, scale_behv_recons: {self.candy_list[0].scale_behv_recons:.5f}\n"

        if self.config.model.contrastive:
            log_str += f"contrastive_loss: {self.metrics[train_valid]['contrastive_loss'].compute():.5f}, contrastive_scale: {self.config.model.contrastive_scale:.5f}\n"

        # Finally, log model_loss and total_loss to optimize
        log_str += f"model_loss: {self.metrics[train_valid]['model_loss'].compute():.5f}, total_loss: {self.metrics[train_valid]['total_loss'].compute():.5f}\n"
        return log_str


    def train_epoch(self, epoch, train_loader_list, verbose=True):
        '''
        Performs training for a single epoch over batches, logging to Tensorboard and generating plots.

        Parameters: 
        ------------
        - epoch: int, The epoch number for which the training iteration is performed.
        - train_loader_list: list of torch.utils.data.DataLoader, The list of training dataloaders.
        - verbose: bool, Whether to print the training step information for the last batch. Default is True.
        '''

        # Take the model into training mode
        for i_sub in range(self.n_subjects):
            self.candy_list[i_sub].train()

        # Reset the metrics at the beginning of each epoch
        self._reset_metrics(train_valid='train')

        # Keep track of update step for logging the gradient norms
        min_batch_num = min([len(train_loader) for train_loader in train_loader_list])
        step = (epoch - 1) * min_batch_num + 1
        total_steps = self.config.train.num_epochs * min_batch_num

        # Store the losses to calculate the average loss in the end
        running_losses = {
            'model_loss': 0,
            'behv_loss': 0,
            'behv_mse': 0,
            'reg_loss': 0,
            'total_loss': 0,
            'contrastive_loss': 0,
        }
        for k in self.config.loss.steps_ahead:
            running_losses[f'steps_{k}_mse'] = 0

        # Start iterating over batches
        count = 0
        for batch_idx, batch_list in enumerate(zip(*train_loader_list)):
            count += 1
            total_loss = 0.
            total_loss_dict_list = list()
            if self.config.model.contrastive:
                behv_batch_list = list()
                t_list = list()
            a_hat_list = list()
            subject_list = list()
            for i_sub, batch in enumerate(batch_list):

                # Carry data to device
                y_batch, behv_batch, trial_len, trial_type = batch
                mask_batch = torch.ones(y_batch.shape[:-1]+(1,), dtype=torch.float32)
                for i in range(trial_len.shape[0]):
                    mask_batch[i, trial_len[i]:] = 0
                batch = (y_batch, behv_batch, mask_batch)
                batch = carry_to_device(data=list(batch), device=self.device)
                y_batch, behv_batch, mask_batch = batch

                # Perform forward pass and compute loss
                model_vars = self.candy_list[i_sub](y=y_batch, ldm=self.ldm, mapper=self.mapper, mask=mask_batch, normalize=epoch<1)

                loss, loss_dict = self.candy_list[i_sub].compute_loss(y=y_batch, 
                                                        model_vars=model_vars, 
                                                        mask=mask_batch, 
                                                        behv=behv_batch)
                total_loss += loss

                total_loss_dict_list.append(loss_dict)

                a_hat = model_vars['a_hat']
                a_hat = [a[:_len] for a, _len in zip(a_hat, trial_len)]

                # only use the latent behavior dimensions for contrastive learning
                a_hat = torch.cat(a_hat, dim=0)
                subject_list.extend([i_sub]*a_hat.shape[0])
                a_hat_list.append(a_hat)

                if self.config.model.contrastive:
                    t = [torch.arange(_len, dtype=torch.float32, device=a_hat[0].device)[:, None] for _len in trial_len]
                    t = torch.cat(t, dim=0)
                    t_list.append(t)

                    # only use the latent behavior dimensions for contrastive learning
                    behv_batch = [b[:_len] for b, _len in zip(behv_batch, trial_len)]
                    behv_batch = torch.cat(behv_batch, dim=0)
                    behv_batch_list.append(behv_batch)

            a_hat_list = torch.cat(a_hat_list, dim=0)
            total_loss /= len(train_loader_list)
            indices = torch.randperm(a_hat_list.shape[0])[:self.config.model.contrastive_num_batch*self.config.train.batch_size]
            a_hat_list = a_hat_list[indices]
            subject_list = torch.tensor(subject_list, device=a_hat_list.device, dtype=torch.long)
            subject_list = subject_list[indices]
            if self.config.model.contrastive:
                behv_batch_list = torch.cat(behv_batch_list, dim=0)
                t_list = torch.cat(t_list, dim=0)*self.config.model.contrastive_time_scaler
                # Select 4*batch_size samples randomly
                behv_batch_list = behv_batch_list[indices]
                t_list = t_list[indices]
                contrastive_loss = self.contrastive_loss(features=a_hat_list, labels=torch.cat([behv_batch_list, t_list], dim=1))
                total_loss += contrastive_loss * self.config.model.contrastive_scale
                total_loss_dict_list.append({'contrastive_loss': contrastive_loss.detach()})

            # Compute model gradients
            self.optimizer.zero_grad()
            total_loss.backward()
            # print(total_loss_dict_list)

            # Skip gradient clipping for the first epoch
            # if epoch > 1:
            for i_sub in range(self.n_subjects):
                clip_grad_norm_(self.candy_list[i_sub].parameters(), self.config.optim.grad_clip)
            clip_grad_norm_(self.ldm.parameters(), self.config.optim.grad_clip)
            if self.config.model.supervise_behv:
                clip_grad_norm_(self.mapper.parameters(), self.config.optim.grad_clip)

            # Update model parameters
            self.optimizer.step()

            # Update metrics
            avg_loss_dict = self._update_metrics(loss_dict_list=total_loss_dict_list, 
                                 batch_size=y_batch.shape[0], 
                                 train_valid='train', 
                                 verbose=False)

            for metric, loss in avg_loss_dict.items():
                    running_losses[metric] += avg_loss_dict[metric].detach().cpu().numpy().item()
            # Update the step 
            step += 1
        
        # Save model, optimizer and learning rate scheduler (we save the initial and the last model no matter what config.model.save_steps is)
        if epoch % self.config.model.save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
            self._save_ckpt(epoch=epoch, 
                            candy_list=self.candy_list, 
                            ldm=self.ldm,
                            mapper=self.mapper,
                            optimizer=self.optimizer, 
                            lr_scheduler=self.lr_scheduler)

        # Update LR 
        self.lr_scheduler.step()

        # calculate the loss for this epoch
        epoch_loss = {key: value / count for key, value in running_losses.items()}

        # Logging the training step information for last batch
        if verbose and (epoch % self.config.train.print_log_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs):
            log_str = self._get_log_str(epoch=epoch, train_valid='train')
            self.logger.info(log_str)

        return epoch_loss

    def valid_epoch(self, epoch, valid_loader_list, verbose=True):
        '''
        Performs validation for a single epoch over batches, logging to Tensorboard and generating plots.

        Parameters: 
        ------------
        - epoch: int, The epoch number for which the validation iteration is performed.
        - valid_loader_list: list of torch.utils.data.DataLoader, The list of validation dataloaders.
        - verbose: bool, Whether to print the validation step information for the last batch. Default is True.
        '''

        with torch.no_grad():
            for i_sub in range(self.n_subjects):
                self.candy_list[i_sub].eval()

            # Reset the metrics at the beginning of each epoch
            self._reset_metrics(train_valid='valid')

            # Store the losses to calculate the average loss in the end
            running_losses = {
                'model_loss': 0,
                'behv_loss': 0,
                'behv_mse': 0,
                'reg_loss': 0,
                'total_loss': 0,
                'contrastive_loss': 0,
            }
            for k in self.config.loss.steps_ahead:
                running_losses[f'steps_{k}_mse'] = 0

            # Keep track of update step for logging the gradient norms
            min_batch_num = min([len(val_loader) for val_loader in valid_loader_list])
            step = (epoch - 1) * min_batch_num + 1
            total_steps = self.config.train.num_epochs * min_batch_num

            # Start iterating over batches
            count = 0
            for batch_idx, batch_list in enumerate(zip(*valid_loader_list)):
                count += 1
                total_loss = 0.
                total_loss_dict_list = list()
                if self.config.model.contrastive:
                    behv_batch_list = list()
                    t_list = list()
                a_hat_list = list()
                subject_list = list()
                for i_sub, batch in enumerate(batch_list):

                    # Carry data to device
                    y_batch, behv_batch, trial_len, ttype = batch
                    mask_batch = torch.ones(y_batch.shape[:-1]+(1,), dtype=torch.float32)
                    for i in range(trial_len.shape[0]):
                        mask_batch[i, trial_len[i]:] = 0
                    batch = (y_batch, behv_batch, mask_batch)
                    batch = carry_to_device(data=list(batch), device=self.device)
                    y_batch, behv_batch, mask_batch = batch

                    # Perform forward pass and compute loss
                    model_vars = self.candy_list[i_sub](y=y_batch, ldm=self.ldm, mapper=self.mapper, mask=mask_batch, normalize=False)

                    loss, loss_dict = self.candy_list[i_sub].compute_loss(y=y_batch, 
                                                            model_vars=model_vars, 
                                                            mask=mask_batch, 
                                                            behv=behv_batch)
                    if loss > 10:
                        print(f'[DEBUG] WARNING: loss is too high, check the model! [subject {i_sub}, loss {loss}]')
                    total_loss += loss

                    total_loss_dict_list.append(loss_dict)

                    a_hat = model_vars['a_hat']
                    a_hat = [a[:_len] for a, _len in zip(a_hat, trial_len)]

                    # only use the latent behavior dimensions for contrastive learning
                    a_hat = torch.cat(a_hat, dim=0)
                    subject_list.extend([i_sub]*a_hat.shape[0])
                    a_hat_list.append(a_hat)

                    if self.config.model.contrastive:
                        t = [torch.arange(_len, dtype=torch.float32, device=a_hat[0].device)[:, None] for _len in trial_len]
                        t_list.append(torch.cat(t, dim=0))
                        behv_batch = [b[:_len] for b, _len in zip(behv_batch, trial_len)]
                        behv_batch = torch.cat(behv_batch, dim=0)
                        behv_batch_list.append(behv_batch)
                
                a_hat_list = torch.cat(a_hat_list, dim=0)
                total_loss /= len(valid_loader_list)
                indices = torch.randperm(a_hat_list.shape[0])[:self.config.model.contrastive_num_batch*self.config.train.batch_size]
                a_hat_list = a_hat_list[indices]
                subject_list = torch.tensor(subject_list, device=a_hat_list.device, dtype=torch.long)
                subject_list = subject_list[indices]
                if self.config.model.contrastive:
                    behv_batch_list = torch.cat(behv_batch_list, dim=0)
                    t_list = torch.cat(t_list, dim=0)*self.config.model.contrastive_time_scaler
                    # Select 4*batch_size samples randomly
                    t_list = t_list[indices]
                    behv_batch_list = behv_batch_list[indices]
                    contrastive_loss = self.contrastive_loss(features=a_hat_list, labels=torch.cat([behv_batch_list, t_list], dim=1))
                    total_loss += contrastive_loss * self.config.model.contrastive_scale
                    total_loss_dict_list.append({'contrastive_loss': contrastive_loss.detach()})
                    
                # Update metrics
                avg_loss_dict = self._update_metrics(loss_dict_list=total_loss_dict_list, 
                                    batch_size=y_batch.shape[0], 
                                    train_valid='valid', 
                                    verbose=False)
                
                for metric, loss in avg_loss_dict.items():
                    running_losses[metric] += avg_loss_dict[metric].detach().cpu().numpy().item()
                    
                del total_loss

            # Save the best validation loss model (and best behavior reconstruction loss model if supervised)
            if self.metrics['valid']['model_loss'].compute() < self.best_val_loss:
                self.best_val_loss = self.metrics['valid']['model_loss'].compute()
                self._save_ckpt(epoch='best_loss', 
                                candy_list=self.candy_list,
                                ldm=self.ldm,
                                mapper=self.mapper,
                                optimizer=self.optimizer, 
                                lr_scheduler=self.lr_scheduler)

            if self.config.model.supervise_behv: 
                if self.metrics['valid']['behv_loss'].compute() < self.best_val_behv_loss:
                    self.best_val_behv_loss = self.metrics['valid']['behv_loss'].compute()
                    self._save_ckpt(epoch='best_behv_loss', 
                                    candy_list=self.candy_list,
                                    ldm=self.ldm,
                                    mapper=self.mapper,
                                    optimizer=self.optimizer, 
                                    lr_scheduler=self.lr_scheduler)
            
            # calculate the loss for this epoch
            epoch_loss = {key: value / count for key, value in running_losses.items()}

            if verbose and (epoch % self.config.train.print_log_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs):
                # Logging the validation step information for last batch
                log_str = self._get_log_str(epoch=epoch, train_valid='valid')
                self.logger.info(log_str)

        return epoch_loss
                                    

    def save_encoding_results(self, loader_list=None, do_full_inference=True):
        '''
        Performs inference, reconstruction, and predictions for training data and validation data (if provided), and saves training and inference time statistics.
        Then, encoding results are saved under {config.model.save_dir}/encoding_results.pt.

        Parameters:
        ------------
        - loader_list: list of torch.utils.data.DataLoader, List of data loaders for training and validation data
        - do_full_inference: bool, Whether to perform inference on flattened trials of batches of segments
        '''

        encoding_dict_list = list()
        with torch.no_grad():
            for candy, loader in zip(self.candy_list, loader_list):
                candy.eval()

                ############################################################################ BATCH INFERENCE ############################################################################
                # Create the keys for encoding results dictionary
                encoding_dict = {}

                encoding_dict['x_pred'] = list()
                encoding_dict['x_filter'] = list()
                encoding_dict['x_smooth'] = list()
                
                encoding_dict['a_hat'] = list()
                encoding_dict['a_pred'] = list()
                encoding_dict['a_filter'] = list()
                encoding_dict['a_smooth'] = list()

                encoding_dict['mask'] = list()

                y_key_list = ['y', 'y_hat', 'y_filter', 'y_smooth', 'y_pred']
                for k in self.config.loss.steps_ahead:
                    if k != 1:
                        y_key_list.append(f'y_{k}_pred')

                for y_key in y_key_list:
                    encoding_dict[y_key] = list()

                # If model is behavior-supervised, create the keys for behavior reconstruction 
                encoding_dict['behv'] = list()
                encoding_dict['behv_hat'] = list()

                encoding_dict['ttype'] = list()
                # Start iterating over dataloaders
                if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                    # If loader is not None, start iterating over the batches
                    for _, batch in enumerate(loader):
                        
                        y_batch, behv_batch, trial_len, type_batch = batch
                        mask_batch = torch.ones(y_batch.shape[:-1]+(1,), dtype=torch.int32)
                        for i in range(trial_len.shape[0]):
                            mask_batch[i, trial_len[i]:] = 0
                        batch = (y_batch, behv_batch, mask_batch, type_batch)
                        batch = carry_to_device(data=list(batch), device=self.device)
                        y_batch, behv_batch, mask_batch, type_batch = batch
                        model_vars = candy(y=y_batch, mask=mask_batch, ldm=self.ldm, mapper=self.mapper, normalize=False)

                        # Append the inference variables to the empty lists created in the beginning
                        encoding_dict['x_pred'].append(model_vars['x_pred'].detach().cpu())
                        encoding_dict['x_filter'].append(model_vars['x_filter'].detach().cpu())
                        encoding_dict['x_smooth'].append(model_vars['x_smooth'].detach().cpu())

                        encoding_dict['a_hat'].append(model_vars['a_hat'].detach().cpu())
                        encoding_dict['a_pred'].append(model_vars['a_pred'].detach().cpu())
                        encoding_dict['a_filter'].append(model_vars['a_filter'].detach().cpu())
                        encoding_dict['a_smooth'].append(model_vars['a_smooth'].detach().cpu())

                        encoding_dict['mask'].append(mask_batch.detach().cpu())
                        encoding_dict['y'].append(y_batch.detach().cpu())
                        encoding_dict['y_hat'].append(model_vars['y_hat'].detach().cpu())
                        encoding_dict['y_pred'].append(model_vars['y_pred'].detach().cpu())
                        encoding_dict['y_filter'].append(model_vars['y_filter'].detach().cpu())
                        encoding_dict['y_smooth'].append(model_vars['y_smooth'].detach().cpu())

                        for k in self.config.loss.steps_ahead:
                            if k != 1:
                                y_pred_k, _, _ = candy.get_k_step_ahead_prediction(model_vars, k)
                                encoding_dict[f'y_{k}_pred'].append(y_pred_k)
                        
                        encoding_dict['behv'].append(behv_batch.detach().cpu())
                        if self.config.model.supervise_behv:
                            encoding_dict['behv_hat'].append(model_vars['behv_hat'].detach().cpu())
                        else:
                            encoding_dict['behv_hat'].append(torch.zeros_like(behv_batch)) # behv_hat is behv

                        # record shuffled trials type 
                        encoding_dict['ttype'].append(type_batch)
                    # Convert lists to tensors (concatenate all batches)
                    encoding_dict['x_pred'] = torch.cat(encoding_dict['x_pred'], dim=0)
                    encoding_dict['x_filter'] = torch.cat(encoding_dict['x_filter'], dim=0)
                    encoding_dict['x_smooth'] = torch.cat(encoding_dict['x_smooth'], dim=0)

                    encoding_dict['a_hat'] = torch.cat(encoding_dict['a_hat'], dim=0)
                    encoding_dict['a_pred'] = torch.cat(encoding_dict['a_pred'], dim=0)
                    encoding_dict['a_filter'] = torch.cat(encoding_dict['a_filter'], dim=0)
                    encoding_dict['a_smooth'] = torch.cat(encoding_dict['a_smooth'], dim=0)

                    encoding_dict['mask'] = torch.cat(encoding_dict['mask'], dim=0)
                    for y_key in y_key_list:
                        encoding_dict[y_key] = torch.cat(encoding_dict[y_key], dim=0)

                    encoding_dict['behv'] = torch.cat(encoding_dict['behv'], dim=0)                    
                    encoding_dict['behv_hat'] = torch.cat(encoding_dict['behv_hat'], dim=0)

                    encoding_dict['ttype'] = [ttype for tup_ttype in encoding_dict['ttype'] for ttype in tup_ttype]
                ############################################################################ FULL INFERENCE w/ FLATTENED SEQUENCE ############################################################################
                encoding_dict_full_inference = {}

                if do_full_inference:
                    # Create the keys for encoding results dictionary
                    encoding_dict_full_inference = {}

                    encoding_dict_full_inference['x_pred'] = list()
                    encoding_dict_full_inference['x_filter'] = list()
                    encoding_dict_full_inference['x_smooth'] = list()
                    
                    encoding_dict_full_inference['a_hat'] = list()
                    encoding_dict_full_inference['a_pred'] = list()
                    encoding_dict_full_inference['a_filter'] = list()
                    encoding_dict_full_inference['a_smooth'] = list()

                    encoding_dict_full_inference['mask'] = list() 
                
                    for y_key in y_key_list:
                        encoding_dict_full_inference[y_key] = list() 

                    # If model is behavior-supervised, create the keys for behavior reconstruction 
                    encoding_dict_full_inference['behv'] = list()
                    encoding_dict_full_inference['behv_hat'] = list()
                    
                    encoding_dict_full_inference['ttype'] = list()

                    # Dump variables to encoding_dict_full_inference
                    if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                        # Flatten the batches of neural observations, corresponding mask and behavior if model is supervised
                        encoding_dict_full_inference['y'] = encoding_dict['y'].reshape(1, -1, candy.dim_y) 
                        encoding_dict_full_inference['mask'] = encoding_dict['mask'].reshape(1, -1, 1) 

                        # if self.config.model.supervise_behv:
                        total_dim_behv = encoding_dict['behv'].shape[-1]
                        encoding_dict_full_inference['behv'] = encoding_dict['behv'].reshape(1, -1, total_dim_behv)

                        # record trials type
                        encoding_dict_full_inference['ttype'] = encoding_dict['ttype']
                        
                        model_vars = candy(y=encoding_dict_full_inference['y'].to(self.device), mask=encoding_dict_full_inference['mask'].to(self.device), ldm=self.ldm, mapper=self.mapper, normalize=False)

                        # Append the inference variables to the empty lists created in the beginning
                        encoding_dict_full_inference['x_pred'] = model_vars['x_pred'].detach().cpu()
                        encoding_dict_full_inference['x_filter'] = model_vars['x_filter'].detach().cpu()
                        encoding_dict_full_inference['x_smooth'] = model_vars['x_smooth'].detach().cpu()

                        encoding_dict_full_inference['a_hat'] = model_vars['a_hat'].detach().cpu()
                        encoding_dict_full_inference['a_pred'] = model_vars['a_pred'].detach().cpu()
                        encoding_dict_full_inference['a_filter'] = model_vars['a_filter'].detach().cpu()
                        encoding_dict_full_inference['a_smooth'] = model_vars['a_smooth'].detach().cpu()

                        encoding_dict_full_inference['y_hat'] = model_vars['y_hat'].detach().cpu()
                        encoding_dict_full_inference['y_pred'] = model_vars['y_pred'].detach().cpu()
                        encoding_dict_full_inference['y_filter'] = model_vars['y_filter'].detach().cpu()
                        encoding_dict_full_inference['y_smooth'] = model_vars['y_smooth'].detach().cpu()

                        for k in self.config.loss.steps_ahead:
                            if k != 1:
                                y_pred_k, _, _ = candy.get_k_step_ahead_prediction(model_vars, k)
                                encoding_dict_full_inference[f'y_{k}_pred'] = y_pred_k

                        if self.config.model.supervise_behv:
                            encoding_dict_full_inference['behv_hat'] = model_vars['behv_hat'].detach().cpu()
                        else:
                            encoding_dict_full_inference['behv_hat'] = torch.zeros_like(encoding_dict_full_inference['behv'])


                # Dump batch and full inference encoding dictionaries into encoding_results
                encoding_results = dict(batch_inference=encoding_dict, full_inference=encoding_dict_full_inference)
                encoding_dict_list.append(encoding_results)

        return encoding_dict_list


    def fit(self, train_loader_list, val_loader_list, inputs=None, verbose=False, **kwargs):
        """
        This method fits the model to the training data.

        Parameters:
        ------------
        - train_loader_list: list of torch.utils.data.DataLoader, The list of training dataloaders.
        - val_loader_list: list of torch.utils.data.DataLoader, The list of validation dataloaders.
        - inputs: None, Optional inputs for the model. Default is None.
        - verbose: bool, Whether to print training information. Default is False.
        - **kwargs: Additional keyword arguments.

        Keyword Arguments:
        ------------------
        - wandb: bool, Whether to use wandb for logging. Default is False.
        - num_epochs: int, Number of epochs for training. Default is None.
        - batch_size: int, Batch size for training. Default is None.
        - lr: float, Learning rate for the optimizer. Default is None.
        - seed: int, Seed for numpy and torch. Default is None.
        - tolerance: int, Tolerance value. Default is None.
        """
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Load ckpt if asked, model with best validation model loss can be loaded as well, which is saved with name 'best_loss_ckpt.pth'
        if (isinstance(self.config.load.ckpt, int) and self.config.load.ckpt > 1) or isinstance(self.config.load.ckpt, str): 
            self.candy_list, self.ldm, self.mapper, self.optimizer, self.lr_scheduler = self._load_ckpt(candy_list=self.candy_list,
                                                                            ldm=self.ldm,
                                                                            mapper=self.mapper,
                                                                            optimizer=self.optimizer,
                                                                            lr_scheduler=self.lr_scheduler)

        # Bookkeeping the validation NRMSEs over the course of training
        self.training_valid_one_step_nrmses = []

        # Start iterating over the epochs
        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            # Perform validation with the initialized model
            if epoch == self.start_epoch:
                self.valid_epoch(epoch, val_loader_list, verbose=False)

            # Perform training iteration over train_loader
            train_epoch_loss = self.train_epoch(epoch, train_loader_list)

            # Perform validation over valid_loader if it's not None and we're at validation epoch
            if (epoch % self.config.train.valid_step == 0):
                valid_epoch_loss = self.valid_epoch(epoch, val_loader_list)
            
            # Update the global losses buffer
            epoch_losses = {'train': train_epoch_loss, 'valid': valid_epoch_loss}
            for train_valid, epoch_loss in epoch_losses.items():
                self.losses[train_valid]['behv_losses'].append(epoch_loss['behv_loss'])
                self.losses[train_valid]['model_losses'].append(epoch_loss['model_loss'])
                self.losses[train_valid]['reg_losses'].append(epoch_loss['reg_loss'])
                self.losses[train_valid]['total_losses'].append(epoch_loss['total_loss'])
            # Write to wandb
            if kwargs['wandb']:
                wandb_log = {
                    'epoch': epoch + 1,
                }
                for key, value in train_epoch_loss.items():
                    wandb_log[f'train_{key}'] = value 
                for key, value in valid_epoch_loss.items():
                    wandb_log[f'valid_{key}'] = value
                wandb.log(wandb_log)

   
    def fit_transform(self, train_loader_list, val_loader_list, inputs=None, verbose=False, **kwargs):
        self.fit(train_loader_list, val_loader_list, inputs=None, verbose=verbose, **kwargs)
        Z_bytrial, data_recon = self.transform(train_loader_list, verbose=verbose, do_full_inference=True)
        return Z_bytrial, data_recon


    def transform(self, data_loader_list, verbose=False, do_full_inference=True, data_type='hat'):
        results = self.save_encoding_results(data_loader_list, do_full_inference)
        return transform2bytrial_byagent(results, data_type=data_type)


    def forecast(self, z, inputs=None, horizon=1):
        raise NotImplementedError


class Mean(Metric):
    '''
    Mean metric class to log batch-averaged metrics to Tensorboard. 
    '''

    def __init__(self):
        '''
        Initializer for Mean metric. Note that this class is a subclass of torchmetrics.Metric.
        '''

        super().__init__(dist_sync_on_step=False)
        
        # Define total sum and number of samples that sum is computed over
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")


    def update(self, value, batch_size):
        '''
        Updates the total sum and number of samples

        Parameters: 
        ------------
        - value: torch.Tensor, shape: (), Value to add to sum
        - batch_size: torch.Tensor, shape: (), Number of samples that 'value' is averaged over
        '''

        value = value.clone().detach()
        batch_size = torch.tensor(batch_size, dtype=torch.float32)
        self.sum += value.cpu() * batch_size
        self.num_samples += batch_size


    def reset(self):
        '''
        Resets the total sum and number of samples to 0
        '''

        self.sum = torch.tensor(0, dtype=torch.float32)
        self.num_samples = torch.tensor(0, dtype=torch.float32)


    def compute(self):
        '''
        Computes the mean metric.

        Returns: 
        ------------
        - avg: Average value for the metric
        '''

        avg = self.sum / self.num_samples
        return avg


class CANDY(nn.Module):
    '''
    CANDY (Dynamical Flexible Inference for Nonlinear Embeddings) Model. 

    CANDY is a novel neural network model of neural population activity with the ability to perform 
    flexible inference while modeling the nonlinear latent manifold structure and linear temporal dynamics. 
    To model neural population activity, two sets of latent factors are defined: the dynamic latent factors 
    which characterize the linear temporal dynamics on a nonlinear manifold, and the manifold latent factors 
    which describe this low-dimensional manifold that is embedded in the high-dimensional neural population activity space. 
    These two separate sets of latent factors together enable all the above flexible inference properties 
    by allowing for Kalman filtering on the manifold while also capturing embedding nonlinearities.
    Here are some mathematical notations used in this repository:
    - y: The high dimensional neural population activity, (num_seq, num_steps, dim_y). It must be Gaussian distributed, e.g., Gaussian-smoothed firing rates, or LFP, ECoG, EEG
    - a: The manifold latent factors, (num_seq, num_steps, dim_a).
    - x: The dynamic latent factors, (num_seq, num_steps, dim_x).


    * Please note that CANDY can perform learning and inference both for continuous data or trial-based data or segmented continuous data. In the case of continuous data,
    num_seq and batch_size can be set to 1, and we let the model be optimized from the long time-series (this is basically gradient descent and not batch-based gradient descent). 
    In case of trial-based data, we can just pass the 3D tensor as the shape (num_seq, num_steps, dim_y) suggests. In case of segmented continuous data,
    num_seq can be the number of segments and CANDY provides both per-segment and concatenated inference at the end for the user's convenience. In the concatenated inference, 
    the assumption is the concatenation of segments form a continuous time-series (single time-series with batch size of 1).
    '''

    def __init__(self, config):
        '''
        Initializer for an CANDY object. Note that CANDY is a subclass of torch.nn.Module. 

        Parameters: 
        ------------

        - config: yacs.config.CfgNode, yacs config which contains all hyperparameters required to create the CANDY model
                                       Please see config_candy.py for the hyperparameters, their default values and definitions. 
        '''

        super(CANDY, self).__init__()

        # Get the config and dimension parameters
        self.config = config

        # Set the seed, seed is by default set to a random integer, see config_candy.py
        torch.manual_seed(self.config.seed)

        # Set the factor dimensions and loss scales
        self._set_dims_and_scales()

        # Initialize encoder and decoder(s)
        self.encoder = self._get_MLP(input_dim=self.dim_y, 
                                     output_dim=self.dim_a, 
                                     layer_list=self.config.model.hidden_layer_list, 
                                     activation_str=self.config.model.activation)

        self.decoder = self._get_MLP(input_dim=self.dim_a, 
                                     output_dim=self.dim_y, 
                                     layer_list=self.config.model.hidden_layer_list[::-1], 
                                     activation_str=self.config.model.activation)
        
    def _set_dims_and_scales(self):
        '''
        Sets the observation (y), manifold latent factor (a) and dynamic latent factor (x)
        (and behavior data dimension if supervised model is to be trained) dimensions,
        as well as behavior reconstruction loss and regularization loss scales from config. 
        '''

        # Set the dimensions
        self.dim_y = self.config.model.dim_y
        self.dim_a = self.config.model.dim_a
        self.dim_x = self.config.model.dim_x

        if self.config.model.supervise_behv:
            self.dim_behv = len(self.config.model.which_behv_dims)
        
        # Set the loss scales for behavior component and for the regularization
        if self.config.model.supervise_behv:
            self.scale_behv_recons = self.config.loss.scale_behv_recons
        self.scale_l2 = self.config.loss.scale_l2


    def _get_MLP(self, input_dim, output_dim, layer_list, activation_str='tanh'):
        '''
        Creates an MLP object

        Parameters:
        ------------
        - input_dim: int, Dimensionality of the input to the MLP network
        - output_dim: int, Dimensionality of the output of the MLP network
        - layer_list: list, List of number of neurons in each hidden layer
        - activation_str: str, Activation function's name, 'tanh' by default

        Returns: 
        ------------
        - mlp_network: an instance of MLP class with desired architecture
        '''

        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.nn_kernel_initializer)
    
        mlp_network = MLP(input_dim=input_dim,
                          output_dim=output_dim,
                          layer_list=layer_list,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )
        return mlp_network

    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, y, ldm, mapper, mask=None, normalize=False):
        '''
        Forward pass for CANDY Model

        Parameters: 
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), High-dimensional neural observations
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                              observations at each timestep exist (1) or are missing (0)

        Returns: 
        ------------
        - model_vars: dict, Dictionary which contains learned parameters, inferrred latents, predictions and reconstructions. Keys are: 
            - a_hat: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors. 
            - a_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_a), Batch of predicted estimates of manifold latent factors (last index of the second dimension is removed)
            - a_filter: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of filtered estimates of manifold latent factors 
            - a_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of smoothed estimates of manifold latent factors 
            - x_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x), Batch of predicted estimates of dynamic latent factors
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - x_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of smoothed estimates of dynamic latent factors
            - Lambda_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x, dim_x), Batch of predicted estimates of dynamic latent factor estimation error covariance
            - Lambda_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of filtered estimates of dynamic latent factor estimation error covariance
            - Lambda_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of smoothed estimates of dynamic latent factor estimation error covariance
            - y_hat: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of projected estimates of neural observations
            - y_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_y), Batch of predicted estimates of neural observations
            - y_filter: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of filtered estimates of neural observations
            - y_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of smoothed estimates of neural observations
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Repeated (tile) state transition matrix of LDM, same for each time-step in the 2nd axis
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x), Repeated (tile) observation matrix of LDM, same for each time-step in the 2nd axis
            - behv_hat: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of reconstructed behavior. None if unsupervised model is trained

        * Terminology definition:
            projected: noisy estimations of manifold latent factors after nonlinear manifold embedding via encoder 
            predicted: one-step ahead predicted estimations (t+1|t), the first and last time indices are (1|0) and (T|T-1)
            filtered: causal estimations (t|t)
            smoothed: non-causal estimations (t|T)
        '''

        # Get the dimensions from y
        num_seq, num_steps, _ = y.shape

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Get the encoded low-dimensional manifold factors (project via nonlinear manifold embedding) -> the outputs are (num_seq * num_steps, dim_a)
        a_hat = self.encoder(y.view(-1, self.dim_y))

        # Reshape the manifold latent factors back into 3D structure (num_seq, num_steps, dim_a)
        a_hat = a_hat.view(-1, num_steps, self.dim_a)

        # shared LDM for extracting behavior-relevant dynamics
        # Run LDM to infer filtered and smoothed dynamic latent factors
        x_pred, x_filter, x_smooth, Lambda_pred, Lambda_filter, Lambda_smooth = ldm(a=a_hat, mask=mask, do_smoothing=True, normalize=normalize)
        A = ldm.A.repeat(num_seq, num_steps, 1, 1)
        C = ldm.C.repeat(num_seq, num_steps, 1, 1)
        a_pred = (C @ x_pred.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_filter = (C @ x_filter.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_smooth = (C @ x_smooth.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)

        # Remove the last timestep of predictions since it's T+1|T, which is not of our interest
        x_pred = x_pred[:, :-1, :]
        Lambda_pred = Lambda_pred[:, :-1, :, :]
        a_pred = a_pred[:, :-1, :]

        # Supervise a_seq or a_smooth to behavior if requested -> behv_hat shape: (num_seq, num_steps, dim_behv)
        if self.config.model.supervise_behv:
            if self.config.model.behv_from_smooth:
                behv_hat = mapper(a_smooth.view(-1, self.dim_a))
            else:
                behv_hat = mapper(a_hat.view(-1, self.dim_a))
            behv_hat = behv_hat.view(-1, num_steps, self.dim_behv)
        else:
            behv_hat = None

        # Get filtered and smoothed estimates of neural observations. To perform k-step-ahead prediction, 
        # get_k_step_ahead_prediction(...) function should be called after the forward pass. 
        y_hat = self.decoder(a_hat.view(-1, self.dim_a))
        y_pred = self.decoder(a_pred.reshape(-1, self.dim_a))
        y_filter = self.decoder(a_filter.view(-1, self.dim_a))
        y_smooth = self.decoder(a_smooth.view(-1, self.dim_a))

        y_hat = y_hat.view(num_seq, -1, self.dim_y)
        y_pred = y_pred.view(num_seq, -1, self.dim_y)
        y_filter = y_filter.view(num_seq, -1, self.dim_y)
        y_smooth = y_smooth.view(num_seq, -1, self.dim_y)

        # Dump inferrred latents, predictions and reconstructions to a dictionary
        model_vars = dict(a_hat=a_hat, a_pred=a_pred, a_filter=a_filter, a_smooth=a_smooth, 
                          x_pred=x_pred, x_filter=x_filter, x_smooth=x_smooth,
                          Lambda_pred=Lambda_pred, Lambda_filter=Lambda_filter, Lambda_smooth=Lambda_smooth,
                          y_hat=y_hat, y_pred=y_pred, y_filter=y_filter, y_smooth=y_smooth, 
                          A=A, C=C, behv_hat=behv_hat)
        return model_vars


    def get_k_step_ahead_prediction(self, model_vars, k):
        '''
        Performs k-step ahead prediction of manifold latent factors, dynamic latent factors and neural observations. 

        Parameters: 
        ------------
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information. 
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x) or (dim_x, dim_x), State transition matrix of LDM
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x) or (dim_y, dim_x), Observation matrix of LDM
        - k: int, Number of steps ahead for prediction

        Returns: 
        ------------
        - y_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_y), Batch of predicted estimates of neural observations, 
                                                                           the first index of the second dimension is y_{k|0}
        - a_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_a), Batch of predicted estimates of manifold latent factor, 
                                                                        the first index of the second dimension is a_{k|0}                                                              
        - x_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_x), Batch of predicted estimates of dynamic latent factor, 
                                                                        the first index of the second dimension is x_{k|0}  
        '''

        # Check whether provided k value is valid or not
        if k <= 0 or not isinstance(k, int):
            assert False, 'Number of steps ahead prediction value is invalid or of wrong type, k must be a positive integer!'

        # Extract the required variables from model_vars dictionary
        x_filter = model_vars['x_filter']
        A = model_vars['A']
        C = model_vars['C']

        # Get the required dimensions
        num_seq, num_steps, _ = x_filter.shape

        # Check if shapes of A and C are 4D where first 2 dimensions are (number of trials/time segments) and (number of steps)
        if len(A.shape) == 2:
            A = A.repeat(num_seq, num_steps, 1, 1)

        if len(C.shape) == 2:
            C = C.repeat(num_seq, num_steps, 1, 1)

        # Here is where k-step ahead prediction is iteratively performed
        x_pred_k = x_filter[:, :-k, ...] # [x_k|0, x_{k+1}|1, ..., x_{T}|{T-k}]
        for i in range(1, k+1):
            if i != k:
                x_pred_k = (A[:, i:-(k-i), ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)  
            else:
                x_pred_k = (A[:, i:, ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)
        a_pred_k = (C[:, k:, ...] @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)

        # After obtaining k-step ahead predicted manifold latent factors, they're decoded to obtain k-step ahead predicted neural observations
        y_pred_k = self.decoder(a_pred_k.view(-1, self.dim_a))

        # Reshape mean and variance back to 3D structure after decoder (num_seq, num_steps, dim_y)
        y_pred_k = y_pred_k.reshape(num_seq, -1, self.dim_y)

        return y_pred_k, a_pred_k, x_pred_k

    
    def compute_loss(self, y, model_vars, mask=None, behv=None):
        '''
        Computes k-step ahead predicted MSE loss, regularization loss and behavior reconstruction loss
        if supervised model is being trained. 

        Parameters: 
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of high-dimensional neural observations
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                              observations at each timestep exists (1) or are missing (0)
                                                              if None it will be set to ones.
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information. 
        - behv: torch.tensor, shape: (num_seq, num_steps, dim_behv), Batch of behavior data

        Returns: 
        ------------
        - loss: torch.Tensor, shape: (), Loss to optimize, which is sum of k-step-ahead MSE loss, L2 regularization loss and 
                                         behavior reconstruction loss if model is supervised
        - loss_dict: dict, Dictionary which has all loss components to log on Tensorboard. Keys are (e.g. for config.loss.steps_ahead = [1, 2]): 
            - steps_{k}_mse: torch.Tensor, shape: (), {k}-step ahead predicted masked MSE, k's are determined by config.loss.steps_ahead
            - model_loss: torch.Tensor, shape: (), Negative of sum of all steps_{k}_mse
            - behv_loss: torch.Tensor, shape: (), Behavior reconstruction loss, 0 if model is unsupervised
            - reg_loss: torch.Tensor, shape: (), L2 Regularization loss for CANDY encoder and decoder weights
            - total_loss: torch.Tensor, shape: (), Sum of model_loss, behv_loss and reg_loss
        '''

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1], dtype=torch.float32).unsqueeze(dim=-1)

        # Dump individual loss values for logging or Tensorboard
        loss_dict = dict()
        
        # Iterate over multiple steps ahead
        k_steps_mse_sum = 0  
        for _, k in enumerate(self.config.loss.steps_ahead):
            y_pred_k, _, _ = self.get_k_step_ahead_prediction(model_vars, k=k)
            mse_pred = compute_mse(y_flat=y[:, k:, :].reshape(-1, self.dim_y), 
                                   y_hat_flat=y_pred_k.reshape(-1, self.dim_y),
                                   mask_flat=mask[:, k:, :].reshape(-1,))

            # if torch.isnan(mse_pred).any():
            #     mse_pred = torch.zeros_like(mse_pred)
            k_steps_mse_sum += mse_pred
            loss_dict[f'steps_{k}_mse'] = mse_pred.detach().cpu()

        model_loss = k_steps_mse_sum
        loss_dict['model_loss'] = model_loss.detach().cpu()

        # Get MSE loss for behavior reconstruction, 0 if we dont supervise our model with behavior data
        if self.config.model.supervise_behv:
            behv_mse = compute_mse(y_flat=behv[..., self.config.model.which_behv_dims].reshape(-1, self.dim_behv), 
                                   y_hat_flat=model_vars['behv_hat'].reshape(-1, self.dim_behv),
                                   mask_flat=mask.reshape(-1,))
            behv_loss = self.scale_behv_recons * behv_mse
        else:
            behv_mse = torch.tensor(0, dtype=torch.float32, device=model_loss.device)
            behv_loss = torch.tensor(0, dtype=torch.float32, device=model_loss.device)
        loss_dict['behv_mse'] = behv_mse.detach().cpu()
        loss_dict['behv_loss'] = behv_loss.detach().cpu()

        # L2 regularization loss 
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + self.scale_l2 * torch.norm(param)
        loss_dict['reg_loss'] = reg_loss.detach().cpu()

        # Final loss is summation of model loss (sum of k-step ahead MSEs), behavior reconstruction loss and L2 regularization loss
        loss = model_loss + behv_loss + reg_loss
        loss_dict['total_loss'] = loss.detach().cpu()
        return loss, loss_dict


class LDM(nn.Module):
    '''
    Linear Dynamical Model backbone for CANDY. This module is used for smoothing and filtering
    given a batch of trials/segments/time-series. 

    LDM equations are as follows:
    x_{t+1} = Ax_{t} + w_{t}; cov(w_{t}) = W
    a_{t} = Cx_{t} + r_{t}; cov(r_{t}) = R
    '''

    def __init__(self, **kwargs):
        '''
        Initializer for an LDM object. Note that LDM is a subclass of torch.nn.Module.

        Parameters
        ------------
        - dim_x: int, Dimensionality of dynamic latent factors, default None  
        - dim_a: int, Dimensionality of manifold latent factors, default None
        - is_W_trainable: bool, Whether dynamics noise covariance matrix (W) is learnt or not, default True
        - is_R_trainable: bool, Whether observation noise covariance matrix (R) is learnt or not, default True
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM, default identity
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM, default identity
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor estimate initial condition (x_{0|-1}) for Kalman filtering, default zeros 
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman Filtering, default identity
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of process noise covariance matrix (W, therefore it is diagonal and PSD), default ones
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD), default ones
        '''

        super(LDM, self).__init__()

        self.dim_x = kwargs.pop('dim_x', None)
        self.dim_a = kwargs.pop('dim_a', None)

        self.is_W_trainable = kwargs.pop('is_W_trainable', True)
        self.is_R_trainable = kwargs.pop('is_R_trainable', True)

        # # Initializer for identity matrix, zeros matrix and ones matrix
        # self.eye_init = lambda shape, dtype=torch.float32: torch.eye(*shape, dtype=dtype)
        # self.zeros_init = lambda shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype)
        # self.ones_init = lambda shape, dtype=torch.float32: torch.ones(*shape, dtype=dtype)

        # Get initial values for LDM parameters
        self.A = kwargs.pop('A', torch.eye(self.dim_x, self.dim_x, dtype=torch.float32).unsqueeze(dim=0)).type(torch.FloatTensor)
        self.C = kwargs.pop('C', torch.eye(self.dim_a, self.dim_x, dtype=torch.float32).unsqueeze(dim=0)).type(torch.FloatTensor)
        
        # Get KF initial conditions
        self.mu_0 = kwargs.pop('mu_0', torch.zeros((self.dim_x, ), dtype=torch.float32)).type(torch.FloatTensor)
        self.Lambda_0 = kwargs.pop('Lambda_0', torch.eye(self.dim_x, self.dim_x, dtype=torch.float32)).type(torch.FloatTensor)

        # Get initial process and observation noise parameters
        self.W_log_diag = kwargs.pop('W_log_diag', torch.ones((self.dim_x, ), dtype=torch.float32)).type(torch.FloatTensor)
        self.R_log_diag = kwargs.pop('R_log_diag', torch.ones((self.dim_a, ), dtype=torch.float32)).type(torch.FloatTensor)

        # Register trainable parameters to module
        self._register_params()


    def _register_params(self):
        '''
        Registers the learnable LDM parameters as nn.Parameters
        '''

        # Check if LDM matrix shapes are consistent
        self._check_matrix_shapes()

        # Register LDM parameters
        self.A = torch.nn.Parameter(self.A, requires_grad=True)
        self.C = torch.nn.Parameter(self.C, requires_grad=True)
        
        self.W_log_diag = torch.nn.Parameter(self.W_log_diag, requires_grad=self.is_W_trainable)
        self.R_log_diag = torch.nn.Parameter(self.R_log_diag, requires_grad=self.is_R_trainable)  

        self.mu_0 = torch.nn.Parameter(self.mu_0, requires_grad=True)
        self.Lambda_0 = torch.nn.Parameter(self.Lambda_0, requires_grad=True)


    def _check_matrix_shapes(self):
        '''
        Checks whether LDM parameters have the correct shapes, which are defined above in the constructor
        '''
        
        # Check A matrix's shape
        if self.A.shape != (self.dim_x, self.dim_x):
            assert False, 'Shape of A matrix is not (dim_x, dim_x)!'

        # Check C matrix's shape
        if self.C.shape != (self.dim_a, self.dim_x):
            assert False, 'Shape of C matrix is not (dim_a, dim_x)!'
        
        # Check mu_0 matrix's shape
        if len(self.mu_0.shape) != 1:
            self.mu_0 = self.mu_0.view(-1, )

        if self.mu_0.shape != (self.dim_x, ):
            assert False, 'Shape of mu_0 matrix is not (dim_x, )!'

        # Check Lambda_0 matrix's shape
        if self.Lambda_0.shape != (self.dim_x, self.dim_x):
            assert False, 'Shape of Lambda_0 matrix is not (dim_x, dim_x)!'

        # Check W_log_diag matrix's shape
        if len(self.W_log_diag.shape) != 1:
            self.W_log_diag = self.W_log_diag.view(-1, )

        if self.W_log_diag.shape != (self.dim_x, ):
            assert False, 'Shape of W_log_diag matrix is not (dim_x, )!'

        # Check R_log_diag matrix's shape
        if len(self.R_log_diag.shape) != 1:
            self.R_log_diag = self.R_log_diag.view(-1, )

        if self.R_log_diag.shape != (self.dim_a, ):
            assert False, 'Shape of R_log_diag matrix is not (dim_x, )!'


    def _get_covariance_matrices(self):
        '''
        Get the process and observation noise covariance matrices from log-diagonals. 

        Returns:
        ------------
        - W: torch.Tensor, shape: (self.dim_x, self.dim_x), Process noise covariance matrix
        - R: torch.Tensor, shape: (self.dim_a, self.dim_a), Observation noise covariance matrix
        '''

        # W = torch.diag(torch.exp(self.W_log_diag))
        # R = torch.diag(torch.exp(self.R_log_diag))
        W = torch.diag(torch.nn.functional.softplus(self.W_log_diag) + 1e-6)
        R = torch.diag(torch.nn.functional.softplus(self.R_log_diag) + 1e-6)
        # print('max: ', torch.max(torch.exp(self.W_log_diag)), torch.max(torch.exp(self.R_log_diag)), torch.min(torch.exp(self.R_log_diag)))
        return W, R
    

    def compute_forwards(self, a, mask=None):
        '''
        Performs the forward iteration of causal flexible Kalman filtering, given a batch of trials/segments/time-series

        Parameters: 
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)

        Returns: 
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''

        
        def robust_psd_inverse(S: torch.Tensor, jitter_base: float = 1e-6, max_tries: int = 6) -> torch.Tensor:
            """
            Stable inverse of symmetric (near-)PSD S using Cholesky + adaptive jitter.
            Handles batched S with shape (..., m, m). Returns S^{-1} with same batch shape.
            """
            # enforce symmetry
            S = 0.5 * (S + S.transpose(-1, -2))
            *batch, m, _ = S.shape
            device, dtype = S.device, S.dtype

            # batch-shaped identity to match S (..., m, m)
            I = torch.eye(m, device=device, dtype=dtype).expand(*batch, m, m)

            # scale jitter to S's magnitude; never below jitter_base
            mean_diag = S.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)  # (...,1,1)
            jitter = torch.clamp(jitter_base * mean_diag, min=jitter_base)                            # (...,1,1)

            for _ in range(max_tries):
                try:
                    # Add jitter * batched identity
                    L = torch.linalg.cholesky(S + jitter * I)
                    # Solve S X = I via cholesky_solve; RHS must match batch dims
                    S_inv = torch.cholesky_solve(I, L)
                    return S_inv
                except RuntimeError:
                    jitter = jitter * 10.0  # escalate jitter and retry

            # Last resort: dense solve (RHS must match batch dims)
            return torch.linalg.solve(S + jitter * I, I)

        if mask is None:
            mask = torch.ones(a.shape[:-1], dtype=torch.float32)
        
        num_seq, num_steps, _ = a.shape

        # Make sure that mask is 3D (last axis is 1-dimensional)
        if len(mask.shape) != len(a.shape):
            mask = mask.unsqueeze(dim=-1) # (num_seq, num_steps, 1)

        # To make sure we do not accidentally use the real outputs in the steps with missing values, set them to a dummy value, e.g., 0.
        # The dummy values of observations at masked points are irrelevant because:
        # Kalman disregards the observations by setting Kalman Gain to 0 in K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1)) @ line 204
        a_masked = torch.mul(a, mask) # (num_seq, num_steps, dim_a) x (num_seq, num_steps, 1)
        
        # Initialize mu_0 and Lambda_0 
        mu_0 = self.mu_0.unsqueeze(dim=0).repeat(num_seq, 1) # (num_seq, dim_x)
        Lambda_0 = self.Lambda_0.unsqueeze(dim=0).repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)

        mu_pred = mu_0 # (num_seq, dim_x)
        Lambda_pred = Lambda_0 # (num_seq, dim_x, dim_x)

        # Create empty arrays for filtered and predicted estimates, NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        mu_pred_all = torch.zeros((num_steps, num_seq, self.dim_x), dtype=torch.float32, device=mu_0.device)
        mu_t_all = torch.zeros((num_steps, num_seq, self.dim_x), dtype=torch.float32, device=mu_0.device)

        # Create empty arrays for filtered and predicted error covariance, NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        Lambda_pred_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), dtype=torch.float32, device=mu_0.device)
        Lambda_t_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), dtype=torch.float32, device=mu_0.device)

        # Get covariance matrices 
        W, R = self._get_covariance_matrices()
        # print(f"Norm of A: {torch.norm(self.A)}, Norm of C: {torch.norm(self.C)}")

        for t in range(num_steps):
            # Tile C matrix for each time segment
            C_t = self.C.repeat(num_seq, 1, 1)

            # Obtain residual
            a_pred = (C_t @ mu_pred.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)
            r = a_masked[:, t, ...] - a_pred # (num_seq, dim_a)

            # Project system uncertainty into measurement space, get Kalman Gain
            S = C_t @ Lambda_pred @ torch.permute(C_t, (0, 2, 1)) + R # num_seq, dim_a, dim_a)
            # try:
            #     # S_inv = torch.inverse(S) # num_seq, dim_a, dim_a)
            #     S_inv = torch.linalg.solve(S, torch.eye(S.size(-1), device=S.device)) # num_seq, dim_a, dim_a)
            # except torch.linalg.LinAlgError as e:
            #     # print(e)
            #     S_inv = torch.pinverse(S)
            S_inv = robust_psd_inverse(S)
            K = Lambda_pred @ torch.permute(C_t, (0, 2, 1)) @ S_inv # (num_seq, dim_x, dim_a)
            K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1))  # (num_seq, dim_x, dim_a) x (num_seq, 1,  1)

            # Get current mu and Lambda
            mu_t = mu_pred + (K @ r.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x)
            I_KC = torch.eye(self.dim_x, dtype=torch.float32, device=mu_0.device) - K @ C_t # (num_seq, dim_x, dim_x)
            Lambda_t = I_KC @ Lambda_pred # (num_seq, dim_x, dim_x)

            # Tile A matrix for each time segment
            A_t = self.A.repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)

            # Prediction 
            mu_pred = (A_t @ mu_t.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x, dim_x) x (num_seq, dim_x, 1) --> (num_seq, dim_x, 1) --> (num_seq, dim_x)
            Lambda_pred = A_t @ Lambda_t @ torch.permute(A_t, (0, 2, 1)) + W # (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) --> (num_seq, dim_x, dim_x)

            # Keep predictions and updates
            mu_pred_all[t, ...] = mu_pred
            mu_t_all[t, ...] = mu_t

            Lambda_pred_all[t, ...] = Lambda_pred
            Lambda_t_all[t, ...] = Lambda_t

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all

        
    def filter(self, a, mask=None):
        '''
        Performs Kalman Filtering  

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)

        Returns: 
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''
        
        # Run the forward iteration
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, mask=mask)

        # Swab num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all


    def compute_backwards(self, mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all):
        '''
        Performs backward iteration for Rauch-Tung-Striebel (RTS) Smoother

        Parameters:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}

        Returns: 
        ------------
        - mu_back_all: torch.Tensor, shape: (num_steps, num_seq, dim_x), Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}
        - Lambda_back_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x), Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}
        '''

        # Get number of steps and number of trials  
        num_steps, num_seq, _ = mu_pred_all.shape 

        # Create empty arrays for smoothed dynamic latent factors and error covariances
        mu_back_all = torch.zeros((num_steps, num_seq, self.dim_x), dtype=torch.float32, device=mu_pred_all.device) # (num_steps, num_seq, dim_x)
        Lambda_back_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), dtype=torch.float32, device=mu_pred_all.device) # (num_steps, num_seq, dim_x, dim_x)

        # Last smoothed estimation is equivalent to the filtered estimation
        mu_back_all[-1, ...] = mu_t_all[-1, ...]
        Lambda_back_all[-1, ...] = Lambda_t_all[-1, ...]

        # Initialize iterable parameter
        mu_back = mu_t_all[-1, ...]
        Lambda_back = Lambda_back_all[-1, ...]

        for t in range(num_steps-2, -1, -1): # iterate loop over reverse time: T-2, T-3, ..., 0, where the last time-step is T-1
            A_t = self.A.repeat(num_seq, 1, 1)

            try:
                J_t = Lambda_t_all[t, ...] @ torch.permute(A_t, (0, 2, 1)) @ torch.inverse(Lambda_pred_all[t, ...]) # (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x)
            except torch.linalg.LinAlgError:
                J_t = Lambda_t_all[t, ...] @ torch.permute(A_t, (0, 2, 1)) @ torch.pinverse(Lambda_pred_all[t, ...]) # (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x)
            mu_back = mu_t_all[t, ...] + (J_t @ (mu_back - mu_pred_all[t, ...]).unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x) + (num_seq, dim_x, dim_x) x (num_seq, dim_x)

            Lambda_back = Lambda_t_all[t, ...] + J_t @ (Lambda_back - Lambda_pred_all[t, ...]) @ torch.permute(J_t, (0, 2, 1)) # (num_seq, dim_x, dim_x)

            mu_back_all[t, ...] = mu_back
            Lambda_back_all[t, ...] = Lambda_back

        return mu_back_all, Lambda_back_all


    def smooth(self, a, mask=None):
        '''
        Performs Rauch-Tung-Striebel (RTS) Smoothing

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)

        Returns: 
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - mu_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        - Lambda_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}
        '''
        
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, mask=mask)
        mu_back_all, Lambda_back_all = self.compute_backwards(mu_pred_all=mu_pred_all, 
                                                             mu_t_all=mu_t_all, 
                                                             Lambda_pred_all=Lambda_pred_all, 
                                                             Lambda_t_all=Lambda_t_all)

        # Swab num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        mu_back_all = torch.permute(mu_back_all, (1, 0, 2))

        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))
        Lambda_back_all = torch.permute(Lambda_back_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all


    def forward(self, a, mask=None, do_smoothing=False, normalize=False):
        '''
        Forward pass function for LDM Module

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)
        do_smoothing: bool, Whether to run RTS Smoothing or not

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - mu_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}. Ones tensor if do_smoothing is False
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        - Lambda_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}. Ones tensor if do_smoothing is False
        '''
        # self.A.data = cap_eigenvalues(self.A.data, cap=10.0)
        if self.A.data.numel() > 0:
            A_norm = torch.linalg.norm(self.A.data, ord=2)
        else:
            # print(f'[WARNING] A is empty', self.A.data, flush=True)
            A_norm = 0.
        if self.C.data.numel() > 0:
            C_norm = torch.linalg.norm(self.C.data, ord=2)
        else:
            # print(f'[WARNING] C is empty', self.C.data, flush=True)
            C_norm = 0.
        norm_threshold = 10.0
        if normalize or (A_norm > norm_threshold) or (C_norm > norm_threshold):
            if self.A.data.numel() > 0 and A_norm > norm_threshold:
                # self.A.data = self.A.data / max(1.0, torch.linalg.norm(self.A.data, ord=2))
                self.A.data = self.A.data / A_norm * min(A_norm, norm_threshold)
            if self.C.data.numel() > 0 and C_norm > norm_threshold:
                self.C.data = self.C.data / C_norm * min(C_norm, norm_threshold)
        if do_smoothing:
            mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all = self.smooth(a=a, mask=mask) 
        else:
            mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.filter(a=a, mask=mask)
            mu_back_all = torch.ones_like(mu_t_all, dtype=torch.float32, device=mu_t_all.device)
            Lambda_back_all = torch.ones_like(Lambda_t_all, dtype=torch.float32, device=Lambda_t_all.device)

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all


class MLP(nn.Module):
    '''
    MLP Module for CANDY encoder and decoder in addition to the mapper to behavior for supervised CANDY. 
    Encoder encodes the high-dimensional neural observations into low-dimensional manifold latent factors space 
    and decoder decodes the manifold latent factors into high-dimensional neural observations.
    '''

    def __init__(self, **kwargs):
        '''
        Initializer for an Encoder/Decoder/Mapper object. Note that Encoder/Decoder/Mapper is a subclass of torch.nn.Module.

        Parameters
        ------------
        input_dim: int, Dimensionality of inputs to the MLP, default None
        output_dim: int, Dimensionality of outputs of the MLP , default None
        layer_list: list, List of number of neurons in each hidden layer, default None
        kernel_initializer_fn: torch.nn.init, Hidden layer weight initialization function, default nn.init.xavier_normal_
        activation_fn: torch.nn, Activation function of neurons, default nn.Tanh
        '''

        super(MLP, self).__init__()
        
        self.input_dim = kwargs.pop('input_dim', None)
        self.output_dim = kwargs.pop('output_dim', None)
        self.layer_list = kwargs.pop('layer_list', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', nn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', nn.Tanh)

        # Create the ModuleList to stack the hidden layers 
        self.layers = nn.ModuleList()
        
        # Create the hidden layers and initialize their weights based on desired initialization function
        current_dim = self.input_dim
        for i, dim in enumerate(self.layer_list):
            self.layers.append(nn.Linear(current_dim, dim))
            self.kernel_initializer_fn(self.layers[i].weight)
            current_dim = dim

        # Create output layer and initialize their weights based on desired initialization function
        self.out_layer = nn.Linear(current_dim, self.output_dim)
        self.kernel_initializer_fn(self.out_layer.weight)

        
    def forward(self, inp):
        '''
        Forward pass function for MLP Module 

        Parameters: 
        ------------
        inp: torch.Tensor, shape: (num_seq * num_steps, input_dim), Flattened batch of inputs

        Returns: 
        ------------
        out: torch.Tensor, shape: (num_seq * num_steps, output_dim),Flattened batch of outputs
        '''

        # Push neural observations thru each hidden layer
        for layer in self.layers:
            inp = layer(inp)
            inp = self.activation_fn(inp)
        
        # Obtain the output
        out = self.out_layer(inp)
        return out
    

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None