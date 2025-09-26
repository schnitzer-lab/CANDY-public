import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.decoding.base_decoding import Decoder

def identity_func(x):
    return x

class MLP(nn.Module):
    '''
    MLP Module for DFINE encoder and decoder in addition to the mapper to behavior for supervised DFINE. 
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

class NonLinearDecoder(Decoder):
    def __init__(self, **params):
        super().__init__()
        self.seed = params['seed']
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.hidden_layers = params['hidden_layer_list_mapper']
        activation_str = params.pop('activation_mapper', 'linear')
        kernal_str     = params.pop('kernal_initializer', 'xavier_normal')
        self.grad_clip = params.pop('grad_clip', 1)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(kernal_str)
    
        self.model = MLP(input_dim=self.input_dim,
                          output_dim=self.output_dim,
                          layer_list=self.hidden_layers,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )

        # Training configuration
        self.device = params.pop("device", 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=params.pop("lr", 1e-3),  # Increased learning rate
                                  weight_decay=params.pop("weight_decay", 1e-3))  # Optional L2
        self.num_epochs = params.pop("num_epochs", 300)  # Increased epochs
        self.batch_size = params.pop("batch_size", 64)  # Added batch size

        # Checkpoint setup
        self.ckpt_save_dir = params.pop("ckpt_save_dir")
        self.checkpoint_folder = os.path.join(self.ckpt_save_dir, "behv_ckpt")
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_folder, "best_losses.pth")
        self.best_val_loss = float('inf')

        scaler_type = params.pop('normalizor', None)
        if scaler_type is None:
            self.scaler = None 
        elif scaler_type == 'zscore':
            self.scaler = StandardScaler()

    def fit(self, X, y, X_valid, y_valid, **params):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Prepare training data
        X = np.vstack(X)
        y = np.vstack(y).squeeze()
        if self.scaler is not None:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        
        # Create DataLoader with batches
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Prepare validation data
        if X_valid is not None and y_valid is not None:
            X_valid = np.vstack(X_valid)
            X_val_scaled = X_valid # self.scaler.transform(X_valid)  # Critical scaling fix
            y_valid = np.vstack(y_valid).squeeze()
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_loader = None

        # Training loop
        self.losses = {'train': [], 'valid': []}
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                if epoch > 1:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)
            
            # Track average training loss
            epoch_train_loss /= len(train_loader.dataset)
            self.losses['train'].append(epoch_train_loss)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        outputs = outputs.squeeze()
                        val_loss += self.criterion(outputs, batch_y).item() * batch_X.size(0)
                val_loss /= len(val_loader.dataset)
                self.losses['valid'].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_loss': self.best_val_loss
                    }, self.checkpoint_path)

            # Progress reporting
            if epoch % 50 == 0:
                val_msg = f" | Val Loss: {val_loss:.4f}" if val_loader else ""
                print(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {epoch_train_loss:.4f}{val_msg}")
    
    def load_best_model(self):
        """
        Load the best model from the checkpoint into the current model.
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded best model with validation loss {checkpoint['best_val_loss']:.4f} from {self.checkpoint_path}.")
        else:
            print(f"[INFO] No checkpoint found at {self.checkpoint_path}.")
    
    def predict(self, X, load_best=False):
        """
        Make predictions for X (a list of numpy array).
        If load_best is True, load the best model checkpoint before predicting.
        Returns predictions as a numpy array.
        """
        if load_best:
            self.load_best_model()

        self.model.eval()
        y_pred_bytrial = []
        for x in X:
            if self.scaler is not None:
                x = self.scaler.transform(x)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            y_pred = self.model(x_tensor) 
            y_pred_bytrial.append(y_pred.detach().cpu().numpy())

        return y_pred_bytrial
    
    # def score(self, y_true_bytrial, y_pred_bytrial):
    #     """
    #     Compute the R^2 score of the predictions.
    #     """
    #     behv_dims = y_true_bytrial[0].shape[-1]
    #     print(f'[INFO] behavior dimension is {behv_dims}')
    #     y_true = np.concatenate(y_true_bytrial, axis=0).squeeze()
    #     y_pred = np.concatenate(y_pred_bytrial, axis=0).squeeze()
    #     mse = mean_squared_error(y_true, y_pred)
    #     mae = mean_absolute_error(y_true, y_pred)
    #     r2  = r2_score(y_true, y_pred)
    #     y_true_mean = y_true - np.mean(y_true, axis=0, keepdims=True)
    #     y_pred_mean = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    #     numerator = np.sum(y_true_mean * y_pred_mean, axis=0)
    #     denominator = np.sqrt(np.sum(y_true_mean**2, axis=0)) * np.sqrt(np.sum(y_pred_mean**2, axis=0))
    #     corr = numerator / denominator

    #     return {'MSE': mse, 'MAE': mae, 'R2': r2, 'Corr': corr}#, 'Corr_pvalue': corr.pvalue}