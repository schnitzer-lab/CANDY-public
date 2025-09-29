import numpy as np
import pandas as pd 
from threadpoolctl import threadpool_limits
import os

from scipy import stats

from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold, GridSearchCV

from src.decoding.base_decoding import Decoder

class LinearDecoder(Decoder):
    def __init__(self, model_name='lasso', **kwargs):
        super().__init__()
        normalize = kwargs['normalize']
        tol       = kwargs['tol']
        match model_name:
            case 'lasso':
                self.model = linear_model.Lasso()
            case 'ridge':
                self.model = linear_model.Ridge()
            case 'regression':
                self.model = linear_model.LinearRegression()
            case 'elastic_net':
                self.model = linear_model.ElasticNet()
    
    def fit(self, X, y, X_valid, y_valid, **kwargs):
        """
        This function trian the linear behavior decoder with K-fold cross-validation and grid-search on the hyperparameters.

        INPUT:
            [X] : a list of [num_trials] input with shape [T_i x latent_dim] of trial i.
            [y] : a list of [num_trials] ground truth output with shape [T_i x num_behaviors] of trial i.
            [kwargs] : keywords are (params_grid, scoring, n_split, n_repeats, seed)
        """
        params_grid = kwargs['params_grid']
        scoring     = kwargs['scoring']
        n_split     = kwargs['n_split']
        n_repeats   = kwargs['n_repeats']
        seed        = kwargs['seed']
        
        cv = RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=seed)
        # cpu_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        # num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
        # total_cpus = cpu_per_task * num_tasks
        # total_cpus = int(2**np.floor(np.log2(total_cpus)))
        # print(f'Total CPUs: {total_cpus}')
        search = GridSearchCV(self.model, params_grid, scoring=scoring, cv=cv, n_jobs=1)

        X = np.vstack(X)
        y = np.vstack(y).squeeze()
                # ---------- single‑CPU guard ----------
        with threadpool_limits(8):            # BLAS / OpenMP layer
            self.model = search.fit(X, y)
    
    def predict(self, X):
        """
        Return the predicted behavior [y_pred] from the input [X].

        INPUT:
            [X] : a list of [num_trials] input with shape [T_i x latent_dim] of trial i.
        OUTPUT:
            [y_pred_bytrial] : a list of [num_trials] output with shape [T_i x num_behaviors] of trial i.
        """
        y_pred_bytrial = []
        for x in X:
            y_pred = self.model.predict(x) 
            y_pred_bytrial.append(y_pred)
        return y_pred_bytrial

    # def score(self, y_true_bytrial, y_pred_bytrial):
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
