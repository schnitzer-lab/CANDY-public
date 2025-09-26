import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class Dynamics(object):
    def __init__(self, obs_dim, latent_dim, **kwargs):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
    
    def fit(self, data):
        pass

    def fit_transform(self, data, **kwargs):
        """
        INPUT
            [orig_data]
        OUTPUT
            [latent_states]
            [data_recon]
        """
        return data, data

    def transform(self, data):
        """
        INPUT
            [orig_data]
        OUTPUT
            [latent_states]
            [data_recon]
        """
        return data, data
    
    def inverse_transform(self, z, data):
        return data

    def forecast(self, z):
        raise NotImplementedError

    def scoring(self, data, data_recon):
        X_true  = np.vstack(data)
        X_recon = np.vstack(data_recon)
        if not np.any(np.isnan(X_recon)):
            mse = mean_squared_error(X_true, X_recon)
            mae = mean_absolute_error(X_true, X_recon)
            r2  = r2_score(X_true.T, X_recon.T)
            # vx = X_recon - np.mean(X_recon, axis=1, keepdims=True)
            # vy = X_true - np.mean(X_true, axis=1, keepdims=True)
            # corr = np.mean(np.sum(vx * vy, axis=1) / (np.sqrt(np.sum(vx ** 2, axis=1)) * np.sqrt(np.sum(vy ** 2, axis=1))))
            X_true_mean = X_true - np.mean(X_true, axis=0, keepdims=True)
            X_recon_mean = X_recon - np.mean(X_recon, axis=0, keepdims=True)
            numerator = np.sum(X_true_mean * X_recon_mean, axis=0)
            denominator = np.sqrt(np.sum(X_true_mean**2, axis=0)) * np.sqrt(np.sum(X_recon_mean**2, axis=0))
            corr = np.mean(numerator / denominator)

        else:
            mse = np.nan 
            mae = np.nan
            r2  = np.nan 
            corr = np.nan

        return {'MSE': mse, 'MAE': mae, 'R2': r2, 'Corr': corr}

class sharedDynamics(object):
    def __init__(self, obs_dim_lst, latent_dim, **kwargs):
        self.obs_dim = obs_dim_lst
        self.latent_dim = latent_dim
    
    def fit(self, data_lst):
        pass

    def fit_transform(self, data_lst):
        return data_lst

    def transform(self, data_lst):
        return data_lst
    
    def inverse_transform(self, z_lst, data_lst):
        return data_lst

    def forecast(self, z):
        raise NotImplementedError

    def scoring(self, data_lst, data_recon_lst):
        num_agent = len(data_lst)
        results_lst = []
        for i in range(num_agent):
            data = data_lst[i]
            data_recon = data_recon_lst[i]
            X_true  = np.vstack(data)
            X_recon = np.vstack(data_recon)
            mse = mean_squared_error(X_true, X_recon)
            mae = mean_absolute_error(X_true, X_recon)
            r2  = r2_score(X_true.T, X_recon.T)

            # FIXME 
            # vx = X_recon - np.mean(X_recon, axis=1, keepdims=True)
            # vy = X_true - np.mean(X_true, axis=1, keepdims=True)
            # corr = np.mean(np.sum(vx * vy, axis=1) / (np.sqrt(np.sum(vx ** 2, axis=1)) * np.sqrt(np.sum(vy ** 2, axis=1))))
            X_true_mean = X_true - np.mean(X_true, axis=0, keepdims=True)
            X_recon_mean = X_recon - np.mean(X_recon, axis=0, keepdims=True)
            numerator = np.sum(X_true_mean * X_recon_mean, axis=0)
            denominator = np.sqrt(np.sum(X_true_mean**2, axis=0)) * np.sqrt(np.sum(X_recon_mean**2, axis=0))
            corr = np.mean(numerator / denominator)

            results_dict = {'MSE': mse, 'MAE': mae, 'R2': r2, 'Corr': corr}

            results_lst.append(results_dict)

        return results_lst
