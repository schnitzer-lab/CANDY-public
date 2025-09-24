import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class Decoder(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass

    def score(self, y_true_bytrial, y_pred_bytrial):
        """
        Compute the R^2 score of the predictions.
        """
        behv_dims = y_true_bytrial[0].shape[-1]
        # print(f'[INFO] behavior dimension is {behv_dims}')
        y_true = np.concatenate(y_true_bytrial, axis=0).squeeze()
        y_pred = np.concatenate(y_pred_bytrial, axis=0).squeeze()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        y_true_mean = y_true - np.mean(y_true, axis=0, keepdims=True)
        y_pred_mean = y_pred - np.mean(y_pred, axis=0, keepdims=True)
        numerator = np.sum(y_true_mean * y_pred_mean, axis=0)
        denominator = np.sqrt(np.sum(y_true_mean**2, axis=0)) * np.sqrt(np.sum(y_pred_mean**2, axis=0))
        corr = numerator / denominator

        return {'MSE': mse, 'MAE': mae, 'R2': r2, 'Corr': corr}#, 'Corr_pvalue': corr.pvalue}