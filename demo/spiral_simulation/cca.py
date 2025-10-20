from scipy.linalg import qr, svd
from sklearn.decomposition import PCA
import numpy as np
#!/usr/bin/python
#-*- coding: utf-8 -*-

def check_rank(Q, R, time_len, n_neuron):
    rank = sum(np.abs(np.diagonal(R)) > np.finfo(type((np.abs(R[0,0])))).eps*max([time_len, n_neuron]))
    # print(rank, Q.shape, R.shape, time_len, n_neuron)

    if rank == 0:
        raise ValueError
    elif rank < n_neuron:
        # logging.warning('stats:canoncorr:NotFullRank = X')
        Q = Q[:,:rank]
        R = R[rank,:rank]
    
    return rank, Q, R

class CCA:
    def __init__(self, n_pc) -> None:
        self.n_pc = n_pc

    def fit(self, X_1, X_2):
        # CCA algorithm to align x_2 to x_1
        # X_1: n_neuron_1 * T
        # X_2: n_neuron_2 * T
        # n_pc: number of principle components
        #
        # [return]
        # X_1: PCA-smoothed X_1
        # X_2: PCA_smoothed X_2
        # align_mat: matrix that aligns X_2 to X_1
        # S: correlation coefficient between each neural mode

        n_neuron_1, time_len = X_1.shape
        n_neuron_2, time_len = X_2.shape
        X_1 = X_1.T
        X_2 = X_2.T
        X_1 -= X_1.mean(0, keepdims=True)
        X_2 -= X_2.mean(0, keepdims=True)
        pca_1 = PCA(n_components=self.n_pc).fit(X_1)
        pca_2 = PCA(n_components=self.n_pc).fit(X_2)
        X_1 = pca_1.transform(X_1)
        X_2 = pca_2.transform(X_2)

        # QR decomposition
        Q_1, R_1, perm_1 = qr(X_1, pivoting=True, mode='economic', check_finite=True)
        rank_1, Q_1, R_1 = check_rank(Q_1, R_1, time_len, self.n_pc)
        Q_2, R_2, perm_2 = qr(X_2, pivoting=True, mode='economic', check_finite=True)
        rank_2, Q_2, R_2 = check_rank(Q_2, R_2, time_len, self.n_pc)
        d = min(rank_1, rank_2)

        Q_1_T_Q_2 = Q_1.T @ Q_2

        # SVD
        try:
            U, S, Vh = svd(Q_1_T_Q_2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
        except np.linalg.LinAlgError:
            U, S, Vh = svd(Q_1_T_Q_2, full_matrices=True, check_finite=True, lapack_driver='gesvd')

        # new manifold
        M_1 = np.linalg.inv(R_1) @ U[:, :d] * np.sqrt(time_len)
        M_2 = np.linalg.inv(R_2) @ Vh.T[:, :d] * np.sqrt(time_len)
        S = S[:d]
        S[S>=1] = 1
        S[S<=0] = 0

        # Put coefficients back to their full size and their correct order
        M_1[perm_1,:] = np.vstack((M_1, np.zeros((self.n_pc-rank_1, d))))
        M_2[perm_2,:] = np.vstack((M_2, np.zeros((self.n_pc-rank_2, d))))

        # Compute the canonical variates
        X_hat_1 = X_1 @ M_1
        X_hat_2 = X_2 @ M_2

        X_hat_1 = X_hat_1.T
        X_hat_2 = X_hat_2.T

        self.M_1 = M_1
        self.M_2 = M_2
        self.S = S
        self.pca_1 = pca_1
        self.pca_2 = pca_2

        return X_hat_1, X_hat_2, S
    
    def transform(self, X_1, X_2):
        # X_1: n_neuron_1 * T
        # X_2: n_neuron_2 * T
        X_1 = X_1.T
        X_2 = X_2.T
        X_1 -= X_1.mean(0, keepdims=True)
        X_2 -= X_2.mean(0, keepdims=True)
        X_1 = self.pca_1.transform(X_1)
        X_2 = self.pca_2.transform(X_2)

        X_hat_1 = X_1 @ self.M_1
        X_hat_2 = X_2 @ self.M_2
        X_hat_1 = X_hat_1.T
        X_hat_2 = X_hat_2.T
        return X_hat_1, X_hat_2
    
if __name__ == '__main__':
    t_arr = np.linspace(0, 2*np.pi, 100)
    x = np.array([np.sin(t_arr), np.cos(t_arr)])
    x_1 = np.array([np.sin(t_arr+np.pi/4), np.cos(t_arr+np.pi/4)/2])
    x_2 = np.array([np.sin(t_arr+np.pi/4.5), np.cos(t_arr+np.pi/4.5)/2])

    cca = CCA(2)
    x_hat, x_hat_1, S = cca.fit(x, x_1)
    x_align_1 = (x_hat_1.T @ np.linalg.inv(cca.M_1)).T

    pca = PCA(n_components=2)
    x_pca_2 = pca.fit_transform(x_2.T)
    x_pca_2 = x_pca_2.T
    x_align_2 = (x_pca_2.T @ cca.M_2 @ np.linalg.inv(cca.M_1)).T

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    ax.plot(x[0], x[1], 'r', label='x')
    ax.plot(x_1[0], x_1[1], 'b', label='x_1')
    ax.plot(x_2[0], x_2[1], 'g', label='x_2')
    ax.set_title('Original')

    ax = axs[1]
    ax.plot(x[0], x[1], 'r', label='x')
    ax.plot(x_align_1[0], x_align_1[1], 'b', label='x_1')
    ax.plot(x_align_2[0], x_align_2[1], 'b', label='x_2')
    ax.set_title('Aligned')
    plt.show()