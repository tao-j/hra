import numpy as np
import scipy
import scipy.sparse
import scipy.special


class Initializer:
    def __init__(self, data_pack, config):
        self.data_pack = data_pack
        self.config = config
        self.verbose = False

    @staticmethod
    def calculate_probability_ground_truth(s_true, beta_true):
        m_size = beta_true.shape[0]
        n_size = s_true.shape[0]
        t = np.zeros((m_size, n_size, n_size), dtype=np.double)
        for k in range(m_size):
            for i in range(n_size):
                for j in range(n_size):
                    if i != j:
                        t[k][j][i] = 1. / (1 + np.exp((s_true[j] - s_true[i]) / beta_true[k]))
        return t

    @staticmethod
    def calculate_transition(prob_mat, prob_regularization=False):
        import copy
        prob_mat = copy.deepcopy(prob_mat)
        # NOTE: adjusted here
        if prob_regularization:
            prob_mat += 0.1
        n_size = prob_mat.shape[0]
        prob_mat = prob_mat.astype(np.double)
        for i in range(n_size):
            for j in range(i + 1, n_size):
                if (prob_mat[i][j] + prob_mat[j][i]) != 0:
                    prob_mat[j][i] = prob_mat[j][i] / (prob_mat[i][j] + prob_mat[j][i])
                    prob_mat[i][j] = 1. - prob_mat[j][i]
                else:
                    prob_mat[i][j] = 0.
                    prob_mat[j][i] = 0.

        outer_degree = np.max((prob_mat > 0).sum(axis=1))
        #     outer_degree = (c > 0).sum(axis=1)
        t = (prob_mat.T / (outer_degree + 1e-10)).T
        row_sum = t.sum(axis=1)
        t = t + np.eye(n_size) * (1 - row_sum)
        return t

    @staticmethod
    def calculate_stationary_distribution(t):
        m_size = t.shape[0]

        e = t - np.eye(m_size)
        e = e.T
        e[-1] = np.ones(m_size)

        y = np.zeros(m_size)
        y[-1] = 1

        #     res = np.linalg.solve(e, y)
        w = np.matmul(np.linalg.inv(e), y)
        return w


class RandomInitializer(Initializer):
    def get_initialization_point(self):
        p = self.data_pack.count_mat
        # shape is n_judges n_items*n_items^T
        n_judges = p.shape[0]
        n_items = p.shape[1]

        s_init = np.random.random(n_items)
        s_init -= np.min(s_init)
        s_init /= np.sum(s_init)
        beta_init = np.random.random(n_judges) * 0.05
        return s_init, beta_init


class GroundTruthInitializer(Initializer):
    def get_initialization_point(self):
        return self.data_pack.s_true, self.data_pack.beta_true

    # TODO: disturb ground truth by little bit noise
    # if init_method == INIT_RANDOM:
    #     pass
    #
    # if init_method == INIT_SPECTRAL:
    #     s_init_tout, s_init_individual, beta_init = calc_s_beta(data_mat, verbose=verbose)
    #     eps_init = np.sqrt(np.abs(beta_init))
    #     if override_beta:
    #         beta_init = np.random.random(n_judges) * 0.05
    #         eps_init = np.random.random(n_judges) * 0.05
    #
    # if init_method == INIT_GROUND_TRUTH:
    #     s_init = s_true + np.random.normal(0, ground_truth_disturb, size=n_items)
    #     beta_init = beta_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
    #     eps_init = eps_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
    #     if override_beta:
    #         beta_init = np.random.random(n_judges) * 0.05
    #         eps_init = np.random.random(n_judges) * 0.05
    #     else:
    #         lr = 1e-5