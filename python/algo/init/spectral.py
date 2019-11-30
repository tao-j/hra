from .init import Initializer
import numpy as np
import scipy
import scipy.sparse


class PopulationInitializer(Initializer):
    def get_initialization_point(self):
        if self.config.ground_truth_prob_mat:
            p = self.calculate_probability_ground_truth(self.data_pack.s_true, self.data_pack.beta_true)
        else:
            p = self.data_pack.count_mat
        # shape is n_judges n_items*n_items^T
        n_judges = p.shape[0]
        n_items = p.shape[1]

        # ------------ use all judge info to calc
        mixed = p.sum(axis=0)
        t = self.calculate_transition(mixed, self.config.prob_regularization)
        w = self.calculate_stationary_distribution(t)
        s_init = np.log(np.abs(w))
        # normalize s for easy comparison
        s_init -= np.min(s_init)
        # TODO: removed summation
        s_init /= s_init.sum()

        # s_init = np.array(list(map(float, open('s_init.txt').read().split('\n'))))
        return s_init


class IndividualInitializer(Initializer):
    def __init__(self, data_pack, config):
        super(IndividualInitializer, self).__init__(data_pack, config)
        self.popular_correction = config.popular_correction

    def get_initialization_point(self):
        if self.config.ground_truth_prob_mat:
            p = self.calculate_probability_ground_truth(self.data_pack.s_true, self.data_pack.beta_true)
        else:
            p = self.data_pack.count_mat
        # shape is n_judges n_items*n_items^T
        n_judges = p.shape[0]
        n_items = p.shape[1]

        mixed = p.sum(axis=0)
        if self.popular_correction:
            p = np.concatenate([mixed[np.newaxis, :], p], axis=0)
            n_judges += 1

        # ------------ use each judge to calc
        # e^(s/beta) = esdb = w (normalized) = u
        # s/beta = sdb = q
        beta_init = np.zeros(n_judges)
        beta_init[0] = 1.
        s_init = np.ones(n_items) * 0.

        # step 1:
        qs = []
        for c_i in range(n_judges):
            d = p[c_i]
            t = self.calculate_transition(d, self.config.prob_regularization)

            w = self.calculate_stationary_distribution(t)
            # bol = w == 0
            # idx = 0
            # for i in range(bol.shape[0]):
            #     if bol[idx] != True:
            #         idx += 1
            #     else:
            #         break
            # if idx < bol.shape[0]:
            #     p[idx][idx] = np.nan
            #     print(p[idx])
            #     print(p[:][idx])

            # first_flag = True
            # for wmin in np.sort(w):
            #     if wmin > 10e-10:
            #         break
            #     first_flag = False
            #
            # if not first_flag:
            #     w += wmin
            #     w = w / w.sum()

            a = np.ones(n_items - 1) - np.diag(1. / (w[1:] + self.config.err_const))
            a = np.vstack([np.ones((1, n_items - 1)), a])

            b = np.array([-1.] * n_items)
            b[0] += 1. / (w[0] + self.config.err_const)

            # u1 = np.matmul(np.matmul(np.linalg.gamma(np.matmul(A.T, A)), A.T), b)
            u = scipy.sparse.linalg.lsqr(a, b, show=self.verbose)[0]

            q = np.hstack([np.log(np.abs(u))])
            qs.append(q)

        qs = np.vstack(qs)

        n_items_1 = n_items - 1
        # step 2:
        a = np.zeros((n_items_1 * n_judges, n_items_1 + n_judges - 1))
        for c_i in range(n_judges):
            for s_i in range(n_items_1):
                if c_i != 0:
                    a[n_items_1 * c_i + s_i][n_items_1 + c_i - 1] = -qs[c_i][s_i]
                a[n_items_1 * c_i + s_i][s_i] = 1

        b = np.ones((n_items_1 * n_judges,)) * 0.
        for s_i in range(n_items_1):
            b[s_i] = qs[0][s_i]

        # qq1 = np.matmul(np.matmul(np.linalg.gamma(np.matmul(A.T, A)), A.T), b)
        qq = scipy.sparse.linalg.lsqr(a, b, show=self.verbose)[0]

        # assignment for return
        for s_i in range(n_items_1):
            s_init[s_i + 1] = qq[s_i]
        for b_i in range(n_judges - 1):
            beta_init[b_i + 1] = qq[n_items_1 + b_i]

        if self.popular_correction:
            beta_init = beta_init[1:]

        return s_init, beta_init
