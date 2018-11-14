from algo.ra.ra import RankAggregation

from algo.init.spectral import PopulationInitializer
from algo.init.estimated import MLInitializer, MomentInitializer
from algo.init.init import GroundTruthInitializer, RandomInitializer
from algo.init import *

import numpy as np
import scipy.optimize as op


class BTLNaive(RankAggregation):
    def __init__(self, data_pack, config):
        super(BTLNaive, self).__init__(data_pack, config)
        self.parameters.append(self.s)
        self.named_parameters['s'] = self.s

    def initialize(self):
        init_method = self.config.init_method
        if init_method == INIT_RANDOM:
            initializer = RandomInitializer(self.data_pack, self.config)
        elif init_method == INIT_SPECTRAL:
            initializer = PopulationInitializer(self.data_pack, self.config)
        elif init_method == INIT_GROUND_TRUTH:
            initializer = GroundTruthInitializer(self.data_pack, self.config)
        elif init_method == INIT_MOMENT:
            initializer = MomentInitializer(self.data_pack, self.config)
        elif init_method == INIT_ML:
            initializer = MLInitializer(self.data_pack, self.config)
        else:
            raise NotImplementedError

        self.s_init = initializer.get_initialization_point()

        if init_method == INIT_RANDOM:
            self.s_init = initializer.get_initialization_point()[0]
        self.s += self.s_init

    def compute_likelihood_sparse(self):
        pr = 0.
        # s[j] is winner
        for item, cnt in self.data_cnt.items():
            j, i, k = item
            pr += - cnt * np.log(np.exp((self.s[i] - self.s[j])) + 1)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_np(self, s, beta=None):
        # s_j is winner
        sr_j = self.replicator * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        pr = - np.sum(np.sum(self.count_mat, axis=0) * np.log(np.exp(si_minus_sj) + 1))
        return -pr / self.n_pairs

    def compute_gradient_s(self, s):
        grad = np.zeros(self.n_items)
        if self.data_cnt:
            for item, cnt in self.data_cnt.items():
                # s[i] is winner
                i, j, _ = item
                v = np.exp(s[j]) / (np.exp(s[j]) + np.exp(s[i]))
                grad[i] -= cnt * v
                grad[j] += cnt * v
        # else:
        #     mix = np.sum(self.count_mat, axis=0)
        #     for i in range(self.n_items):
        #         for j in range(self.n_items):
        #             cnt = mix[i][j]
        #             if not cnt:
        #                 continue
        #             v = np.exp(s[j]) / (np.exp(s[j]) + np.exp(s[i]))
        #             grad[i] -= cnt * v
        #             grad[j] += cnt * v
        return grad

    def optimization_step(self):
        res = op.minimize(fun=self.compute_likelihood_np,
                    x0=np.zeros(self.n_items),
                    method='L-BFGS-B',
                    jac=self.compute_gradient_s)
        if not res.success:
            print(res)
        self.s = res.x
        return True

    def post_process(self):
        self.s -= np.min(self.s)

    def consolidate_result(self):
        return self.data_pack.beta_true