from algo.ra.ra import RankAggregation

from algo.init.spectral import PopulationInitializer
from algo.init.estimated import MLInitializer, MomentInitializer
from algo.init.init import GroundTruthInitializer, RandomInitializer
from algo.init import *

import torch
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

        # TODO: temp fix, random initializer doesn't work, because one more param is returned
        # TODO: numpy-backend
        if init_method == INIT_RANDOM:
            self.s_init = initializer.get_initialization_point()[0]
        self.s += self.s_init

        # TODO: pytorch-backend
        '''
        if init_method == INIT_RANDOM:
            self.s_init, _ = initializer.get_initialization_point()
        else:
            self.s_init = initializer.get_initialization_point()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype)
        '''
    def compute_likelihood_sparse(self):
        pr = 0.
        # s[j] is winner
        for item, cnt in self.data_cnt.items():
            j, i, k = item
            pr += - cnt * np.log(np.exp((self.s[i] - self.s[j])) + 1)
        self.pr = -pr / self.n_pairs
        return self.pr

    def compute_likelihood_count_mat(self, s, beta=None):
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
        return grad

    def optimization_step(self):
       #  self.s = np.array([0.2535809 , 0.03395629, 0.02179904, 0.14150608, 0.15534016,
       # 0.22660238, 0.48947107, 0.52969226, 0.78312124, 0.84059461])
       #  print('============================')
       #  print(self.compute_likelihood_sparse(), 'LL sparse')
       #  print(self.compute_likelihood_count_mat(self.s), 'LL ')
       #  print(self.compute_gradient_s(self.s), 'gradient s')
       #  print('==============================')
        res = op.minimize(fun=self.compute_likelihood_count_mat,
                          x0=np.zeros(self.n_items),
                          method='L-BFGS-B',
                          jac=self.compute_gradient_s)
        if not res.success:
            print(res)
        self.s = res.x
        return True
    '''# pytorch-backend
        def compute_likelihood(self):
            sr_j = self.replicator * self.s  # each column is the same value
            sr_i = torch.transpose(sr_j, 1, 0)
            si_minus_sj = sr_i - sr_j
    
            pr = - torch.sum(torch.sum(self.count_mat, dim=0) * torch.log(torch.exp(si_minus_sj) + 1))
            self.pr = -pr / self.n_pairs
    
        def compute_likelihood_sparse(self):
            pr = torch.tensor(0, dtype=self.dtype).to(self.device)
            for item, cnt in self.data_cnt.items():
                i, j, k = item
                pr += - cnt * torch.log(torch.exp((self.s[j] - self.s[i])) + 1)
            self.pr = -pr / self.n_pairs
    
        def compute_likelihood_np(self, s, beta):
            sr_j = self.replicator.cpu().numpy() * s  # each column is the same value
            sr_i = sr_j.T
            si_minus_sj = sr_i - sr_j
    
            pr = - np.sum(np.sum(self.count_mat.cpu().numpy(), dim=0) * np.log(np.exp(si_minus_sj) + 1))
            return -pr / self.n_pairs
    
        def post_process(self):
            self.s.data -= torch.min(self.s.data)
    '''

    def post_process(self):
        self.s -= np.min(self.s)

    def consolidate_result(self):
        return self.data_pack.beta_true