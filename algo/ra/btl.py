from algo.ra.ra import RankAggregation

from algo.init.spectral import PopulationInitializer
from algo.init.estimated import MLInitializer, MomentInitializer
from algo.init.init import GroundTruthInitializer, RandomInitializer
from algo.init import *

import torch
import numpy as np

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
        if init_method == INIT_RANDOM:
            self.s_init, _ = initializer.get_initialization_point()
        else:
            self.s_init = initializer.get_initialization_point()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype)

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

    def consolidate_result(self):
        return self.data_pack.beta_true