from algo.init.spectral import IndividualInitializer
from algo.init.estimated import MLInitializer, MomentInitializer
from algo.init.init import RandomInitializer, GroundTruthInitializer
from algo.init import *
from algo.ra.ra import RankAggregation

import torch
import numpy as np


class GBTL(RankAggregation):
    def __init__(self, data_pack, config):
        super(GBTL, self).__init__(data_pack, config)
        self.beta_init = None
        self.beta_true = data_pack.beta_true
        self.popular_correction=self.config.popular_correction

    def get_initializer(self):
        init_method = self.config.init_method
        if init_method == INIT_RANDOM:
            initializer = RandomInitializer(self.data_pack, self.config)
        elif init_method == INIT_SPECTRAL:
            initializer = IndividualInitializer(self.data_pack, self.config)
        elif init_method == INIT_GROUND_TRUTH:
            initializer = GroundTruthInitializer(self.data_pack, self.config)
        elif init_method == INIT_MOMENT:
            initializer = MomentInitializer(self.data_pack, self.config)
        elif init_method == INIT_ML:
            initializer = MLInitializer(self.data_pack, self.config)
        else:
            raise NotImplementedError

        self.s_init, self.beta_init =\
            initializer.get_initialization_point()


class GBTLEpsilon(GBTL):
    def __init__(self, data_pack, config):
        super(GBTLEpsilon, self).__init__(data_pack, config)
        self.eps = torch.tensor(np.zeros(self.n_judges), device=self.device, dtype=self.dtype, requires_grad=True)
        self.parameters.append(self.eps)
        self.named_parameters['eps'] = self.eps

    def initialize(self):
        self.get_initializer()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype)
        self.eps.data += torch.tensor(np.sqrt(np.abs(self.beta_init)), device=self.device, dtype=self.dtype)

    def compute_likelihood(self):
        sr_j = self.replicator * self.s  # each column is the same value
        sr_i = torch.transpose(sr_j, 1, 0)
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.view((1,) + si_minus_sj.shape)
        ep = (self.eps * self.eps).view(self.eps.shape + (1, 1))
        qu = ex / ep
        mask = (qu > 23).double()
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = torch.log(torch.exp(q_exact) + 1) + q_approx
        pr = - torch.sum(self.count_mat * lg)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_sparse(self):
        pr = torch.tensor(0, dtype=self.dtype).to(self.device)
        for item, cnt in self.data_cnt.items():
            i, j, k = item
            pr += - cnt * torch.log(torch.exp((self.s[j] - self.s[i]) / self.eps[k] / self.eps[k]) + 1)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_np(self, s, eps):
        sr_j = self.replicator.cpu().numpy() * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.reshape((1,) + si_minus_sj.shape)
        ep = (eps * eps).reshape(eps.shape + (1, 1))
        qu = ex / ep
        mask = (qu > 23)
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = np.log(np.exp(q_exact) + 1) + q_approx
        pr = -np.sum(self.count_mat.cpu().numpy() * lg)
        return -pr / self.n_pairs

    def post_process(self):
        self.s.data -= torch.min(self.s.data)
        idx = 0
        s_ratio = 1. / self.eps.data.cpu().numpy()[idx]
        self.eps.data = self.eps.data * np.sqrt(s_ratio)
        self.s.data = self.s.data * s_ratio

    def consolidate_result(self):
        beta_res = self.eps.data.cpu().numpy() ** 2
        return beta_res


class GBTLBeta(GBTL):
    def __init__(self, data_pack, config):
        super(GBTLBeta, self).__init__(data_pack, config)
        self.beta = torch.tensor(np.zeros(self.n_judges), device=self.device, dtype=self.dtype, requires_grad=True)
        self.parameters.append(self.beta)
        self.named_parameters['beta'] = self.beta
        self.beta_true = data_pack.beta_true

    def initialize(self):
        self.get_initializer()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype, requires_grad=False)
        self.beta.data += torch.tensor(self.beta_init, device=self.device, dtype=self.dtype, requires_grad=False)

    def compute_likelihood(self):
        sr_j = self.replicator * self.s  # each column is the same value
        sr_i = torch.transpose(sr_j, 1, 0)
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.view((1,) + si_minus_sj.shape)
        ep = self.beta.view(self.beta.shape + (1, 1))
        qu = ex / ep
        mask = (qu > 23).double()
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = torch.log(torch.exp(q_exact) + 1) + q_approx
        pr = - torch.sum(self.count_mat * lg)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_sparse(self):
        pr = torch.tensor(0, dtype=self.dtype).to(self.device)
        for item, cnt in self.data_cnt.items():
            i, j, k = item
            pr += - cnt * torch.log(torch.exp((self.s[j] - self.s[i]) / self.beta[k]) + 1)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_np(self, s, beta):
        sr_j = self.replicator.cpu().numpy() * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.reshape((1,) + si_minus_sj.shape)
        ep = beta.reshape(beta.shape + (1, 1))
        qu = ex / ep
        mask = (qu > 23)
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = np.log(np.exp(q_exact) + 1) + q_approx
        pr = - np.sum(self.count_mat.cpu().numpy() * lg)
        return -pr / self.n_pairs

    # def compute_gradient(self):
    #     grad_b = torch.sum(torch.sum(self.count_mat * (-1. / ep / ep * ex / (torch.exp(-qu) + 1)), dim=-1), dim=-1) / n_pairs

    def post_process(self):
        self.s.data -= torch.min(self.s.data)
        idx = 0
        s_ratio = np.abs(1. / self.beta.data.cpu().numpy()[idx])
        self.beta.data = self.beta.data * s_ratio
        self.s.data = self.s.data * s_ratio

    def consolidate_result(self):
        beta_res = self.beta.data.cpu().numpy()
        return beta_res


class GBTLGamma(GBTL):
    def __init__(self, data_pack, config):
        super(GBTLGamma, self).__init__(data_pack, config)
        self.gamma = torch.tensor(np.zeros(self.n_judges), device=self.device, dtype=self.dtype, requires_grad=True)
        self.parameters.append(self.gamma)
        self.named_parameters['gamma'] = self.gamma

    def initialize(self):
        self.get_initializer()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype)
        self.gamma.data += torch.tensor(1. / (self.beta_init + 10e-8), device=self.device, dtype=self.dtype)

    def compute_likelihood(self):
        sr_j = self.replicator * self.s  # each column is the same value
        sr_i = torch.transpose(sr_j, 1, 0)
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.view((1,) + si_minus_sj.shape)
        iv = self.gamma.view(self.gamma.shape + (1, 1))
        qu = ex * iv
        mask = (qu > 23).double()
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = torch.log(torch.exp(q_exact) + 1) + q_approx
        pr = - torch.sum(self.count_mat * lg)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_sparse(self):
        pr = torch.tensor(0, dtype=self.dtype).to(self.device)
        for item, cnt in self.data_cnt.items():
            i, j, k = item
            pr += - cnt * torch.log(torch.exp((self.s[j] - self.s[i]) * self.gamma[k]) + 1)
        self.pr = -pr / self.n_pairs

    def compute_likelihood_np(self, s, gamma):
        sr_j = self.replicator.cpu().numpy() * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        ex = si_minus_sj.reshape((1,) + si_minus_sj.shape)
        iv = gamma.reshape(self.gamma.shape + (1, 1))
        qu = ex * iv
        mask = (qu > 23)
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = np.log(np.exp(q_exact) + 1) + q_approx
        pr = - np.sum(self.count_mat.cpu().numpy() * lg)
        return -pr / self.n_pairs

    def post_process(self):
        self.s.data -= torch.min(self.s.data)
        idx = 0
        s_ratio = np.abs(1. / self.gamma.data.cpu().numpy()[idx])
        self.gamma.data = self.gamma.data * s_ratio
        self.s.data = self.s.data / s_ratio

    # def compute_gradient(self):
    #     grad_b = torch.sum(torch.sum(self.count_mat * ex / (torch.exp(-qu) + 1), dim=-1), dim=-1) / n_pairs

    def consolidate_result(self):
        beta_res = 1. / self.gamma.data.cpu().numpy()
        return beta_res
