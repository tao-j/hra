from algo.init.spectral import IndividualInitializer
from algo.init.estimated import MLInitializer, MomentInitializer
from algo.init.init import RandomInitializer, GroundTruthInitializer
from algo.init import *
from algo.ra.ra import RankAggregation

import torch
import numpy as np
import scipy.optimize as op


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

    def compute_likelihood_count_mat(self, s, eps):
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

    def compute_likelihood_count_mat(self, s, beta):
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
        self.gamma = np.zeros(self.n_judges)
        self.parameters.append(self.gamma)
        self.named_parameters['gamma'] = self.gamma

        self.opt_iter = 0

    def initialize(self):
        self.get_initializer()
        self.s += self.s_init
        self.gamma += 1. / (self.beta_init + 10e-8)

    def compute_likelihood_count_mat(self, s, gamma=None):
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

    # def compute_likelihood_sparse(self):
    #     pr = 0.
    #     # s[i] is winner
    #     for item, cnt in self.data_cnt.items():
    #         i, j, k = item
    #         pr += - cnt * np.log(np.exp((self.s[j] - self.s[i]) * self.gamma[k]) + 1)
    #     self.pr = -pr / self.n_pairs
    def compute_likelihood_np_exact(self, s, gamma):
        # s[j] is winner
        sr_j = self.replicator * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        # gamma = np.ones(self.n_judges)

        ex = si_minus_sj.reshape((1,) + si_minus_sj.shape)
        iv = gamma.reshape(self.gamma.shape + (1, 1))
        qu = ex * iv
        lg = np.log(np.exp(qu) + 1)
        pr = - np.sum(self.count_mat * lg)
        return -pr / self.n_pairs

    def compute_likelihood_np_approx(self, s, gamma):
        # s[j] is winner
        sr_j = self.replicator * s  # each column is the same value
        sr_i = sr_j.T
        si_minus_sj = sr_i - sr_j

        # gamma = np.ones(self.n_judges)

        ex = si_minus_sj.reshape((1,) + si_minus_sj.shape)
        iv = gamma.reshape(self.gamma.shape + (1, 1))
        qu = ex * iv
        mask = (qu > 23)
        q_approx = mask * qu
        q_exact = qu - q_approx
        lg = np.log(np.exp(q_exact) + 1) + q_approx
        pr = - np.sum(self.count_mat * lg)
        return -pr / self.n_pairs

    def compute_likelihood_np_sparse(self, s, gamma):
        pr = 0.
        if self.data_cnt:
            for item, cnt in self.data_cnt.items():
                # s[i] is winner
                i, j, k = item
                # if cnt >= 2:
                #     print(item, cnt)
                v = (1. + np.exp((s[j] - s[i]) * gamma[k]))
                pr += - cnt * np.log(v)
        return -pr /self.n_pairs

    def compute_likelihood_np_s(self, s):
        # return self.compute_likelihood_np_approx(s, self.gamma)
        # return self.compute_likelihood_np(s, self.gamma)
        return self.compute_likelihood_np_sparse(s, self.gamma)

    def compute_likelihood_np_gamma(self, gamma):
        return self.compute_likelihood_np_sparse(self.s, gamma)

    def compute_gradient_s(self, s):
        gamma = self.gamma
        grad = np.zeros(self.n_items)
        if self.data_cnt:
            for item, cnt in self.data_cnt.items():
                # s[i] is winner
                i, j, k = item
                v = gamma[k] / (1. + np.exp((s[i] - s[j]) * gamma[k]))
                grad[i] -= cnt * v
                grad[j] += cnt * v
        return grad

    def compute_gradient_hessian_gamma(self, gamma):
        grad = np.zeros(self.n_judges)
        h = np.zeros(self.n_judges)
        s = self.s
        if self.data_cnt:
            for item, cnt in self.data_cnt.items():
                if not cnt:
                    continue
                # s[i] is winner
                i, j, k = item
                vp = (1 + np.exp(gamma[k] * (s[i] - s[j])))
                vi = (1 + np.exp(gamma[k] * (s[j] - s[i])))
                grad[k] += cnt * (s[j] - s[i]) / vp
                h[k] += cnt * np.power(s[j] - s[i], 2.) / vp / vi
                # h[k] += cnt * np.power(s[j] - s[i], 2.) * np.exp(gamma[k] * (s[i] + s[j])) / \
                #         np.power(np.exp(gamma[k] * s[i]) + np.exp(gamma[k] * s[j]), 2.)
        return grad, np.diag(h)
        # return grad, h#np.diag(h)

    def compute_gradient_gamma(self, gamma):
        return self.compute_gradient_hessian_gamma(gamma)[0]

    def compute_hessian_gamma(self, gamma):
        return self.compute_gradient_hessian_gamma(gamma)[1]

    def optimization_step(self):
        res_gamma = op.minimize(fun=self.compute_likelihood_np_gamma,
                    x0=self.gamma,
                    jac=self.compute_gradient_gamma,
                                # )
                    method='Newton-CG',
                    hess=self.compute_hessian_gamma,
                                )
        if not res_gamma.success:
            print('----alter, step for gamma')
            print(res_gamma)
            pass
        self.gamma = res_gamma.x

        # new_gamma = np.ones(self.n_judges)
        # for k in range(self.n_judges):
        #     new_gamma[k] = MLInitializer.BetaMLEstimate(self.s, self.count_mat[k])
        # self.gamma = new_gamma
        '''
        self.s = np.array([0.2535809, 0.03395629, 0.02179904, 0.14150608, 0.15534016,
                           0.22660238, 0.48947107, 0.52969226, 0.78312124, 0.84059461])
        self.gamma = np.ones(self.n_judges)
        print('===================')
        print('likeli gamma=1')
        print(self.compute_likelihood_np_sparse(self.s, self.gamma), 'LL sparse')
        print(self.compute_likelihood_np_approx(self.s, self.gamma), 'LL approx')
        print(self.compute_likelihood_np_exact(self.s, self.gamma), 'LL exact')
        print(self.compute_gradient_s(self.s), 'grad s')
        print(self.compute_gradient_hessian_gamma(self.gamma), 'grad hessian gamms')

        print('likeli gamma=blablah')
        self.gamma = np.array([33, 33, 22, 11, 55, 22, 22, 11, 88])
        print(self.compute_likelihood_np_sparse(self.s, self.gamma), 'LL sparse')
        print(self.compute_likelihood_np_approx(self.s, self.gamma), 'LL approx')
        print(self.compute_likelihood_np_exact(self.s, self.gamma), 'LL exact')
        print(self.compute_gradient_s(self.s), 'grad s')
        print(self.compute_gradient_hessian_gamma(self.gamma), 'grad hessian gamms')
        print('====================')
        if self.opt_iter == 0:
            self.s = np.random.random(self.n_items)
        self.opt_iter += 1
        '''

        res_s = op.minimize(fun=self.compute_likelihood_np_s,
                    x0=self.s,
                    method='L-BFGS-B',
                    jac=self.compute_gradient_s)
        if not res_s.success:
            print('----alter, step for s')
            print(res_s)
            pass
        self.s = res_s.x

        return False

    def post_process(self):
        self.s.data -= torch.min(self.s.data)
        idx = 0
        s_ratio = np.abs(1. / self.gamma.data.cpu().numpy()[idx])
        self.gamma.data = self.gamma.data * s_ratio
        self.s.data = self.s.data / s_ratio

    # def compute_gradient(self):
    #     grad_b = torch.sum(torch.sum(self.count_mat * ex / (torch.exp(-qu) + 1), dim=-1), dim=-1) / n_pairs

    def consolidate_result(self):
        beta_res = 1. / self.gamma
        return beta_res
