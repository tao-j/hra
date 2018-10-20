import torch
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import scipy.special
import time
import scipy.sparse.linalg
from addict import Dict

def BinSearch(fun,y): # find x s.t fun(x)=y, function must be strictly increasing with range (0,infinity), fun(0) = 0
    if y == 0:
        return 0
    x_hi = 1
    x_low = 0
    while 1:
        f = fun(x_hi)
        if f < y:
            x_low = x_hi
            x_hi *= 2
        elif f==y:
            return x_hi
        else:
            break
    for i in range(10):
        x_mid = (x_low+x_hi)/2
        f_mid = fun(x_mid)
        if y < f_mid:
            x_hi = x_mid
        elif y==f_mid:
            return x_mid
        else:
            x_low = x_mid
    return (x_low+x_hi)/2

def BetaErrorRate(x):
    if x==0:
        return 0
    return -(1/6)*x*(x*np.pi*np.pi-12*np.log(2)+12*x*scipy.special.spence(1+np.exp(-1/x)))

def BetaSearch(y):
    if y<=1/2:
        return BinSearch(BetaErrorRate,y)
    else:
        return -BinSearch(BetaErrorRate,1-y)

def computeLogLikelihood(s,gamma,comps):
    LL = 0
    for comp in comps:
        si = s[comp[0]]
        sj = s[comp[1]]
        prefered = comp[2]
        LL += - np.log(1+np.exp(-(prefered*(si-sj)+(1-prefered)*(sj-si))*gamma))
    return LL

def computeGrad(s,gamma,comps):
    LL = 0
    for comp in comps:
        si = s[comp[0]]
        sj = s[comp[1]]
        prefered = comp[2]
        deltaS = prefered*(si-sj)+(1-prefered)*(sj-si)
        LL += deltaS/(1+np.exp(deltaS*gamma))
    return LL

def compute2ndD(s,gamma,comps):
    LL = 0
    for comp in comps:
        si = s[comp[0]]
        sj = s[comp[1]]
        prefered = comp[2]
        deltaS = prefered*(si-sj)+(1-prefered)*(sj-si)
        LL -= 1/4*deltaS**2/np.cosh(deltaS*gamma/2)
    return LL

def BetaMLEstimate(s,data,init=1):
    comps = []
    for i in range(np.size(data,0)):
        for j in range(np.size(data,1)):
            if data[i][j]:
                comp = [j,i,1]
                comps = comps + [comp * int(data[i][j])]
    gamma_old = 1/init
    # print('--------------------')
    while 1:

        gamma = gamma_old - computeGrad(s,gamma_old,comps)/compute2ndD(s,gamma_old,comps)
        # print(gamma)
        if (gamma-gamma_old)/gamma_old < 0.001:
            break
        gamma_old = gamma
    return 1/gamma


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
        sp = np.log(np.abs(w))
        # normalize s for easy comparison
        sp -= np.min(sp)
        # TODO: removed summation
        sp /= sp.sum()
        return sp


class MomentInitializer(PopulationInitializer):
    def get_initialization_point(self):
        s_est_btl = super(MomentInitializer, self).get_initialization_point()

        betaEstimates = np.zeros(len(self.data_pack.beta_true))
        s_max = max(s_est_btl)
        s_min = min(s_est_btl)
        for k in range(len(self.data_pack.count_mat)):
            data = self.data_pack.count_mat[k]
            incorrect = 0
            total = 0
            for i in range(len(s_est_btl)):
                for j in range(len(s_est_btl)):
                    d = data[i][j]
                    total += d
                    if s_est_btl[i] > s_est_btl[j]:
                        incorrect += d
            ratio = incorrect / total
            betaEstimates[k] = BetaSearch(ratio) * (s_max - s_min)

        return s_est_btl, betaEstimates


class MLInitializer(PopulationInitializer):
    def get_initialization_point(self):
        s_est_btl = super(MLInitializer, self).get_initialization_point()

        betaEstimatesML = np.zeros(len(self.data_pack.beta_true))
        for k in range(len(self.data_pack.count_mat)):
            data = self.data_pack.count_mat[k]
            betaEstimatesML[k] = BetaMLEstimate(s_est_btl, data)

        return s_est_btl, betaEstimatesML


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

    # if init_method == 'random':
    #     pass
    #
    # if init_method == 'spectral':
    #     s_init_tout, s_init_individual, beta_init = calc_s_beta(data_mat, verbose=verbose)
    #     eps_init = np.sqrt(np.abs(beta_init))
    #     if override_beta:
    #         beta_init = np.random.random(n_judges) * 0.05
    #         eps_init = np.random.random(n_judges) * 0.05
    #
    # if init_method == 'ground_truth_disturb':
    #     s_init = s_true + np.random.normal(0, ground_truth_disturb, size=n_items)
    #     beta_init = beta_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
    #     eps_init = eps_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
    #     if override_beta:
    #         beta_init = np.random.random(n_judges) * 0.05
    #         eps_init = np.random.random(n_judges) * 0.05
    #     else:
    #         lr = 1e-5


class RankAggregation:
    def __init__(self, data_pack, config):
        self.config = config
        self.data_pack = data_pack

        self.verbose = False

        self.n_items = data_pack.n_items
        self.n_pairs = data_pack.n_pairs
        self.n_judges = data_pack.n_judges

        if config.init_seed:
            self.init_seed = config.init_seed
        else:
            self.init_seed = int(time.time() * 10e7) % 2 ** 32

        self.s_true = data_pack.s_true
        self.s_init = None

        self.pr = None
        self.parameters = []
        self.named_parameters = {}
        self.optimizer = None
        self.sched = None

        self.pr_list = []
        self.pr_noreg_list = []
        self.s_list = []

        # TODO if config.backend == 'torch':
        self.dtype = torch.double
        if config.GPU:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.s = torch.tensor(np.zeros(self.n_items), device=self.device, dtype=self.dtype, requires_grad=True)
        self.count_mat = torch.tensor(data_pack.count_mat, device=self.device, dtype=self.dtype)
        self.replicator = torch.tensor(np.ones([self.n_items, self.n_items]),
                                               dtype=self.dtype, device=self.device)
        torch.manual_seed(self.init_seed)
        if config.opt_func == 'SGD':
            self.opt_func = torch.optim.SGD
        elif config.opt_func == 'Adam':
            self.opt_func = torch.optim.Adam
        else:
            raise NotImplementedError

    def setup_optimizer(self):
        if not self.config.fix_s or self.config.algo == 'simple':
            self.parameters.append(self.s)
            self.named_parameters['s'] = self.s
        self.optimizer = self.opt_func(self.parameters, lr=self.config.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=500)
        return self.config.opt

    def initialize(self):
        raise NotImplementedError

    def init_print(self):
        print('................ init value ...............')
        for k, v in self.named_parameters.items():
            if k == 'eps':
                print('beta delegated by ' + k, v ** 2)
            elif k == 'gamma':
                print('beta delegated by ' + k, 1. / v)
            else:
                print(k, v)

    def compute_likelihood(self):
        raise NotImplementedError

    def compute_likelihood_np(self, s, beta):
        raise NotImplementedError

    def optimization_step(self):
        # TODO: backend
        self.optimizer.zero_grad()

        if self.config.grad_method == 'auto':
            self.pr.backward()
        else:
            # TODO: manual gradient
            raise NotImplementedError

        if self.config.normalize_gradient and not self.config.linesearch:
            for pa_index, pa in enumerate(self.parameters):
                grad_norm = torch.sqrt(torch.sum(pa.grad.data * pa.grad.data))
                pa.grad.data /= grad_norm

        if self.config.linesearch:
            a = 0.35
            b = 0.8
            t = 1.
            s = self.parameters[1]
            bb = self.parameters[0]

            assert s.shape[0] == self.n_items
            assert bb.shape[0] == self.n_judges

            x = np.hstack([s.data, bb.data])
            dx = np.hstack([s.grad.data, bb.grad.data])
            if np.sqrt(np.sum(dx * dx)) < 10e-5:
                return True

            test_x = x - t * dx
            test_s = test_x[:self.n_items]
            test_bb = test_x[self.n_items:]

            test_p = self.compute_likelihood_np(test_s, test_bb)
            old_p = np.array(self.pr.data)
            # old_p2 = self.compute_likelihood_np(self.s.detach().cpu().numpy(), self.beta.detach().cpu().numpy())
            # print('starting search, initial p: ', old_p, 'test_p', test_p)

            while test_p > old_p - a * t * dx.dot(dx):
                t = b * t
                test_x = x - t * dx
                test_s = test_x[:self.n_items]
                test_bb = test_x[self.n_items:]
                # print('new step:', t, 'test_p: ', test_p)
                # prev_p = test_p
                test_p = self.compute_likelihood_np(test_s, test_bb)
                # if (prev_p == test_p) / prev_p < 0.0001:
                # if prev_p == test_p:
                #     break

            s.data *= 0.
            s.data += torch.tensor(test_s, dtype=self.dtype, device=self.device)
            bb.data *= 0.
            bb.data += torch.tensor(test_bb, dtype=self.dtype, device=self.device)
        else:
            self.optimizer.step()

        if not self.config.linesearch:
            self.sched.step(self.pr)

        if not self.config.fix_s:
            self.post_process()

        num_pr = self.pr.detach().cpu().numpy().tolist()
        assert not isinstance(num_pr, list)
        self.pr_list.append(num_pr)
        self.s_list.append(np.linalg.norm(self.s.data.cpu().numpy() - self.s_true))

        # if self.config.linesearch and len(self.pr_list) > 10 and abs(self.pr_list[-10] - self.pr_list[-1]) / self.pr_list[-10] < 0.001:
        #     return True
        # else:
        return False

    def post_process(self):
        raise NotImplementedError


class BTLNaive(RankAggregation):
    def __init__(self, data_pack, config):
        super(BTLNaive, self).__init__(data_pack, config)
        self.parameters.append(self.s)
        self.named_parameters['s'] = self.s

    def initialize(self):
        init_method = self.config.init_method
        if init_method == 'random':
            initializer = RandomInitializer(self.data_pack, self.config)
        elif init_method == 'spectral':
            initializer = PopulationInitializer(self.data_pack, self.config)
        elif init_method == 'ground_truth_disturb':
            initializer = GroundTruthInitializer(self.data_pack, self.config)
        elif init_method == 'moment':
            initializer = MomentInitializer(self.data_pack, self.config)
        elif init_method == 'ml':
            initializer = MLInitializer(self.data_pack, self.config)
        else:
            raise NotImplementedError

        self.s_init = initializer.get_initialization_point()
        self.s.data += torch.tensor(self.s_init, device=self.device, dtype=self.dtype)

    def compute_likelihood(self):
        sr_j = self.replicator * self.s  # each column is the same value
        sr_i = torch.transpose(sr_j, 1, 0)
        si_minus_sj = sr_i - sr_j

        pr = - torch.sum(torch.sum(self.count_mat, dim=0) * torch.log(torch.exp(si_minus_sj) + 1))
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


class GBTL(RankAggregation):
    def __init__(self, data_pack, config):
        super(GBTL, self).__init__(data_pack, config)
        self.beta_init = None
        self.beta_true = data_pack.beta_true
        self.popular_correction=self.config.popular_correction

    def get_initializer(self):
        init_method = self.config.init_method
        if init_method == 'random':
            initializer = RandomInitializer(self.data_pack, self.config)
        elif init_method == 'spectral':
            initializer = IndividualInitializer(self.data_pack, self.config)
        elif init_method == 'ground_truth_disturb':
            initializer = GroundTruthInitializer(self.data_pack, self.config)
        elif init_method == 'moment':
            initializer = MomentInitializer(self.data_pack, self.config)
        elif init_method == 'ml':
            initializer = MLInitializer(self.data_pack, self.config)
        else:
            raise NotImplementedError

        self.s_init, self.beta_init =\
            initializer.get_initialization_point()


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


def make_estimation(data_pack, config):
    if config.algo == 'simple':
        algorithm = BTLNaive(data_pack, config)
    elif config.algo == 'individual':
        algorithm = GBTLEpsilon(data_pack, config)
    elif config.algo == 'negative':
        algorithm = GBTLBeta(data_pack, config)
    elif config.algo == 'inverse':
        algorithm = GBTLGamma(data_pack, config)
    else:
        raise NotImplementedError

    algorithm.initialize()
    require_opt = algorithm.setup_optimizer()
    algorithm.init_print()

    if require_opt:
        for i in range(config.max_iter):
            algorithm.compute_likelihood()
            if algorithm.optimization_step():
                print('line search stop condition met -------')
                break

    beta_est = algorithm.consolidate_result()
    s_est = algorithm.s.data.cpu().numpy()
    # print(np.sum(beta_est > 0), 'sum')
    if np.sum(beta_est > 0) < np.sum(beta_est < 0):
        beta_est = -beta_est
        s_est = -s_est
        # print('reversed')

    rank = np.argsort(s_est)
    # print(rank)
    res_pack = Dict()
    res_pack.s_est = s_est
    res_pack.beta_est = beta_est
    # print(res_pack)
    res_pack.data_pack = data_pack

    res_pack.pr_list = algorithm.pr_list[0:]
    res_pack.s_list = algorithm.s_list[0:]

    if config.linesearch:
        pr_name = 'line'
    else:
        pr_name = 'sgd'
    if config.init_method == 'disturb':
        pr_name = 'gt'
    res_pack.pr_name = pr_name
    return res_pack
