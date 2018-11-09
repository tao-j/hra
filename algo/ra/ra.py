import torch
import time
import numpy as np
from addict import Dict


class RankAggregation:
    def __init__(self, data_pack, config):
        self.config = config
        self.data_pack = data_pack
        if self.data_pack.data_cnt:
            self.data_cnt = self.data_pack.data_cnt

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
        from algo.init import BTL
        if not self.config.fix_s or self.config.algo == BTL:
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


def make_estimation(data_pack, config):
    from algo.ra.btl import BTLNaive
    from algo.ra.gbtl import GBTLGamma, GBTLBeta, GBTLEpsilon
    from algo.init import BTL, GBTLEPSILON, GBTLBETA, GBTLGAMMA

    if config.algo == BTL:
        algorithm = BTLNaive(data_pack, config)
    elif config.algo == GBTLEPSILON:
        algorithm = GBTLEpsilon(data_pack, config)
    elif config.algo == GBTLBETA:
        algorithm = GBTLBeta(data_pack, config)
    elif config.algo == GBTLGAMMA:
        algorithm = GBTLGamma(data_pack, config)
    else:
        raise NotImplementedError

    try:
        algorithm.initialize()
    except np.linalg.linalg.LinAlgError:
        # this_algo_acc.append(0.)
        print("Cannot solve equation for initialization.")
        res_pack = Dict()
        res_pack.init_fail = True
        return res_pack

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
    # TODO: determine sign of beta
    # print(np.sum(beta_est > 0), 'sum')
    # if np.sum(beta_est > 0) < np.sum(beta_est < 0):
    #     beta_est = -beta_est
    #     s_est = -s_est
    #     print('reversed')

    res_pack = Dict()
    res_pack.s_est = s_est
    res_pack.beta_est = beta_est

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
