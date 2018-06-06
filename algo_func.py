import torch
import numpy as np
import scipy
import scipy.stats
import time
from torch.autograd import Variable

#from bokeh.plotting import *
import matplotlib.pyplot as plt

from addict import Dict

def calc_transition(c):
    import copy
    c = copy.deepcopy(c)
    m_size = c.shape[0]

    for i in range(m_size):
        for j in range(i + 1, m_size):
            if (c[i][j] + c[j][i]) != 0:
                c[j][i] = c[j][i] / (c[i][j] + c[j][i])
                c[i][j] = 1 - c[j][i]
            else:
                c[i][j] = 0.
                c[j][i] = 0.

    outer_degree = (c > 0).sum(axis=1)

    d = (c.T / (outer_degree + 1e-10)).T
    row_sum = d.sum(axis=1)
    d = d + np.eye(m_size) * (1 - row_sum)
    return d


def stationary_distribution(d):
    
    m_size = d.shape[0]
    
    e = d - np.eye(m_size)
    e = e.T
    e[-1] = np.ones(m_size)
    
    y = np.zeros(m_size)
    y[-1] = 1
    
    res = np.linalg.solve(e, y)
    return res


def calc_s_beta(data_mat):
    
    # note the latter two dimension is also flipped
    c = data_mat.transpose(0, 2, 1)
    
    # ------------ use all judge info to calc
    mixed = c.sum(axis=0)
    p = calc_transition(mixed)
    sp = np.log(stationary_distribution(p))
#     sp -= np.min(sdb)
#     sp /= sdb.sum()
    
    # ------------ use each judge to calc
    # e^(s/beta) = esdb
    # s/beta = sdb
    sdb_mat = np.zeros((c.shape[0], c.shape[1]))
    betas = np.zeros(c.shape[0])
    betas[0] = 1
    
    for c_i in range(c.shape[0]):
        d = c[c_i]
        p = calc_transition(d)
        esdb = stationary_distribution(p)
        sdb = np.log(esdb + 1e-13)
#         TODO: test what would happen if with below transformation to make everyhing >0
#         sdb -= np.min(sdb)
#         sdb /= sdb.sum()
        sdb_mat[c_i] = sdb
        
        # TODO: any other ratio may work?
        betas[c_i] = np.abs(np.mean(sdb_mat[0] / sdb_mat[c_i]) + 1e-13) # * b_mat[0]
    
    s = np.mean((sdb_mat.T / betas).T, axis=0)
    
    return sp, s, betas


def train_func_torchy(data_pack, init_seed=None, init_method='random', ground_truth_disturb=1e-3, 
                      override_beta=False, max_iter=500, lr=1e-3, lr_decay=True,
                      opt=True, opt_func='SGD', opt_stablizer='default', opt_sparse=False,
                      debug=False, verbose=False, algo='simple', 
                      result_pack=None, GPU=True):
    dtype = torch.float
    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    data = data_pack.data
    n_items = data_pack.n_items
    n_pairs = data_pack.n_pairs
    n_judges = data_pack.n_judges
    s_true = data_pack.s
    beta_true = np.array(data_pack.beta)
    eps_true = np.sqrt(beta_true)
    
    if opt_func == 'SGD':
        opt_func = torch.optim.SGD
    elif opt_func == 'Adam':
        opt_func = torch.optim.Adam
    else:
        assert(False, 'specificed optimization method is incorrect.')

    if not init_seed:
        init_seed = int(time.time() * 10e7) % 2**32
    torch.manual_seed(init_seed)

    data_cnt = {}
    data_mat = np.zeros((n_judges, n_items, n_items))
    for i, j, k in data:
        data_mat[k][i][j] += 1
        if (i, j, k) in data_cnt:
            data_cnt[(i, j, k)] += 1
        else:
            data_cnt[(i, j, k)] = 1

    # --------------- initialization -----------------
    if init_method == 'random':
        s_init = np.random.random(n_items)
        s_init -= np.min(s_init)
        s_init /= np.sum(s_init)
        beta_init = np.random.random(n_judges) * 0.05
        eps_init = np.random.random(n_judges) * 0.05
    
    if init_method == 'spectral':
        s_init_tout, s_init_individual, beta_init = calc_s_beta(data_mat)
        eps_init = np.sqrt(beta_init)
        if algo == 'simple':
            s_init = s_init_tout
        if algo == 'individual':
            s_init = s_init_individual
        if algo == 'negative':
            s_init = s_init_individual
        if algo == 'inverse':
            s_init = s_init_individual
        if override_beta:
            beta_init = np.random.random(n_judges) * 0.05
            eps_init = np.random.random(n_judges) * 0.05
        
    if init_method == 'ground_truth_disturb':
        s_init = s_true + np.random.normal(0, ground_truth_disturb, size=n_items)
        beta_init = beta_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
        eps_init = eps_true + np.random.normal(0, ground_truth_disturb, size=n_judges)
    # TODO: die gracefully when no init_method matches
        
    s = torch.tensor(s_init, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(beta_init, device=device, dtype=dtype, requires_grad=True)
    eps = torch.tensor(eps_init, device=device, dtype=dtype, requires_grad=True)
    inv = torch.tensor(1. / (beta_init + 10e-8), device=device, dtype=dtype, requires_grad=True)
    if debug:
        print('initial ranking result', np.argsort(s_init))
        print('initial: s, beta', s.data.cpu().numpy(), eps.data.cpu().numpy()**2)


    # --------------- training / optimization -----------------
    if algo == 'simple':
        params = [s]
    elif algo == 'individual':
        params = [s, eps]
    elif algo == 'inverse':
        params = [s, inv]
    elif algo == 'negative':
        params = [s, beta]
    
    p_list = []
    p_noreg_list = []
    s_list = []
    if opt:
        # NOTE: average gradient manually 
        optimizer = opt_func(params, lr=lr/n_pairs)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, 400)

        data_mat = torch.tensor(data_mat, device=device, dtype=dtype)

        for iter_num in range(max_iter):
            # TODO: minibatch training
            # np.random.shuffle(data)

            if lr_decay:
                sched.step()

            if debug:
                # print("iter ", iter_num, '\r')
                if verbose:
                    print('s', s.data, 'eps', eps.data)

            if opt_sparse:
                p = torch.tensor(0, dtype=dtype).to(device)
                p_noreg = torch.tensor(0, dtype=dtype).to(device)
                for item, cnt in data_cnt.items():
                    i, j, k = item
                    if algo == 'simple':
                        p += - cnt * torch.log(torch.exp((s[j] - s[i])) + 1)
                    elif algo == 'individual':
                        p += - cnt * torch.log(torch.exp((s[j] - s[i]) / eps[k] / eps[k]) + 1)
                    elif algo == 'negative':
                        p += - cnt * torch.log(torch.exp((s[j] - s[i]) / beta[k]) + 1)
                    elif algo == 'inverse':
                        p += - cnt * torch.log(torch.exp((s[j] - s[i]) * inv[k]) + 1)
            else:
                replicator = torch.tensor(torch.FloatTensor(np.ones([n_items, n_items])).to(device))
                sr_m = replicator * s
                sr_t = torch.transpose(sr_m, 1, 0)
                sd_m = sr_m - sr_t
                if algo == 'simple':
                    p = - torch.sum(torch.sum(data_mat, dim=0) * torch.log(torch.exp(sd_m) + 1))
                elif algo == 'individual':
                    ex = sd_m.view((1,) + sd_m.shape)
                    ep = (eps * eps).view(eps.shape + (1, 1))
                    lg = torch.log(torch.exp(ex / ep) + 1)
                    p = - torch.sum(data_mat * lg)
                elif algo == 'negative':
                    ex = sd_m.view((1,) + sd_m.shape)
                    ep = beta.view(beta.shape + (1, 1))
                    lg = torch.log(torch.exp(ex / ep) + 1)
                    p = - torch.sum(data_mat * lg)
                elif algo == 'inverse':
                    ex = sd_m.view((1,) + sd_m.shape)
                    iv = inv.view(inv.shape + (1, 1))
                    lg = torch.log(torch.exp(ex * iv) + 1)
                    p = - torch.sum(data_mat * lg)

            # ----- regularization
            p_noreg_list.append(np.array(p.data))
#             if algo == 'simple':
#                 p += - s.pow(2).sum() / 1000.
#             elif algo == 'individual':
#                 p += - s.pow(2).sum() / 1000. - eps.pow(2).sum() / 1000.
            p_list.append(np.array(p.data))

            optimizer.zero_grad()
            p = -p
            p.backward()
            if debug and verbose:
                print('s.grad', s.grad)
                print('eps.grad', eps.grad)
                print('inv.grad', inv.grad)
                print('beta.grad', beta.grad)
            optimizer.step()

            # shift and scale after optimization
            if debug and verbose:
                print('shift by', np.min(s.data.cpu().numpy()))
            s.data -= torch.min(s.data)

            if opt_stablizer == 'default':
                s_ratio = torch.sum(s.data)
#                 print('s_ratio', s_ratio.device, s_ratio.type())
                if debug and verbose:
                    print('scale by', 1. / s_ratio)
                s.data = s.data / s_ratio
                eps.data = eps.data / s_ratio.pow(0.5)
                beta.data = beta.data / s_ratio
                inv.data = inv.data * s_ratio
            elif opt_stablizer == 'decouple':
                pass
            
            s_list.append(np.sum((s.data.cpu().numpy() - s_true)**2))
            if debug and verbose:
                print('-------iter--------')
                
        if debug:
            plt.plot(p_list[1:])
#             ax.set_yscale('log')
            plt.show()
            plt.plot(s_list[1:])
            plt.show()
#             plt.plot(p_noreg_list)
#             plt.show()
    
    # ----------- summary -------------
    res_s = s.data.cpu().numpy()
    if algo == 'simple':
        res_beta = eps.data.cpu().numpy()
    if algo == 'individual':
        res_beta = eps.data.cpu().numpy() ** 2
    if algo == 'negative':
        res_beta = beta.data.cpu().numpy()
    if algo == 'inverse':
        res_beta = 1. / inv.data.cpu().numpy()
    rank = np.argsort(res_s)

    if algo == 'simple':
        print('predicted rank btl', rank)
    elif algo == 'individual':
        print('predicted rank gbtl', rank)
    elif algo == 'individual':
        print('predicted rank gbtl-inv', rank)

    res_pack = Dict()
    res_pack.res_s = res_s
    res_pack.res_beta = res_beta
    print(res_pack)
    res_pack.p_list = np.array(p_list)
    res_pack.s_list = np.array(s_list)
    return res_pack
