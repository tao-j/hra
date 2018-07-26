import torch
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import time
from torch.autograd import Variable
import scipy.sparse.linalg

#from bokeh.plotting import *
import matplotlib.pyplot as plt

from addict import Dict


def calc_transition_ground(s_true, beta_true):
    m_size = beta_true.shape[0]
    n_size = s_true.shape[0]
    c = np.zeros((m_size, n_size, n_size), dtype=np.float)
    for k in range(m_size):
        for i in range(n_size):
            for j in range(n_size):
                if i != j:
                    c[k][j][i] = 1. / (1 + np.exp( (s_true[j] - s_true[i]) / beta_true[k]))
    return c


def calc_transition(c):
    import copy
    c = copy.deepcopy(c)
    n_size = c.shape[0]
    c = c.astype(np.float)
    for i in range(n_size):
        for j in range(i + 1, n_size):
            if (c[i][j] + c[j][i]) != 0:
                c[j][i] = c[j][i] / (c[i][j] + c[j][i])
                c[i][j] = 1. - c[j][i]
            else:
                c[i][j] = 0.
                c[j][i] = 0.

    outer_degree = np.max((c > 0).sum(axis=1))
#     outer_degree = (c > 0).sum(axis=1)
    d = (c.T / (outer_degree + 1e-10)).T
    row_sum = d.sum(axis=1)
    d = d + np.eye(n_size) * (1 - row_sum)
    return d


def stationary_distribution(d):
    
    m_size = d.shape[0]

    e = d - np.eye(m_size)
    e = e.T
    e[-1] = np.ones(m_size)
    
    y = np.zeros(m_size)
    y[-1] = 1
    
#     res = np.linalg.solve(e, y)
    res = np.matmul(np.linalg.inv(e), y)
    return res


def calc_s_beta(data_mat, verbose=False, popular_correction=True):
    # if verbose == False:
    def pppp(*arg):
        return
    print = pppp

    c = data_mat
    # shape is n_judges n_items*n_items^T
    n_judges = c.shape[0]
    n_items = c.shape[1]
    
    # ------------ use all judge info to calc
    mixed = c.sum(axis=0)
    p = calc_transition(mixed)
    sp = np.log(stationary_distribution(p))
    # normalize s for easy comparison
    sp -= np.min(sp)
    sp /= sp.sum()

    if popular_correction:
        c = np.concatenate([mixed[np.newaxis,:], c], axis=0)
        n_judges += 1

    # ------------ use each judge to calc
    # e^(s/beta) = esdb = w (normalized) = u
    # s/beta = sdb = q
    betas = np.zeros(n_judges)
    betas[0] = 1.
    s = np.ones(n_items) * 0.

    print('--------------- step 1 ')
    # step 1:    
    qs = []
    for c_i in range(n_judges):
        d = c[c_i]
        p = calc_transition(d)
        
#         print('imshow')
#         plt.figure()
#         plt.imshow(p)
#         plt.show()
        
        w = stationary_distribution(p)
        print('w before', w)
#         bol = w == 0
#         idx = 0
#         for i in range(bol.shape[0]):
#             if bol[idx] != True:
#                 idx += 1
#             else:
#                 break
#         if idx < bol.shape[0]:
#             p[idx][idx] = np.nan
#             print(p[idx])
#             print(p[:][idx])
        first_flag = True
        for wmin in np.sort(w):
            if wmin > 10e-10:
                print('wmin', wmin)
                break
            first_flag = False
        
        if not first_flag:
            w += wmin
            w = w/w.sum()
            print('w after', w)
        
        A = np.ones(n_items-1) - np.diag(1. / w[1:])        
        A = np.vstack([np.ones((1, s.shape[0] -1)), A])
        print('A', pd.DataFrame(A))
        
        b = np.array([-1.] * s.shape[0])
        b[0] += 1. / w[0]
        print('b', b)
        
        u1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
        print('u by matmul', u1)
        u = scipy.sparse.linalg.lsqr(A, b, show=verbose)[0]
        print('u by lsqr', u)
        
        q = np.hstack([np.log(u)])
        print('q=\hat{s/\\beta}', q)
        print('======================')
        qs.append(q)
        
    qs = np.vstack(qs)
    print('qs', qs)
    
    print('--------------- step 2 ')
    n_items_1 = n_items-1
    # step 2:
    A = np.zeros(((n_items_1)*n_judges, n_items_1+n_judges-1))
    for c_i in range(n_judges):        
            for s_i in range(n_items_1):
                if c_i != 0:
                    A[n_items_1*c_i+s_i][n_items_1 + c_i - 1] = -qs[c_i][s_i]
                A[n_items_1*c_i+s_i][s_i] = 1
    print('A', pd.DataFrame(A))
    
    b = np.ones((n_items_1*n_judges,)) * 0.
    for s_i in range(n_items_1):
        b[s_i] = qs[0][s_i]
    print('b', b)
    
    qq1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
    print('qq by matmul', qq1)
    qq = scipy.sparse.linalg.lsqr(A, b, show=verbose)[0]
    print('qq by lsqr', qq)

    # assignment for return
    for s_i in range(n_items_1):
        s[s_i+1] = qq[s_i]
    for b_i in range(n_judges-1):
        betas[b_i+1] = qq[n_items_1+b_i]

    if popular_correction:
        betas = betas[1:]

    return sp, s, betas


def train_func_torchy(data_pack, init_seed=None, init_method='random', ground_truth_disturb=0, 
                      override_beta=False, gt_transition=False, max_iter=500, lr=1e-3, lr_decay=True,
                      opt=True, opt_func='SGD', opt_stablizer='default', opt_sparse=False, fix_s=False,
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

    if not init_seed:
        init_seed = int(time.time() * 10e7) % 2**32
    torch.manual_seed(init_seed)

    if gt_transition:
        data_mat = calc_transition_ground(s_true, beta_true)
    else:
        data_cnt = {}
        data_mat = np.zeros((n_judges, n_items, n_items))
        for i, j, k in data:
            data_mat[k][j][i] += 1
            if (i, j, k) in data_cnt:
                data_cnt[(i, j, k)] += 1
            else:
                data_cnt[(i, j, k)] = 1
    data_mat = data_mat.astype(np.float)

    # --------------- initialization -----------------
    if init_method == 'random':
        s_init = np.random.random(n_items)
        s_init -= np.min(s_init)
        s_init /= np.sum(s_init)
        beta_init = np.random.random(n_judges) * 0.05
        eps_init = np.random.random(n_judges) * 0.05

    if init_method == 'spectral':
        s_init_tout, s_init_individual, beta_init = calc_s_beta(data_mat, verbose=verbose)
        eps_init = np.sqrt(np.abs(beta_init))
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
        if override_beta:
            beta_init = np.random.random(n_judges) * 0.05
            eps_init = np.random.random(n_judges) * 0.05
        else:
            lr = 1e-5
            
    # TODO: die gracefully when no init_method matches
        
    s = torch.tensor(s_init, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(beta_init, device=device, dtype=dtype, requires_grad=True)
    eps = torch.tensor(eps_init, device=device, dtype=dtype, requires_grad=True)
    inv = torch.tensor(1. / (beta_init + 10e-8), device=device, dtype=dtype, requires_grad=True)
    if debug:
        print('initial ranking result', np.argsort(s_init))
        print('initial: ','\ns', s, '\nbeta', beta, '\neps', eps, '\ninv', inv)

    # --------------- training / optimization -----------------
    params = []
    if algo == 'individual':
        params = [eps]
    elif algo == 'inverse':
        params = [inv]
    elif algo == 'negative':
        params = [beta]

    if not fix_s or algo == 'simple':
        params.append(s)

    p_list = []
    p_noreg_list = []
    s_list = []
    if opt:
        # NOTE: average gradient manually 
        print('lr', lr)
        optimizer = opt_func(params, lr=lr/n_pairs)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, max_iter)

        data_mat = torch.tensor(data_mat, device=device, dtype=dtype)

        for iter_num in range(max_iter):
            # TODO: minibatch training
            # np.random.shuffle(data)

            if lr_decay:
                sched.step()
            
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
                sr_j = replicator * s # each column is the same value
                sr_i = torch.transpose(sr_j, 1, 0)
                si_minus_sj = sr_i - sr_j
                if algo == 'simple':
                    p = - torch.sum(torch.sum(data_mat, dim=0) * torch.log(torch.exp(si_minus_sj) + 1))
                elif algo == 'individual':
                    ex = si_minus_sj.view((1,) + si_minus_sj.shape)
                    ep = (eps * eps).view(eps.shape + (1, 1))
                    qu = ex / ep
                    mask = (qu > 11).float()
                    q_approx = mask * qu
                    q_exact = qu - q_approx
                    lg = torch.log(torch.exp(q_exact) + 1) + q_approx
                    p = - torch.sum(data_mat * lg)
                elif algo == 'negative':
                    ex = si_minus_sj.view((1,) + si_minus_sj.shape)
                    ep = beta.view(beta.shape + (1, 1))
                    qu = ex / ep
                    mask = (qu > 11).float()
                    q_approx = mask * qu
                    q_exact = qu - q_approx
                    lg = torch.log(torch.exp(q_exact) + 1) + q_approx
                    p = - torch.sum(data_mat * lg)
                elif algo == 'inverse':
                    ex = si_minus_sj.view((1,) + si_minus_sj.shape)
                    iv = inv.view(inv.shape + (1, 1))
                    qu = ex * iv
                    mask = (qu > 11).float()
                    q_approx = mask * qu
                    q_exact = qu - q_approx
                    lg = torch.log(torch.exp(q_exact) + 1) + q_approx
                    p = - torch.sum(data_mat * lg)
            if iter_num == 0:
                print('initial likelihood', p)
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
                print('>>> s and beta(or its delegate) gradient values')
                for pa in params:
                    print(pa, pa.grad)
            optimizer.step()

            # shift and scale after optimization
            if debug and verbose:
                print('shift by', np.min(s.data.cpu().numpy()))
            s.data -= torch.min(s.data)

            if not fix_s:
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
                    if algo == 'individual':
                        s_ratio = np.sqrt(1. / eps.data.cpu().numpy()[0])
                        eps.data = eps.data * s_ratio
                        s.data = s.data * s_ratio
                    elif algo == 'negative':
                        s_ratio = 1. / beta.data.cpu().numpy()[0]
                        beta.data = beta.data * s_ratio
                        s.data = s.data * s_ratio
                    elif algo == 'inverse':
                        s_ratio = 1. / inv.data.cpu().numpy()[0]
                        inv.data = inv.data * s_ratio
                        s.data = s.data / s_ratio
                
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
    elif algo == 'inverse':
        print('predicted rank gbtl-inv', rank)

    res_pack = Dict()
    res_pack.res_s = res_s
    res_pack.res_beta = res_beta
    print(res_pack)
    res_pack.p_list = np.array(p_list)
    res_pack.s_list = np.array(s_list)
    return res_pack
