import torch
import numpy as np
import scipy
import scipy.stats
import time
from torch.autograd import Variable

#from bokeh.plotting import *
import matplotlib.pyplot as plt

# GPU
# dtypeF = torch.cuda.FloatTensor
# dtypeI = torch.cuda.LongTensor
dtypeF = torch.FloatTensor
dtypeI = torch.LongTensor

def calc_transition(c):
    import copy
    c = copy.deepcopy(c)
    m_size = c.shape[0]

    for i in range(m_size):
        for j in range(i + 1, m_size):
            assert((c[i][j] + c[j][i]) != 0)
            c[j][i] = c[j][i] / (c[i][j] + c[j][i])
            c[i][j] = 1 - c[j][i]

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
    c = data_mat.transpose(2, 1, 0)
    
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


def train_func_torchy(data_pack, init_seed=None, init_method='random', ground_truth_disturb=None, 
                      max_iter=500, lr=1e-3, lr_decay=True,
                      opt_func=torch.optim.SGD, opt_stablizer='default', 
                      debug=False, verbose=False, algo='simple', 
                      result_pack=None, ):

    data, n_items, n_judges, n_pairs, s_true, betas = data_pack
    eps_true = np.sqrt(betas)

    if not init_seed:
        init_seed = int(time.time() * 10e7) % 2**32
    torch.manual_seed(init_seed)
    
    s = Variable(torch.randn(n_items).type(dtypeF), requires_grad=True)
    s.data -= torch.min(s.data)
    s.data /= torch.sum(s.data)
    eps = Variable(torch.randn(n_judges).type(dtypeF), requires_grad=True)
    
    
    data_cnt = {}
    data_mat = np.zeros((n_items, n_items, n_judges))
    for i, j, k in data:
        data_mat[i][j][k] += 1
        if (i, j, k) in data_cnt:
            data_cnt[(i, j, k)] += 1
        else:
            data_cnt[(i, j, k)] = 1

    # --------------- initialization -----------------
    if init_method == 'spectral':
        sp_init, s_init, beta_init = calc_s_beta(data_mat)
        s.data = torch.FloatTensor(s_init)
        eps.data = torch.FloatTensor(np.sqrt(beta_init))
        
        print('spectral result', np.argsort(s_init))
        print('reinitialize s', s.data.numpy(), s.data.numpy().sum())
        print('reinitialize beta', eps.data.numpy()**2)
        
    if init_method == 'ground_truth_disturb':
        s.data = torch.FloatTensor(s_true + np.random.normal(0, beta_disturb, size=n_items))
        eps.data = torch.FloatTensor(eps_true + np.random.normal(0, beta_disturb, size=n_judges))

    if debug:
        print('initial: s, beta', s.data.cpu().numpy(), eps.data.cpu().numpy()**2)

        
    if algo == 'simple':
        params = [s]
    elif algo == 'individual':
        params = [s, eps]
    
    if algo == 'spectral':
        res_s = s.data.cpu().numpy()    
    else:
        # average gradient manually 
        optimizer = opt_func(params, lr=lr/n_pairs)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, 400)
        # --------------- training -----------------
        p_list = []
        p_noreg_list = []
        s_list = []
        for iter_num in range(max_iter):
            # TODO: minibatch training
            # np.random.shuffle(data)

            if lr_decay:
                sched.step()

            if debug and verbose:
                print("iter ", iter_num, '\n s', s.data, 'eps', eps.data)
#             s.data = s.data / torch.sum(s.data)

            p = 0
            p_noreg = 0

            for item, cnt in data_cnt.items():
                i, j, k = item
                if algo == 'simple':
                    p += - cnt * torch.log(torch.exp((s[j] - s[i])) + 1)
                elif algo == 'individual':
                    p += - cnt * torch.log(torch.exp((s[j] - s[i]) /
                     eps[k] / eps[k]) + 1)
                # torch.tanh(eps[k]) / torch.tanh(eps[k])) + 1)

            p_noreg_list.append(np.array(p.data))

            # ----- regularization
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
            optimizer.step()

            if debug and verbose:
                print('shift by', np.min(s.data.numpy()))
            s.data -= torch.min(s.data)

            if opt_stablizer == 'default':
                s_ratio = torch.sum(s.data)
                if debug and verbose:
                    print('scale by', 1. / s_ratio)
                s.data = s.data / s_ratio
                eps.data = eps.data / np.sqrt(s_ratio)

            if opt_stablizer == 'decouple':
                pass

            s_list.append(np.sum((s.data.numpy() - s_true)**2))
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
    res_eps = eps.data.cpu().numpy()
    rank = np.argsort(res_s)

    if algo == 'simple':
        print('predicted rank without beta', rank)
        print('s', res_s[rank])
    elif algo == 'individual':
        print('predicted rank with beta', rank)
        print('s', res_s[rank])
        print('beta', res_eps**2)
    
    if np.any(np.isnan(np.array(res_s))):
        acc = np.nan
    else:
        acc = acc_func(rank)
    
    if result_pack is not None:
        result_pack.append(
            [np.linalg.norm(res_s[rank]-s_true),
             np.linalg.norm(( (res_eps**2-eps_true**2)/eps_true**2))
            ])
    
    return rank, acc
