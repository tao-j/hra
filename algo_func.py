import torch
import numpy as np
import scipy
import scipy.stats
from torch.autograd import Variable

#from bokeh.plotting import *
import matplotlib.pyplot as plt

dtypeF = torch.FloatTensor
dtypeI = torch.LongTensor

# https://gist.github.com/bwhite/3726239
def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()

def acc_func(pred):
    return scipy.stats.spearmanr(np.arange(len(pred)), pred)


def train_func_torchy(data_pack, init_seed=1362, max_iter=500, lr=1e-3, opt_func=torch.optim.SGD, debug=False, algo='simple', verbose=False, beta_disturb=None):

    data, n_items, n_judges, n_pairs, s_true, betas = data_pack
    eps_true = np.sqrt(betas)

    torch.manual_seed(init_seed)
    s = Variable(torch.randn(n_items).type(dtypeF), requires_grad=True)
    s.data -= s.data[0]
    s.data /= torch.sum(s.data)
    eps = Variable(torch.randn(n_judges).type(dtypeF), requires_grad=True)
    if beta_disturb:
        s.data = torch.FloatTensor(s_true + np.random.normal(0, beta_disturb, size=n_items))
        eps.data = torch.FloatTensor(eps_true + np.random.normal(0, beta_disturb, size=n_judges))

    if debug:
        print('initial: s, beta', s.data.numpy(), eps.data.numpy()**2)

    if algo == 'simple':
        params = [s]
    elif algo == 'individual':
        params = [s, eps]
        
    # average gradient manually 
    optimizer = opt_func(params, lr=lr/n_pairs)

    p_list = []
    p_noreg_list = []
    s_list = []
    for iter_num in range(max_iter):
        # TODO: minibatch training
        # np.random.shuffle(data)
  
        if debug and verbose:
            print("iter ", iter_num, '\n s', s.data, 'eps', eps.data)
#         s.data = s.data / torch.sum(s.data)

        p = 0
        p_noreg = 0
        for i, j, k in data:
            if algo == 'simple':
                p += -torch.log(torch.exp((s[j] - s[i])) + 1)
            elif algo == 'individual':
                p += -torch.log(torch.exp((s[j] - s[i]) /
                 eps[k] / eps[k]) + 1)
            # torch.tanh(eps[k]) / torch.tanh(eps[k])) + 1)

        p_noreg_list.append(np.array(p.data)[0])

        # add regularization
#         if algo == 'simple':
#             p += - s.pow(2).sum() / 1000.
#         elif algo == 'individual':
#             p += - s.pow(2).sum() / 1000. - eps.pow(2).sum() / 1000.
        
        
        p_list.append(np.array(p.data)[0])

        optimizer.zero_grad()
        p = -p
        p.backward()
        
        if debug and verbose:
            print('s.grad', s.grad)
            print('eps.grad', eps.grad)
        optimizer.step()
        
        if debug and verbose:
            print('shift by', s.data[0])
        s.data -= s.data[0]
        
        s_ratio = torch.sum(s.data)
        if debug and verbose:
            print('scale by', 1. / s_ratio)
        s.data = s.data / s_ratio
        eps.data = eps.data / np.sqrt(s_ratio)
      
        s_list.append(np.sum((s.data.numpy() - s_true)**2))
        if debug and verbose:
            print('-------iter--------')

    res_s = s.data.cpu().numpy()
    res_eps = eps.data.cpu().numpy()
    rank = np.argsort(res_s)

    if debug:
        plt.plot(p_list)
        plt.show()
        plt.plot(s_list)
        plt.show()
        # plt.plot(p_noreg_list)
        # plt.show()
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
    
    return rank, acc
