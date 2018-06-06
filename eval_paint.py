import multiprocessing as mp
import numpy as np
import scipy

from algo_func import *
from data_helper import *

# https://gist.github.com/bwhite/3726239
def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()


def get_eval(all_pack):
    s_pred = all_pack.res_s
    rank_pred = np.argsort(s_pred)
    rank_orig = np.argsort(all_pack.data_pack.s)
    if np.any(np.isnan(np.array(s_pred))):
        acc = np.nan
    else:
        acc = acc_func(rank_pred, rank_orig)
    
#     if result_pack is not None:
#         result_pack.append(
#             [np.linalg.norm(res_s[rank]-s_true),
#              np.linalg.norm(( (res_eps**2-eps_true**2)/eps_true**2))
#             ])
    return acc

def acc_func(pred, src=None):
    if src is None:
        src = np.arange(len(pred))
    return scipy.stats.spearmanr(pred, src)[0]


def run(data_name, algo_name, ds):
    dn = data_name.split('-')

    if dn[0] == 'po':
        bgf = 'power'
        sb = float(dn[1][1:])
    if dn[0] == 'be':
        bgf = 'beta'
        sb = list(map(float, dn[1][1:].split(',')))
    if dn[0] == 'rd':
        bgf = 'shrink'
        sb = float(dn[1][1:])
    if dn[0] == 'ne':
        bgf = 'negative'
        sb = float(dn[1][1:])
    if dn[0] == 'ma':
        bgf = 'manual'
        sb = list(map(float, dn[1][1:].split(',')))
    nj = int(dn[2][1:])
    ni = int(dn[3][1:])
    np = int(dn[4][1:])
    data_kwarg = {
        'data_seed': ds,
        'shrink_b': sb,
        'beta_gen_func': bgf,
        'n_items': ni,
        'n_judges': nj,
        'n_pairs': np,
        'visualization': 0,
    }
#         print(data_kwarg)
    data_pack = generate_data(**data_kwarg)

    
    an = algo_name.split('-')
    # algo_map = {
    # TODO: make dict
    # }
    if an[0] == 'gbtl':
        algo = 'individual'
    if an[0] == 'btl':
        algo = 'simple'
    if an[0] == 'gbtlinv':
        algo = 'inverse'
    if an[0] == 'gbtlneg':
        algo = 'negative'
    
    if 'spectral' in an[1]:
        init = 'spectral'
    elif 'disturb' in an[1]:
        init = 'ground_truth_disturb'
    else:
        init = 'random'
        
    ob = 'random_b' in an[1]
    opt = 'mle' in an[2]
    
    algo_kwarg = {
        'init_seed': ds, 'init_method': init,
        'override_beta': ob, 'max_iter': 800, 'lr': 1e-3, 'lr_decay': False,             
        'opt': opt, 'opt_func': 'SGD', 'opt_stablizer': 'default', 'opt_sparse': False,
        'debug': 1, 'verbose': 0, 'algo': algo, 
    }
#         print(algo_kwarg)        
    algo_kwarg['data_pack'] = data_pack
    
    res_pack = train_func_torchy(**algo_kwarg)

    cb = {**data_kwarg, **algo_kwarg, **res_pack}
    return cb


def async_train(fp, args_ls):
    
    def f(q, fp, args_l):
        print(args_l[1])
        res = fp(*args_l)
        q.put(res)
    result_err = []

    q = mp.Queue()
    ps = []
    for args_l in args_ls:
    #     rets = pool.apply_async(f, (q, np.arange(pid)))
        p = mp.Process(target=f, args=(q, fp, args_l))
        ps.append(p)
        p.start()

    for p in ps:
        result_err.append(q.get())

    for p in ps:
        p.join()
    
    return result_err
