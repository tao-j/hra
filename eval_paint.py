import multiprocessing as mp
import numpy as np
import scipy

from algo_func import *
from data_helper import *

def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()


def acc_func(pred, src=None):
    if src is None:
        src = np.arange(len(pred))
    return scipy.stats.spearmanr(pred, src)[0]


def get_eval(all_pack):
    s_pred = all_pack.res_s
    rank_pred = np.argsort(s_pred)
    rank_orig = np.argsort(all_pack.data_pack.s)
    if np.any(np.isnan(np.array(s_pred))):
        acc = np.nan
    else:
        acc = acc_func(rank_pred, rank_orig)
    
    return acc


def gen_data(data_name, seed):
    dn = data_name.split('-')

    dn1_split = list(map(float, dn[1][1:].split(',')))
    dn_lookup = {
        'po': ('power', dn1_split[0]),
        'be': ('beta', dn1_split),
        'rd': ('shrink', dn1_split[0]),
        'ne': ('negative', dn1_split[0]),
        'ma': ('manual', dn1_split)
    }
    bgf, sb = dn_lookup[dn[0]]
    
    nj = int(dn[2][1:])
    ni = int(dn[3][1:])
    np = int(dn[4][1:])
    data_kwarg = {
        'data_seed': seed,
        'shrink_b': sb,
        'beta_gen_func': bgf,
        'n_items': ni,
        'n_judges': nj,
        'n_pairs': np,
        'visualization': 1,
    }
    data_pack = generate_data(**data_kwarg)
    return data_pack, data_kwarg


def run_algo(data_pack, data_kwarg, algo_name, seed):
    
    an = algo_name.split('-')
    algo_lookup = {
        'gbtl': 'individual',
        'btl': 'simple',
        'gbtlinv': 'inverse',
        'gbtlneg': 'negative'
    }
    algo = algo_lookup[an[0]]
    
    if 'spectral' in an[1]:
        init = 'spectral'
    elif 'disturb' in an[1]:
        init = 'ground_truth_disturb'
    else:
        init = 'random'
        
    ob = 'random_b' in an[1]
    opt = 'mle' in an[2]
    
    algo_kwarg = {
        'init_seed': seed, 'init_method': init,
        'override_beta': ob, 'max_iter': 2000, 'lr': 1e-4, 'lr_decay': False,
        'opt': opt, 'opt_func': 'SGD', 'opt_stablizer': 'default', 'opt_sparse': False,
        'debug': 1, 'verbose': 0, 'algo': algo,
    }
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
        # rets = pool.apply_async(f, (q, np.arange(pid)))
        p = mp.Process(target=f, args=(q, fp, args_l))
        ps.append(p)
        p.start()

    for p in ps:
        result_err.append(q.get())

    for p in ps:
        p.join()
    
    return result_err
