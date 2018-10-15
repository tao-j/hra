import multiprocessing as mp
import numpy as np
import scipy

from algo_func import *
from data_helper import *
from addict import Dict


def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()


def acc_func(pred, src=None):
    if src is None:
        src = np.arange(len(pred))
    # return scipy.stats.spearmanr(pred, src)[0]
    return scipy.stats.kendalltau(pred, src)[0]


def get_eval(all_pack):
    s_est = all_pack.s_est
    rank_pred = np.argsort(s_est)
    rank_orig = np.argsort(all_pack.data_pack.s_true)
    if np.any(np.isnan(np.array(s_est))):
        acc = np.nan
    else:
        acc = acc_func(rank_pred, rank_orig)
    return acc


def gen_data(data_name, seed, save_path=None, s_true=None, beta_true=None):
    dn = data_name.split('-')

    dn1_split = list(map(float, dn[1][1:].split(',')))
    dn_lookup = {
        'po': ('power', dn1_split[0]),
        'be': ('beta', dn1_split),
        'ga': ('gamma', dn1_split),
        'rd': ('shrink', dn1_split[0]),
        'ex': ('ex', dn1_split[0]),
        'ne': ('negative', dn1_split[0]),
        'ma': ('manual', dn1_split)
    }
    bgf, sb = dn_lookup[dn[0]]
    
    nj = int(dn[2][1:])
    ni = int(dn[3][1:])
    np = 0
    known_pairs_ratio = 0.
    repeated_comps = 0
    gen_by_pair = False
    if dn[4][0] == 'p':
        np = int(dn[4][1:])
    else:
        # known_pairs_ratio = 0.1  # d/n sampling probability
        # repeated_comps = 32  # k number of repeated comparison
        known_pairs_ratio, repeated_comps = dn[4][0:].split('k')
        known_pairs_ratio = float(known_pairs_ratio)
        repeated_comps = int(repeated_comps)
        gen_by_pair = True
    data_kwarg = {
        'data_seed': seed,
        'shrink_b': sb,
        'beta_gen_func': bgf,
        'n_items': ni,
        'n_judges': nj,
        'n_pairs': np,
        'visualization': 1,
        'save_path': save_path,
        'known_pairs_ratio': known_pairs_ratio,
        'repeated_comps': repeated_comps,
        'gen_by_pair': gen_by_pair,
        's_true': s_true,
        'beta_true': beta_true,
    }
    data_pack = generate_data(**data_kwarg)
    return data_pack, data_kwarg


def run_algo(data_pack, data_kwarg, algo_name, seed):
    an = algo_name.split('-')
    algo_lookup = {
        'btl': 'simple',
        'gbtl': 'individual',
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
    ot = 'gt^transition' in an[1]
    fs = 'fix^s' in an[1]

    if len(an) == 4:
        lr = float(an[3])
    else:
        lr = 1e-3

    if ot:
        print("\x1b[31muse gt_transition\x1b[0m")
    
    algo_kwarg = {
        'init_seed': seed, 'init_method': init,
        'override_beta': ob, 'gt_transition': ot, 'max_iter': 2000, 'lr': lr, 'lr_decay': True,
        'opt': opt, 'opt_func': 'SGD', 'opt_stablizer': 'decouple', 'opt_sparse': False, 'fix_s': fs,
        'debug': 1, 'verbose': 0, 'algo': algo,
    }

    config = Dict(algo_kwarg, data_kwarg)
    config.ground_truth_prob_mat = False
    config.popular_correction = False
    config.grad_method = 'auto'
    config.normalize_gradient = True
    config.GPU = True
    config.linesearch = False
    config.prob_regularization = True
    config.err_const = 10e-23
    res_pack = make_estimation(data_pack, config)

    cb = {**data_kwarg, **algo_kwarg, **res_pack}
    return cb

