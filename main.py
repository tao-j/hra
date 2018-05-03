from addict import Dict

from algo_func import *
from data_helper import *
from eval_paint import *

import pandas as pd
import pickle as pkl
import os 

def run(data_name, algo_name, ds):
    dn = data_name.split('-')

    if dn[0] == 'po':
        bgf = 'power'
        sb = float(dn[1][1:])
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
    if an[0] == 'gbtl':
        algo = 'individual'
    if an[0] == 'btl':
        algo = 'simple'
    
    if 'spectral' in an[1]:
        init = 'spectral'
    else:
        init = 'random'
        
    ob = 'beta' in an[1]
    opt = 'mle' in an[2]
    
    algo_kwarg = {
        'init_seed': ds, 'init_method': init,
        'override_beta': ob, 'max_iter': 800, 'lr': 1e-3,              
        'opt': opt, 'opt_func': 'SGD', 'opt_stablizer': 'default',
        'debug': 0, 'verbose': 0, 'algo': algo, 
    }
#         print(algo_kwarg)        
    algo_kwarg['data_pack'] = data_pack
    
    res_pack = train_func_torchy(**algo_kwarg)

    cb = {**data_kwarg, **algo_kwarg, **res_pack}
    return cb

data_names = [
#     'po-b10-j4-i5-p800',
    'po-b10-j4-i10-p800',
    'po-b10-j4-i15-p800',
    'po-b10-j8-i5-p800',
    'po-b10-j8-i10-p800',
    'po-b10-j8-i15-p800',
    'po-b5-j4-i10-p800',
    'po-b2-j4-i10-p800',
    'po-b10-j16-i10-p800',
]

algo_names = [
    'btl-spectral-do',
    'btl-spectral-mle',
    'btl-random-mle',
    'gbtl-spectral_all-do',
    'gbtl-spectral_all-mle',
    'gbtl-spectral_s_random_b-mle',
    'gbtl-random_all-mle',
]

storage = Dict()
acc = Dict()
sf = open('storage.pkl', 'wb')
af = open('acc.pkl', 'wb')
# key: dn + an
# val: [result]

data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62]

for data_name in data_names:
    for algo_name in algo_names:
        st = data_name + '+' + algo_name
        if st not in storage:
            storage[st] = []
        if st not in acc:
            acc[st] = []
        for idx, ds in enumerate(data_seeds[:8]):
            all_pack = run(data_name, algo_name, ds)
            all_pack['data_pack'].pop('data')
            storage[st].append(all_pack)
            score = get_eval(Dict(all_pack))
            acc[st].append(score)
            print(st, score, '\n--------------------\n')
        
        sf = open('storage.pkl', 'wb')
        af = open('acc.pkl', 'wb')
        pkl.dump(acc, af)
        pkl.dump(storage, sf)
        sf.close()
        af.close()
