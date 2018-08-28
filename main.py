from addict import Dict

from algo_func import *
from data_helper import *
from eval_paint import *

import pandas as pd
import pickle as pkl
import os 
import time

data_names = [

'be-b1,1-j8-i100-p800',
'be-b1,1-j8-i100-p8000',
'be-b1,1-j8-i100-p80000',

'be-b1,10-j8-i100-p800',
'be-b1,10-j8-i100-p8000',
'be-b1,10-j8-i100-p80000',
    
'be-b10,100-j8-i100-p800',
'be-b10,100-j8-i100-p8000',
'be-b10,100-j8-i100-p80000',

 'ma-b1.0,0.001-j2-i10-p800',
 'ma-b1.0,0.001,1.0,0.001-j4-i10-p800',
 'ma-b1.0,0.001,1.0,0.001-j4-i10-p400',
 'ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p800',
 'ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p200',

 'ma-b1.0,0.001-j2-i100-p8000',
 'ma-b1.0,0.001,0,0.001-j4-i100-p8000',
 'ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i100-p8000',

 'ma-b1.0,0.005-j2-i10-p800',
 'ma-b1.0,0.005,1.0,0.005-j4-i10-p800',
 'ma-b1.0,0.005,1.0,0.005,1.0,0.005,1.0,0.005-j8-i10-p800',
 'ma-b1.0,0.01-j2-i10-p800',
 'ma-b1.0,0.01,1.0,0.01-j4-i10-p800',
 'ma-b1.0,0.01,1.0,0.01,1.0,0.01,1.0,0.01-j8-i10-p800',
 'ma-b1.0,0.05-j2-i10-p800',
 'ma-b1.0,0.05,1.0,0.05-j4-i10-p800',
 'ma-b1.0,0.05,1.0,0.05,1.0,0.05,1.0,0.05-j8-i10-p800',
  
 'ma-b1.0,0.01,1.0,0.01-j4-i100-p8000',
 'ma-b1.0,0.01,1.0,0.01-j4-i100-p80000',
 'ma-b1.0,0.01,1.0,0.01,1.0,0.01,1.0,0.01-j8-i100-p8000',
 'ma-b1.0,0.01,1.0,0.01,1.0,0.01,1.0,0.01-j8-i100-p80000',
    
 'ma-b1.0,0.1,1.0,0.1-j4-i100-p8000',
 'ma-b1.0,0.1,1.0,0.1-j4-i100-p80000',
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000',
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p80000',
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p4000',
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p40000',
    
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p8000',
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p4000',

#  'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p80000',
#  'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p40000',
    
#  'ma-b1.0,0.1-j2-i10-p800',
#  'ma-b1.0,0.1,1.0,0.1-j4-i10-p800',
#  'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i10-p800',
#  'ma-b1.0,0.5-j2-i10-p800',
#  'ma-b1.0,0.5,1.0,0.5-j4-i10-p800',
#  'ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i10-p800',
    
 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000',
#  'ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i100-p8000',
    
 'po-b10-j4-i10-p800',
 'po-b10-j8-i10-p800',
 'po-b10-j16-i10-p800',
 'po-b10-j4-i100-p8000',
 'po-b10-j8-i100-p8000',
 'po-b10-j16-i100-p8000',
    
 'po-b10-j3-i15-p800',
 'po-b10-j4-i10-p800',
 'po-b10-j4-i5-p800',
 'po-b10-j8-i10-p800',
 'po-b10-j8-i15-p800',
 'po-b10-j8-i5-p800',
 'po-b2-j4-i10-p800',
 'po-b5-j3-i15-p800',
 'po-b5-j4-i10-p800',
'po-b2-j4-i100-p80000',
'be-b5,20-j4-i15-p1600',

 'po-b5-j4-i32-p100',
 'po-b5-j4-i32-p200',
 'po-b5-j4-i32-p400',
 'po-b7-j4-i10-p800',

 'rd-b08-j10-i10-p800',
 'rd-b16-j10-i10-p800',
 'rd-b32-j10-i10-p800',

 'be-b10,100-j4-i100-p800',
 'be-b10,100-j4-i100-p800',
 'be-b10,100-j4-i15-p200',
 'be-b10,100-j8-i15-p400',
 'be-b2,10-j4-i15-p800',
 'be-b3,20-j4-i15-p800',
 'be-b5,100-j4-i15-p800',

    'ne-b0.005-j4-i16-p800',
    'ne-b0.005-j4-i16-p12800',
    'ne-b0.01-j4-i64-p800',
    'ne-b0.01-j4-i64-p12800',
    'ne-b0.005-j4-i16-p800',

    'be-b1,10-j8-i100-p8000',
'ne-b0.01-j4-i64-p12800',
]

data_names = [
    # 'be-b1,10-j8-i100-p8000',
    # 'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000',
    # 'po-b10-j16-i100-p8000', #nan
    'be-b1,10-j8-i100-p80000',
]


algo_names = [
    'btl-spectral-do',
#     'btl-spectral-mle',
##     'btl-random-mle',
    'gbtl-spectral_all-do',
]

algo_names = []

algo = [
    'gbtl-spectral_all-mle',
    ##     'gbtl-random_all-mle',
    'gbtlneg-spectral_all-mle',  #
    'gbtlinv-spectral_all-mle',  #
    # 'gbtlinv-random_all-mle',  #

    ##     'gbtl-disturb-mle',
    # 'gbtlneg-disturb-mle',  #
    # 'gbtlinv-disturb-mle',  #

    # 'gbtl-disturb_random_b_fix^s-mle',  #
    # 'gbtlneg-disturb_random_b_fix^s-mle',  #
    # 'gbtlinv-disturb_random_b_fix^s-mle',  #
    # 'gbtlinv-disturb_random_b_fix^s-mle',  #
]

algo = ['gbtlneg-spectral_all-mle',]

storage = Dict()
acc = Dict()
# key: dn + an
# val: [result]

# import json
import pprint
# pprint.pprint(json.loads(open('res').read().replace("'", '"').replace('nan', '0.0')))
# exit()

for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][0:2]:
    for al in algo:
        algo_names.append(al+'-'+str(lr))

data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62]
start_t = time.time()

this_idx = 0
res_betas = []
for seed in data_seeds[4:5]:

    for data_name in data_names:
        data_pack, data_kwarg = gen_data(data_name, seed)

        for algo_name in algo_names:
            st = data_name + '+' + algo_name
            if st not in storage:
                storage[st] = []
            if st not in acc:
                acc[st] = []

            try:
                all_pack = run_algo(data_pack, data_kwarg, algo_name, seed)
                res_betas.append(all_pack['res_beta'])
                # all_pack['data_pack'].pop('data')
                # storage[st].append(all_pack)
                score = get_eval(Dict(all_pack))
                acc[st].append(score)
                print(st, score, '\n--------------------\n')
            except np.linalg.linalg.LinAlgError:
                print("Cannot solve equation for initialization.")
                pass

            this_idx += 1
            percent = 1. * this_idx / len(data_names) / len(algo_names) / len(data_seeds)
            print('============ progress', percent, 'ETA', (time.time() - start_t) / percent * (1 - percent), 'Elpased',
                  (time.time() - start_t))

# while os.path.isfile('lock'):
#     time.sleep(1)
# os.system('touch lock')
# if os.path.isfile('acc.pkl'):
#     sf = open('storage.pkl', 'rb')
#     af = open('acc.pkl', 'rb')
#     storage =  {**storage, **pkl.load(sf)}
#     acc = {**acc, **pkl.load(af)}
#     sf.close()
#     af.close()
#
# sf = open('storage.pkl', 'wb')
# af = open('acc.pkl', 'wb')
# pkl.dump(acc, af)
# pkl.dump(storage, sf)
# sf.close()
# af.close()
# os.system('rm lock')
# from pprint import pprint
# for i in range(3):
#     pprint(res_betas)
pprint.pprint(acc)
open('acc.txt', 'w').write(str(acc))
