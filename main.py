from addict import Dict

from algo_func import *
from data_helper import *
from eval_paint import *

import json
import pprint
import pandas as pd

import matplotlib.pyplot as plt
import pickle as pkl
import os, sys
import time

if __name__ == '__main__':

    data_name_bases = [
        'be-b5,1-j100-i100',
        'be-b2,1-j100-i100',
        'be-b2,2-j100-i100',
        'be-b1,1-j100-i100',
        'be-b1,2-j100-i100',
        'be-b1,5-j100-i100'
    ]
    # data_name_base = data_name_bases[int(sys.argv[1])]
    data_name_base = data_name_bases[0]
    print(data_name_base, '----------------------\n\n\n\n')

    data_name_suffix = [
        '0.8k1',
        '0.16k5',
        '0.08k10',
        '0.04k20',
        '1.0k1'
        ]
    data_names = [
        'be-b1,10-j8-i100-p8000',
        'ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000',
        'po-b10-j16-i100-p8000',
        'be-b1,10-j8-i100-p80000',
    ]
    for dns in data_name_suffix:
        data_names.append(data_name_base+'-'+dns)

    algo_names = [
        'btl-spectral-do',
        # 'btl-spectral-mle',
        # 'btl-random-mle',
        'gbtl-spectral_all-do',
    ]
    algo = [
        'gbtl-spectral_all-mle',
        'gbtl-random_all-mle',
        'gbtlneg-spectral_all-mle',  #
        'gbtlinv-spectral_all-mle',  #
        'gbtlinv-random_all-mle',  #

        # 'gbtl-disturb-mle',
        # 'gbtlneg-disturb-mle',  #
        # 'gbtlinv-disturb-mle',  #

        # 'gbtl-disturb_random_b_fix^s-mle',  #
        # 'gbtlneg-disturb_random_b_fix^s-mle',  #
        # 'gbtlinv-disturb_random_b_fix^s-mle',  #
        # 'gbtlinv-disturb_random_b_fix^s-mle',  #
    ]

    acc = Dict()
    # key: dn + an
    # val: [result]

    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][0:1]:
        for al in algo:
            algo_names.append(al+'-'+str(lr))

    data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62][3:5]
    start_t = time.time()

    # base_dir = '0_temp'
    base_dir = '1_b_' + data_name_base
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
###########################################################
    if len(sys.argv) == 1:
        cols = data_name_suffix
        res_table = []
        for data_name_base in data_name_bases:
            base_dir = '1_b_' + data_name_base
            res = json.loads(open(os.path.join(base_dir, data_name_base + '.json'), 'r').read())
            this_row = [data_name_base]
            for suffix in data_name_suffix:
                st = data_name_base + '-' + suffix + '+' + algo_names[1]
                print(st)
                this_row.append(res[st][0])
            res_table.append(this_row)
        cols = ['data_name_base'] + cols
        df = pd.DataFrame(res_table, columns=cols).to_csv('gbtl.csv')
        print(df)
        exit()
##########################################################
    this_idx = 0
    for seed in data_seeds:
        for data_name in data_names:
            img_path = os.path.join(base_dir, data_name+'.png')
            data_pack, data_kwarg = gen_data(data_name, seed, save_path=img_path)
            for algo_name in algo_names:
                st = data_name + '+' + algo_name
                if st not in acc:
                    acc[st] = []

                try:
                    all_pack = run_algo(data_pack, data_kwarg, algo_name, seed)
                    all_pack = Dict(all_pack)

                    plt.figure(figsize=(6, 4)).suptitle('likelihood')
                    plt.plot(all_pack.pr_list)
                    plt.savefig(os.path.join(base_dir, st+'.pr_list.png'), bbox_inches='tight', dpi=96)
                    plt.close()

                    plt.figure(figsize=(6, 4)).suptitle('s')
                    plt.plot(all_pack.s_list)
                    plt.savefig(os.path.join(base_dir, st + '.s_list.png'), bbox_inches='tight', dpi=96)
                    plt.close()

                    score = get_eval(all_pack)
                    acc[st].append(score)
                    print(st, score, '\n--------------------\n')
                except np.linalg.linalg.LinAlgError:
                    print("Cannot solve equation for initialization.")
                    pass

                this_idx += 1
                percent = 1. * this_idx / len(data_names) / len(algo_names) / len(data_seeds)
                print('============ progress', percent, 'ETA', (time.time() - start_t) / percent * (1 - percent), 'Elpased',
                      (time.time() - start_t))

    pprint.pprint(acc)
    open(os.path.join(base_dir, data_name_base+'.json'), 'w').write(json.dumps(acc, indent=4))
