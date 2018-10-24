from addict import Dict

from algo_func import *
from data_helper import *
from eval_paint import *

import json
import pprint

import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import time

if __name__ == '__main__':

    base_str = ''

    s_true = list(reversed([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    beta_true = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    n_items = len(s_true)
    n_judges = len(beta_true)
    data_name_base = '-'.join(['ma', ','.join(list(map(str, beta_true))), 'j' + str(n_judges), 'i' + str(n_items)])

    s_true = np.arange(1., 0., -.1)
    s_true = np.arange(0., 1., .1)
    good_beta = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75][:1])
    bad_beta = np.array([0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0][:1])

    data_name_suffix = [
        # '0.8k1',
        # '0.4k1',
        # '0.2k1',
        '0.1k1',
        # '1.0k1',
        # '0.16k5',
        # '0.08k10',
        # '0.04k20',
        # '0.8k4',
        # '1.0k4',
        # '0.16k20',
        # '0.08k40',
        # '0.04k80',
        # 'p1000'  # randomly generate 2000 pairs (i, j) i!=j regardless repeated or not
    ]

    algo_names = [
        'btl-spectral-do',
        # 'btl-spectral-mle',
        # 'btl-random-mle',
        # 'gbtl-spectral_all-do',
        # 'gbtlneg-spectral_all-do',
        # 'gbtlneg-moment-do',
        # 'gbtlneg-ml-do',
    ]
    algo = [
        # 'gbtl-spectral_all-mle',
        # 'gbtlneg-moment-mle',
        'gbtlneg-ml-mle',
        # 'gbtl-random_all-mle',
        # 'gbtlneg-spectral_all-mle',  #
        # 'gbtlneg-random_all-mle',  #
        # 'gbtlinv-spectral_all-mle',  #
        # 'gbtlinv-random_all-mle',  #
        #
        # 'gbtl-disturb-mle',
        # 'gbtlneg-disturb-mle',  #
        # 'gbtlinv-disturb-mle',  #

        # 'gbtl-disturb_random_b_fix^s-mle',  #
        # 'gbtlneg-disturb_random_b_fix^s-mle',  #
        # 'gbtlinv-disturb_random_b_fix^s-mle',  #
    ]

    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][0:1]:
        for al in algo:
            algo_names.append(al+'-'+str(lr))

    acc = []
    bta = []
    for i in range(len(algo_names)):
        acc.append([])
        bta.append([])

    data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62][:1]
    start_t = time.time()

    data_names = []
    for dns in data_name_suffix:
        data_names.append(data_name_base+'-'+dns)

    base_dir = './'
    this_idx = 0

    print(algo_names)
    for data_name in data_names:
        for b_1 in good_beta:
            for b_2 in bad_beta:
                # beta_true = [good_beta] * 2 + [bad_beta] * 4
                beta_true = [b_1] * 2 + [b_2] * 4

                img_path = os.path.join(base_dir, data_name+'.png')
                for algo_idx, algo_name in enumerate(algo_names):
                    this_algo_acc = []
                    this_algo_bta = []
                    for seed in data_seeds:
                        data_pack, data_kwarg = gen_data(data_name, seed, save_path=None, s_true=s_true,
                                                         beta_true=beta_true)
                        st = data_name + '+' + algo_name

                        try:
                            all_pack = run_algo(data_pack, data_kwarg, algo_name, seed)
                            all_pack = Dict(all_pack)

                            score = get_eval(all_pack)
                            berr = np.linalg.norm((all_pack.beta_est - data_pack.beta_true) / data_pack.beta_true)
                            this_algo_acc.append(score)
                            this_algo_bta.append(berr)
                            print('score, bta_err_ratio', score, berr, st)
                            print('............ estimation value .............')
                            print('s_est', all_pack.s_est)
                            print('beta_est', all_pack.beta_est)
                            print('--------------------\n')
                        except np.linalg.linalg.LinAlgError:
                            # this_algo_acc.append(0.)
                            print("Cannot solve equation for initialization.")
                            pass

                        this_idx += 1
                        percent = 1. * this_idx / len(data_names) / len(algo_names) / len(data_seeds)
                        print('============ progress', percent, 'ETA', (time.time() - start_t) / percent * (1 - percent), 'Elpased',
                              (time.time() - start_t))

                    acc[algo_idx].append(np.average(this_algo_acc))
                    bta[algo_idx].append(np.average(this_algo_bta))

    print(acc)
    print(bta)
    for algo_idx, algo_name in enumerate(algo_names):
        print(algo_name, ',', algo_idx, end='')
        for i, b_1 in enumerate(good_beta):
            print('')
            for j, b_2 in enumerate(bad_beta):
                print(acc[algo_idx][i * len(bad_beta) + j], ',', end='')

        print('')

    print('---------------------------')

    for algo_idx, algo_name in enumerate(algo_names):
        print(algo_name, ',', algo_idx, end='')
        for i, b_1 in enumerate(good_beta):
            print('')
            for j, b_2 in enumerate(bad_beta):
                print(bta[algo_idx][i * len(bad_beta) + j], ',', end='')

        print('')
