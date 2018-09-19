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
    data_name_base = []
    eval = False
    if len(sys.argv) == 3:
        base_str = sys.argv[1]
        data_name_base = sys.argv[2]
        print(data_name_base, '----------------------\n\n\n\n')
    elif len(sys.argv) == 2:
        eval = True
        base_str = sys.argv[1]
    else:
        print('argument not given correctly.')
        exit(-1)

    data_name_suffix = [
        '0.8k1',
        '1.0k1',
        '0.16k5',
        '0.08k10',
        '0.04k20',
        '0.8k4',
        '1.0k4',
        '0.16k20',
        '0.08k40',
        '0.04k80',
    ]

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
        'gbtlneg-random_all-mle',  #
        'gbtlinv-spectral_all-mle',  #
        'gbtlinv-random_all-mle',  #

        # 'gbtl-disturb-mle',
        # 'gbtlneg-disturb-mle',  #
        # 'gbtlinv-disturb-mle',  #

        # 'gbtl-disturb_random_b_fix^s-mle',  #
        # 'gbtlneg-disturb_random_b_fix^s-mle',  #
        # 'gbtlinv-disturb_random_b_fix^s-mle',  #
    ]

    acc = Dict()
    # key: dn + an
    # val: [result]

    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][0:1]:
        for al in algo:
            algo_names.append(al+'-'+str(lr))

    data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62][3:4]
    start_t = time.time()

    if eval:
        cols = ['data_name_base'] + data_name_suffix
        f = open(os.path.join(base_str, base_str + '.txt'), 'r')
        data_name_bases = f.read().split(' ')

        for algo_name in algo_names[:2]:
            res_table = []
            for data_name_base in data_name_bases:
                base_dir = base_str + '_' + data_name_base
                res = json.loads(open(os.path.join(base_dir, data_name_base + '.json'), 'r').read())
                # res = json.loads(open(os.path.join(base_str, data_name_base + '.json'), 'r').read())
                this_row = [data_name_base]
                for suffix in data_name_suffix:
                    st = data_name_base + '-' + suffix + '+' + algo_name
                    print(st)
                    this_row.append(np.sum(res[st]) / len(res[st]))
                res_table.append(this_row)
            df = pd.DataFrame(res_table, columns=cols)
            df.to_csv(os.path.join(base_str, algo_name+'.csv'))
            print(df)
        #line + sgd + disturb : likelihood
        #

        # for paint_pair in paint_pairs:
        #
        #     for data_name_base in data_name_bases:
        #         base_dir = base_str
        #         for suffix in data_name_suffix:
        #             plt.figure(figsize=(12, 36))
        #             fig, (ax0, ax1) = plt.subplots(nrows=2)
        #             for idx, algo_name in enumerate(paint_pair):
        #                 st = data_name_base + '-' + suffix + '+' + algo_name + '-' + '0.0005' + '.pr_list.' + '{}'
        #                 sgd = json.loads(open(os.path.join(base_dir, st.format('sgd') + '.json'), 'r').read())
        #                 line = json.loads(open(os.path.join(base_dir, st.format('line') + '.json'), 'r').read())
        #
        #                 ax0.plot(range(len(sgd)), sgd, '--', color=color[idx], label='sgd '+algo_name)
        #                 ax1.plot(range(len(sgd[:25])), sgd[:25], '--', color=color[idx], label='sgd '+algo_name)
        #                 ax0.plot(range(len(line)), line, color=color[idx], label='line '+algo_name)
        #                 ax1.plot(range(len(line[:25])), line[:25], color=color[idx], label='line '+algo_name)
        #
        #             ax1.legend(loc='upper right', fontsize='x-small')
        #             plt.savefig(os.path.join
        #                         (base_dir,
        #                          'summary.' + data_name_base + '-' + suffix + '.{}.png'.format(algo_name.split('-')[0])),
        #                         bbox_inches='tight', dpi=96)
        #             plt.close('all')
        exit(0)

    base_dir = base_str + '_' + data_name_base
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    data_names = []
    for dns in data_name_suffix:
        data_names.append(data_name_base+'-'+dns)

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
                    if (len(all_pack.pr_list) < 1):
                        impp = 0.
                    else:
                        impp = (all_pack.pr_list[0] - all_pack.pr_list[-1]) / all_pack.pr_list[-1]
                    imp[st].append(impp)
                    print(st, score, '\n--------------------\n')
                except np.linalg.linalg.LinAlgError:
                    print("Cannot solve equation for initialization.")
                    pass

                this_idx += 1
                percent = 1. * this_idx / len(data_names) / len(algo_names) / len(data_seeds)
                print('============ progress', percent, 'ETA', (time.time() - start_t) / percent * (1 - percent), 'Elpased',
                      (time.time() - start_t))

    pprint.pprint(acc)
    pprint.pprint(imp)
    acc['imp'] = imp
    open(os.path.join(base_dir, data_name_base+'.json'), 'w').write(json.dumps(acc, indent=4))
