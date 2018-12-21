import numpy as np

import json
import os, sys
sys.path.append(os.getcwd())
import time
import datetime
from ifce.invoke import Job
from addict import Dict

if __name__ == '__main__':

    base_str = 'experiments'
    exp_name = '17_single_dist'
    exp_comments = 'test beta sampled from a single distribution, but s is still arithmetic'

    data_name_suffix = [
        '0.8k1',
        '0.4k1',
        '0.2k1',
        '0.1k1',
        # '0.05k1',
        # '0.02k1',
        # '0.01k1',
        # '0.005k1',
        # '1.0k1',
        # '0.16k5',
        # '0.08k10',
        # '0.04k20',
        # '0.8k4',
        # '1.0k4',
        # '0.16k20',
        # '0.08k40',
        # '0.04k80',
        # 'p1000'
    ]
    beta_sign = 1

    if len(sys.argv) == 3:
        data_name_base = sys.argv[1]
        beta_sign = 1 if sys.argv[2] == '-1' else 0
    else:
        exit(-1)

    algo_names = [
        'btl-spectral-do',
        # 'gbtl-spectral-do',
        # 'gbtlneg-spectral-do',
        # 'gbtlneg-moment-do',
        # 'gbtlneg-ml-do',
    ]
    algo = [
        # 'btl-spectral-mle',
        # 'btl-random-mle',

        # 'gbtl-spectral-mle',
        # 'gbtlneg-moment-mle',
        # 'gbtlneg-ml-mle',
        # 'gbtl-random-mle',
        # 'gbtlneg-spectral-mle',
        # 'gbtlneg-random-mle',
        # 'gbtlinv-moment-mle',
        'gbtlinv-ml-mle',
        # 'gbtlinv-spectral-mle',
        # 'gbtlinv-random-mle',
        #
        # 'gbtl-disturb-mle',
        # 'gbtlneg-disturb-mle',
        # 'gbtlinv-disturb-mle',

        # 'gbtl-disturb_random_b_fix^s-mle',
        # 'gbtlneg-disturb_random_b_fix^s-mle',
        # 'gbtlinv-disturb_random_b_fix^s-mle',
    ]

    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][3:4]:
        for al in algo:
            algo_names.append(al+'-'+str(lr))

    data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62]
    start_t = time.time()

    data_names = []
    for dns in data_name_suffix:
        data_names.append(data_name_base + '-' + dns)

    base_dir = os.path.join(base_str, exp_name)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    done_jobs = 0
    total_jobs = len(data_names) * len(algo_names) * len(data_seeds)

    task_meta = Dict()
    task_meta.exp_name = exp_name
    task_meta.datetime = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    task_meta.exp_comments = exp_comments
    # dt = task_meta.datetime
    dt = ''
    jf_out = open(os.path.join(base_dir, dt + '.json'), 'w')
    jf_out.write(json.dumps(task_meta.to_dict(), indent=4))
    jf_out.close()

    res_all = []
    acc = []
    for i in range(len(algo_names)):
        acc.append([])

    print(algo_names)
    for data_name in data_names:
        img_path = os.path.join(base_dir, data_name+'.png')
        for algo_idx, algo_name in enumerate(algo_names):
            this_algo_acc = []
            st = data_name + '+' + algo_name
            for seed in data_seeds:
                jb = Job(load=False, data_name=data_name, algo_name=algo_name, seed=seed)

                try:
                    jb.run_algo()
                    json_str = jb.serialize(st, dt)
                    res_all.append(json_str)

                    score = jb.eval()
                    this_algo_acc.append(score)
                except:
                    print("this algo and data failed")

                done_jobs += 1
                percent = done_jobs / total_jobs
                print('>> {0:0.2f} ETA {1:0.0f}s Elapsed {2:0.0f}s\n'.format(percent, (time.time() - start_t) / percent * (1 - percent),
                                                                             (time.time() - start_t)))

            acc[algo_idx].append(np.average(this_algo_acc))

    jf_out = open(os.path.join(base_dir, dt + '-res.json'), 'w')
    jf_out.write(json.dumps(res_all, indent=4))
    jf_out.close()

    for algo_idx, algo_name in enumerate(algo_names):
        print(algo_name, ',', algo_idx, end='')
        print('')
        for i, good_beta in enumerate(data_names):
            print(acc[algo_idx][i], ',', end='')

        print('')
