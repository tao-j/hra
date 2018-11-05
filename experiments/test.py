import numpy as np

import os, sys
import time
from ifce.invoke import Job

if __name__ == '__main__':

    base_str = ''

    s_true = np.arange(1., 0., -.1)
    # good_beta = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75])
    good_betas = np.array([0.2, 0.05, 0.02])
    n_good_beta = 3
    n_good_beta_adv = 1
    # bad_beta = np.array([0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0])
    bad_betas = np.array([1., 0.4, 0.2])
    n_bad_beta = 6
    n_bad_beta_adv = 2

    n_judges = n_good_beta + n_bad_beta
    n_items = len(s_true)

    data_name_base = '-'.join(['pa', '{}', 'j' + str(n_judges), 'i' + str(n_items)])

    data_name_suffix = [
        # '0.8k1',
        '0.4k1',
        # '0.2k1',
        # '0.1k1',
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

    data_name_suffix = [sys.argv[1]]
    sign = 1 if sys.argv[2] == '0' else 0
    n_bad_beta_adv *= sign
    n_good_beta_adv *= sign

    algo_names = [
        'btl-spectral-do',
        'btl-spectral-mle',
        # 'btl-random-mle',
        # 'gbtl-spectral-do',
        # 'gbtlneg-spectral-do',
        # 'gbtlneg-moment-do',
        # 'gbtlneg-ml-do',
    ]
    algo = [
        # 'gbtl-spectral-mle',
        # 'gbtlneg-moment-mle',
        'gbtlneg-ml-mle',
        # 'gbtl-random-mle',
        # 'gbtlneg-spectral-mle',  #
        # 'gbtlneg-random-mle',  #
        # 'gbtlinv-spectral-mle',  #
        # 'gbtlinv-random-mle',  #
        #
        # 'gbtl-disturb-mle',
        # 'gbtlneg-disturb-mle',  #
        # 'gbtlinv-disturb-mle',  #

        # 'gbtl-disturb_random_b_fix^s-mle',  #
        # 'gbtlneg-disturb_random_b_fix^s-mle',  #
        # 'gbtlinv-disturb_random_b_fix^s-mle',  #
    ]

    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0][0:2]:
        for al in algo:
            algo_names.append(al+'-'+str(lr))

    acc = []
    bta = []
    for i in range(len(algo_names)):
        acc.append([])
        bta.append([])

    data_seeds = [1313, 3838, 6262, 1338, 1362, 3862, 6238, 6213, 3813, 13, 38, 62][:8]
    start_t = time.time()

    data_names = []
    for dns in data_name_suffix:
        data_names.append(data_name_base + '-' + dns)

    base_dir = './'
    this_idx = 0

    print(algo_names)
    for data_name in data_names:
        for good_beta in good_betas:
            for bad_beta in bad_betas:
                beta_true = [good_beta] * (n_good_beta - n_good_beta_adv) + \
                            [bad_beta] * (n_bad_beta - n_bad_beta_adv) + \
                            [-good_beta] * n_good_beta_adv + \
                            [-bad_beta] * n_bad_beta_adv
                assert len(beta_true) == n_judges
                data_name = data_name.format(','.join(list(map(str, beta_true))))

                img_path = os.path.join(base_dir, data_name+'.png')
                for algo_idx, algo_name in enumerate(algo_names):
                    this_algo_acc = []
                    for seed in data_seeds:
                        jb = Job(load=False, data_name=data_name, algo_name=algo_name, seed=seed,
                                 s_true=s_true, beta_true=beta_true)
                        st = data_name + '+' + algo_name

                        try:
                            res_pack = jb.run_algo()

                            score = jb.eval()
                            # this_algo_acc.append(score)
                            # print('score', score, st)
                            # print('............ estimation value .............')
                            # print('s_est', all_pack.s_est)
                            # print('beta_est', all_pack.beta_est)
                            # print('--------------------\n')
                        except np.linalg.linalg.LinAlgError:
                            # this_algo_acc.append(0.)
                            print("Cannot solve equation for initialization.")
                            pass

                        # this_idx += 1
                        # percent = 1. * this_idx / len(data_names) / len(algo_names) / len(data_seeds)
                        # print('============ progress', percent, 'ETA', (time.time() - start_t) / percent * (1 - percent), 'Elpased',
                        #       (time.time() - start_t))

                    # acc[algo_idx].append(np.average(this_algo_acc))
                    # bta[algo_idx].append(np.average(this_algo_bta))

    print(acc)
    # print(bta)
    for algo_idx, algo_name in enumerate(algo_names):
        print(algo_name, ',', algo_idx, end='')
        for i, good_beta in enumerate(good_betas):
            print('')
            for j, bad_beta in enumerate(bad_betas):
                print(acc[algo_idx][i * len(bad_betas) + j], ',', end='')

        print('')
