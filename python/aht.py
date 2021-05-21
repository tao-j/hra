import numpy as np
from python.cmpsort import *


def atc(i, j, cM, M, eps, delta, ranked_s, original_s, gamma):
    t = 0
    w = 0
    m_t = len(cM)
    b_max = np.ceil(1. / 2 / m_t / eps ** 2 * np.log(2 / delta))
    bn = np.zeros(M)
    p = 0.5
    r = 0
    # gamma = 1. / np.array(gamma)
    for t in range(1, int(b_max)):
        for u in cM:
            # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            pij = np.exp(gamma[u] * original_s[i]) / (np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
            y = 1 if np.random.random() < pij else 0
            if y == 1:
                w += 1
                bn[u] += 1
        r = t
        b_t = np.sqrt(1. / 2 / (r + 1) / m_t * np.log(np.pi ** 2 * (r + 1) ** 2 / 3 / delta))
        p = w / r / len(cM)
        if p > 0.5 + b_t:
            break
        if p < 0.5 - b_t:
            break

    atc_y = 1 if p > 0.5 else 0
    bn = bn if p > 0.5 else r - bn
    return atc_y, bn, r


def elimuser_oneshot(cM, n_t, s_t, eps, delta):
    if len(cM) == 1:
        return cM
    s_max = int(np.ceil(2 / eps / eps * np.log(len(cM) / delta)))
    if s_t > s_max:
        mu_t = n_t / s_t
        i_best = np.argmax(mu_t)
        # mu1 = mu_t[i_best]
        # mu_t[i_best] = 0
        # i2 = np.argmax(mu_t)
        # mu2 = mu_t[i2]
        # d = mu1 - mu2
        # # if d > eps:
        cM = [i_best]
    return cM


def gamma_sweep(act, repeat, delta=0.1):
    random.seed(123)
    np.random.seed(123)

    # data gen method may be mentioned in Ren et al.
    thetas = []
    for i in range(100):
        li = 0.9 * np.power(1.2 / 0.8, i)
        ri = 1.1 * np.power(1.2 / 0.8, i)
        thetas.append(li + (ri - li) * (np.random.random()))
        # thetas.append(np.power(1.2 / 0.8, i))

    N = 10
    M = 1
    # for gb in [0.25, 1., 2.5]:
    #     for gg in [2.5, 5, 10]:
    # for gb in [2.5]:
    #     for gg in [2.5]:
    for gb in [1.]:
        for gg in [1.]:

            gamma = [gg] * (M // 3) + [gb] * (M // 3 * 2)
            gamma += [gb] * (M - len(gamma))
            # s = np.linspace(1 / n, 1, n)
            s = np.log(thetas[:N])
            tts = []
            np.random.shuffle(s)
            for i in range(repeat):
                cM = range(0, M)
                cmp_sort = CmpSort(s, delta)
                # print(s)
                rank_sample_complexity = 0
                n_t = np.zeros(M)
                s_t = 0
                while not cmp_sort.done:
                    pair = cmp_sort.next_pair()
                    assert(0 <= pair[0] <= cmp_sort.n_intree)
                    assert (-1 <= pair[1] <= cmp_sort.n_intree)
                    if pair[1] == -1:
                        cmp_sort.feedback(1)
                    elif pair[1] == cmp_sort.n_intree:
                        cmp_sort.feedback(0)
                    else:
                        y, bn, r = atc(pair[0], pair[1], cM, M, cmp_sort.epsilon_atc_param, cmp_sort.delta_atc_param, cmp_sort.ranked_list, s, gamma)
                        s_t += r
                        n_t += bn
                        rank_sample_complexity += r * len(cM)
                        if act:
                            cM = elimuser_oneshot(cM, n_t, s_t, 0.1, 0.3)
                        cmp_sort.feedback(y)

                tts.append(rank_sample_complexity)
                # print(len(cM))
                a_ms = list(cmp_sort.ranked_list)
                # print(a_ms)
                a_sorted = sorted(s)
                # print(a_sorted)

                # if len(cM) == 1:
                #     if not gamma[cM[0]] == gg:
                #         print(cM)
                #
                assert (a_ms == a_sorted)
                # print(cM)
            # print('&', int(np.average(tts)), " $\\pm$ ", int(np.std(tts)), end='\t\t'),
            print(int(np.average(tts)))
            # print('\n'.join(list(map(str, tts))))
        # print('')


if __name__ == "__main__":
    import random
    import itertools
    repeat = 5
    for delta in np.arange(0.05, 1, 0.05):
        gamma_sweep(act=0, repeat=repeat, delta=delta)
        # print()
        # func_call(act=1, repeat=repeat, delta=delta)
