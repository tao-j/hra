import numpy as np
from python.cmpsort import *


def atc_rounds(i, j, cM, M, eps, delta, ranked_s, original_s, gamma):
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


def atc_rand(i, j, cM, M, eps, delta, ranked_s, original_s, gamma, bs, A):
    m_t = len(cM)
    t_max = int(np.ceil(1. / 2 / (eps ** 2) * np.log(2 / delta)))
    bn = np.zeros(M)
    p = 0.5
    w = 0
    for t in range(1, t_max + 1):
        u = int(np.floor(np.random.random() * m_t))
        bs[u] += 1
        # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
        # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
        pij = np.exp(gamma[u] * original_s[i]) / (np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
        y = 1 if np.random.random() < pij else 0
        if y == 1:
            A[u][i, j] += 1
            w += 1
        else:
            A[u][j, i] += 1
        b_t = np.sqrt(1. / 2 / t * np.log(np.pi * np.pi * t * t / 3 / delta))
        p = w / t
        if p > 0.5 + b_t:
            break
        if p < 0.5 - b_t:
            break

    atc_y = 1 if p > 0.5 else 0
    return atc_y, bs, A, t


def elimuser_uneven_ucb(cM, bn, bs, delta):
    mu = bn / bs
    # TODO: log2 ?
    r = np.sqrt(np.log2(2 * len(cM) / delta) / 2 / bs)
    bucb = mu + r
    blcb = mu - r
    to_remove = set()
    for u in range(len(cM)):
        for up in range(len(cM)):
            if bucb[u] < blcb[up]:
                to_remove.add(u)
                break
    new_cM = []
    for u in cM:
        if u not in to_remove:
            new_cM.append(u)
    return new_cM


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


def best_user_act_rank(N, M, e2, d1, d2, s, gamma):
    cM = range(0, M)
    cmp_sort = CmpSort(s, delta)
    # print(s)
    rank_sample_complexity = 0
    n_t = np.zeros(M)
    s_t = 0
    while not cmp_sort.done:
        pair = cmp_sort.next_pair()
        assert (0 <= pair[0] <= cmp_sort.n_intree)
        assert (-1 <= pair[1] <= cmp_sort.n_intree)
        if pair[1] == -1:
            cmp_sort.feedback(1)
        elif pair[1] == cmp_sort.n_intree:
            cmp_sort.feedback(0)
        else:
            y, bn, r = atc_rounds(pair[0], pair[1], cM, M, cmp_sort.epsilon_atc_param, cmp_sort.delta_atc_param,
                                  cmp_sort.ranked_list, s, gamma)
            s_t += r
            n_t += bn
            rank_sample_complexity += r * len(cM)
            # if act:
            cM = elimuser_oneshot(cM, n_t, s_t, 0.1, 0.3)
            cmp_sort.feedback(y)

    return rank_sample_complexity, cmp_sort.ranked_list


def create_A(N, M):
    A = []
    for i in range(M):
        A.append(np.zeros((N, N)))
    return A


def best_user_act_rank_rand_u(N, M, delta, s, s_idx, gamma):
    cM = range(0, M)
    cmp_sort = CmpSort(s, delta)
    rank_sample_complexity = 0

    # number of times user is asked
    bs = np.zeros(M)
    # number of times user is correct
    bn = np.zeros(M)
    # temp matrix list holding user response
    A = create_A(N, M)
    current_idx = 0
    while not cmp_sort.done:
        pair = cmp_sort.next_pair()
        assert (0 <= pair[0] <= cmp_sort.n_intree)
        assert (-1 <= pair[1] <= cmp_sort.n_intree)
        if pair[1] == -1:
            cmp_sort.feedback(1)
        elif pair[1] == cmp_sort.n_intree:
            cmp_sort.feedback(0)
        else:
            y, bs, A, t = atc_rand(pair[0], pair[1], cM, M, cmp_sort.epsilon_atc_param, cmp_sort.delta_atc_param,
                                  cmp_sort.ranked_list, s, gamma, bs, A)

            rank_sample_complexity += t
            inserted, inserted_place = cmp_sort.feedback(y)
            if inserted:
                # idx
                assert inserted_place != -1
                inserted_idx = len(cmp_sort.ranked_list)
                B = np.zeros((N, N))
                for i in range(inserted_idx):
                    if inserted_place > i:
                        B[i, inserted_place] = 1
                    elif inserted_place < i:
                        B[inserted_place, i] = 1
                for u in range(M):
                    bn[u] += sum(sum(A[u] * B))
                A = create_A(N, M)
                cM = elimuser_uneven_ucb(cM, bn, bs, delta)

    return rank_sample_complexity, cmp_sort.ranked_list


def gamma_sweep(act, repeat, eps=0.1, delta=0.1):
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
    M = 9
    # for gb in [0.25, 1., 2.5]:
    #     for gg in [2.5, 5, 10]:
    # for gb in [2.5]:
    #     for gg in [2.5]:
    for gb in [0.5]:
        for gg in [5.]:

            gamma = [gg] * (M // 3) + [gb] * (M // 3 * 2)
            gamma += [gb] * (M - len(gamma))
            # s = np.linspace(1 / n, 1, n)
            s = np.log(thetas[:N])
            tts = []
            s_idx = list(range(0, len(s)))
            random.shuffle(s_idx)
            s = s[s_idx]
            # np.random.shuffle(s)

            for _ in range(repeat):
                # rank_sample_complexity, ranked_list = best_user_act_rank(N, M, 0.1, delta, delta, s, gamma)
                rank_sample_complexity, ranked_list = best_user_act_rank_rand_u(N, M, delta, s, s_idx, gamma)
                tts.append(rank_sample_complexity)
                # print(len(cM))
                a_ms = list(ranked_list)
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
    repeat = 10
    delta = 0.1
    # for delta in np.arange(0.05, 1, 0.05):
    gamma_sweep(act=0, repeat=repeat, delta=delta)
        # print()
        # func_call(act=1, repeat=repeat, delta=delta)
