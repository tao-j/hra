import numpy as np
import scipy
import scipy.stats

def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()


def acc_func(pred, true):
    print(scipy.stats.mannwhitneyu(pred, true))
    print(scipy.stats.spearmanr(pred, true))
    print(scipy.stats.kendalltau(pred, true))

    n = len(pred)
    nomi = 0.
    deno = 0.
    snomi = 0.
    assert len(pred) == len(true)
    for i in range(n):
        for j in range(n):
            if true[j] > true[i]:
                deno += 1
                if pred[j] > pred[i]:
                    nomi += 1
                else:
                    snomi += 1

    cnt = 0
    total_cnt = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if true[i] == true[j]:
                continue
            else:
                total_cnt += 1
            cnt += 1 if np.sign(true[i]-true[j]) == np.sign(pred[i]-pred[j]) else 0
    auc = 1. * cnt / total_cnt

    print('tau-b-n^2: ', (nomi-snomi) / deno)
    print('ACC', nomi / deno)
    print('AUC', auc)
    return scipy.stats.kendalltau(pred, true)[0]


def get_eval(all_pack):
    s_est = all_pack.s_est
    rank_pred = s_est
    rank_true = all_pack.data_pack.s_true
    if np.any(np.isnan(np.array(s_est))):
        acc = np.nan
    else:
        acc = acc_func(rank_pred, rank_true)
    return acc


def print_all_eval(pred, true):
    print(pred)
    print(true)
    print('man whitney:', scipy.stats.mannwhitneyu(pred, true))
    print('spearman:', scipy.stats.spearmanr(pred, true))
    print('kendall:', scipy.stats.kendalltau(pred, true))


if __name__ == '__main__':
    pred = list(range(2, 5)) + list(range(2, 9))
    true = np.arange(len(pred))
    print_all_eval(pred, true)

    print('arg sorted')
    pred = np.argsort(pred)
    true = np.argsort(true)
    print_all_eval(pred, true)
