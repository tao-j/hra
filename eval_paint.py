import multiprocessing as mp
import numpy as np
import scipy

# https://gist.github.com/bwhite/3726239
def err_func(pred):
    return np.abs(np.arange(len(pred)) - pred).mean()


def get_eval(all_pack):
    s_pred = all_pack.res_s
    rank_pred = np.argsort(s_pred)
    rank_orig = np.argsort(all_pack.data_pack.s)
    if np.any(np.isnan(np.array(s_pred))):
        acc = np.nan
    else:
        acc = acc_func(rank_pred, rank_orig)
    
#     if result_pack is not None:
#         result_pack.append(
#             [np.linalg.norm(res_s[rank]-s_true),
#              np.linalg.norm(( (res_eps**2-eps_true**2)/eps_true**2))
#             ])
    return acc

def acc_func(pred, src=None):
    if src is None:
        src = np.arange(len(pred))
    return scipy.stats.spearmanr(pred, src)[0]

def async_train(fp, args_ls):
    
    def f(q, fp, args_l):
        print(args_l[1])
        res = fp(*args_l)
        q.put(res)
    result_err = []

    q = mp.Queue()
    ps = []
    for args_l in args_ls:
    #     rets = pool.apply_async(f, (q, np.arange(pid)))
        p = mp.Process(target=f, args=(q, fp, args_l))
        ps.append(p)
        p.start()

    for p in ps:
        result_err.append(q.get())

    for p in ps:
        p.join()
    
    return result_err
