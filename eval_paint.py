from algo_func import *
from data_helper import *
from addict import Dict

# algorithm settings string definition
# beta_gen_func-samole_beta_distribution-j_#judges-i_#items
# algo-init_method-optmethod-lr

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


def gen_data(data_name, seed, save_path=None, s_true=None, beta_true=None):
    dn_attrs = data_name.split('-')

    beta_gen_params_str = list(map(float, dn_attrs[1].split(',')))
    beta_gen_func = dn_attrs[0]
    dn_lookup = {
        'po': ('power', beta_gen_params_str[0]),
        'be': ('beta', beta_gen_params_str),
        'ga': ('gamma', beta_gen_params_str),
        'sh': ('shrink', beta_gen_params_str[0]),
        'ex': ('ex', beta_gen_params_str[0]),
        'ne': ('negative', beta_gen_params_str[0]),
        'ma': ('manual', beta_gen_params_str),
        'pa': ('passwd-in', beta_gen_params_str)
    }
    _, beta_gen_params = dn_lookup[beta_gen_func]

    if dn_attrs[2][0] != 'j' and dn_attrs[3][0] != 'i':
        print('item / judge number not given correctly.')
        raise NotImplementedError
    nj = int(dn_attrs[2][1:])
    ni = int(dn_attrs[3][1:])
    nps = 0
    known_pairs_ratio = 0.
    repeated_comps = 0
    gen_by_pair = False
    if dn_attrs[4][0] == 'p':
        nps = int(dn_attrs[4][1:])
    else:
        known_pairs_ratio, repeated_comps = dn_attrs[4][0:].split('k')
        known_pairs_ratio = float(known_pairs_ratio)  # d/n sampling probability
        repeated_comps = int(repeated_comps)  # k number of repeated comparison
        gen_by_pair = True
    data_kwarg = {
        'data_seed': seed,
        'beta_gen_params': beta_gen_params,
        'beta_gen_func': beta_gen_func,
        'n_items': ni,
        'n_judges': nj,
        'n_pairs': nps,
        'visualization': 0,
        'save_path': save_path,
        'known_pairs_ratio': known_pairs_ratio,
        'repeated_comps': repeated_comps,
        'gen_by_pair': gen_by_pair,
        's_true': s_true,
        'beta_true': beta_true,
    }
    data_pack = generate_data(**data_kwarg)
    return data_pack, data_kwarg


def run_algo(data_pack, data_kwarg, algo_name, seed):
    an = algo_name.split('-')
    algo = an[0]
    # algo_lookup = {
    #     'btl': 'simple',
    #     'gbtl': 'individual',
    #     'gbtlinv': 'inverse',
    #     'gbtlneg': 'negative'
    # }
    # algo = algo_lookup[an[0]]

    init = an[1]
    # if 'spectral' in an[1]:
    #     init = 'spectral'
    # elif 'disturb' in an[1]:
    #     init = 'ground_truth_disturb'
    # else:
    #     init = an[1]
        
    ob = 'random_b' in an[1]
    opt = 'mle' in an[2]
    fs = 'fix^s' in an[1]

    if len(an) == 4:
        lr = float(an[3])
    else:
        lr = 1e-3
    
    algo_kwarg = {
        'init_seed': seed, 'init_method': init,
        'override_beta': ob, 'max_iter': 2000, 'lr': lr, 'lr_decay': True,
        'opt': opt, 'opt_func': 'SGD', 'opt_stablizer': 'decouple', 'opt_sparse': False, 'fix_s': fs,
        'debug': 1, 'verbose': 0, 'algo': algo,
    }

    config = Dict(algo_kwarg, data_kwarg)
    config.ground_truth_prob_mat = False
    config.popular_correction = False
    config.grad_method = 'auto'
    config.normalize_gradient = True
    config.GPU = True
    config.linesearch = False
    config.prob_regularization = True
    config.err_const = 10e-23
    res_pack = make_estimation(data_pack, config)

    cb = {**data_kwarg, **algo_kwarg, **res_pack}
    return cb


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