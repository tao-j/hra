from algo.ra.ra import make_estimation
from data import generate_data
from ifce.eval import get_eval

from addict import Dict
import json

# algorithm settings string definition
# beta_gen_func-samole_beta_distribution-j_#judges-i_#items
# algo-init_method-optmethod-lr


def parse_data_name(data_name, seed, save_path=None, s_true=None, beta_true=None):
    dn_attrs = data_name.split('-')

    beta_gen_params_str = list(map(float, dn_attrs[1].replace('_', '-').split(',')))
    beta_gen_func = dn_attrs[0]
    dn_lookup = {
        'po': ('power', beta_gen_params_str[0]),
        'be': ('beta', beta_gen_params_str),
        'ga': ('gamma', beta_gen_params_str),
        'sh': ('shrink', beta_gen_params_str[0]),
        'ex': ('ex', beta_gen_params_str[0]),
        'ne': ('negative', beta_gen_params_str[0]),
        'ma': ('manual', beta_gen_params_str),
        'pa': ('passed-in', beta_gen_params_str),
        'ds': ('dataset', beta_gen_params_str)
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
    return data_kwarg


def parse_algo_name(algo_name, seed):
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
        'override_beta': ob, 'max_iter': 10, 'lr': lr, 'lr_decay': True,
        'opt': opt, 'opt_func': 'minfunc', 'opt_stablizer': 'decouple', 'opt_sparse': False, 'fix_s': fs,
        'debug': 1, 'verbose': 0, 'algo': algo,
    }
    return algo_kwarg


class Job:
    def __init__(self, load, data_name=None, algo_name=None, seed=None,
                s_true=None, beta_true=None, save_path=None, json_str=None):

        self.data_pack = None
        self.config = None
        self.res_pack = None
        self.st = ''
        self.dt = ''
        if not load:
            self.init_gen(data_name, algo_name, seed, s_true, beta_true, save_path)
        else:
            self.init_json(json_str)

    def init_gen(self, data_name=None, algo_name=None, seed=None,
                s_true=None, beta_true=None, save_path=None):
        data_kwarg = parse_data_name(data_name, seed, save_path, s_true, beta_true)
        algo_kwarg = parse_algo_name(algo_name, seed)

        config = Dict(algo_kwarg, data_kwarg)
        config.ground_truth_prob_mat = False
        config.popular_correction = False
        config.grad_method = 'auto'
        config.normalize_gradient = True
        config.GPU = False
        config.linesearch = False
        config.prob_regularization = True
        config.err_const = 10e-23
        # config.backend = 'torch'
        config.backend = 'numpy'

        self.data_pack = generate_data(**data_kwarg)
        self.config = config
        self.res_pack = None

    def init_json(self, json_str=""):
        j = json.loads(json_str)
        self.config = Dict(j['config'])
        self.res_pack = Dict(j['res_pack'])
        self.data_pack = Dict(j['data_pack'])
        self.st = Dict(j['st'])
        self.dt = Dict(j['dt'])

    def serialize(self, st, dt):
        j = Dict()
        j.config = self.config.to_dict()
        j.config['s_true'] = None
        j.config['beta_true'] = None
        j.data_pack.s_true = self.data_pack.s_true.tolist()
        j.data_pack.beta_true = self.data_pack.beta_true.tolist()
        # don't need count mat
        j.res_pack = self.res_pack
        j.res_pack.s_est = self.res_pack.s_est.tolist()
        j.res_pack.beta_est = self.res_pack.beta_est.tolist()
        j.st = st
        j.dt = dt
        return json.dumps(j)

    def run_algo(self):
        self.res_pack = make_estimation(self.data_pack, self.config)

    def eval(self):
        return get_eval(self.data_pack, self.res_pack)
