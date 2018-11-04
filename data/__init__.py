import numpy as np
import time
from data.readinglevel import *
from addict import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_betas(betas):
    # gumbel distribution
    def gb_cdf(x, beta):
        mu = -0.5772*beta
        return np.exp(-np.exp(-(x-mu)/beta) )

    def gb_pdf(x, beta):
        mu = -0.5772*beta
        return 1. / beta * np.exp(-(x-mu)/beta - np.exp(-(x-mu)/beta) )

    # plot pdf
    ts = np.arange(-.618, .618, 0.01)
    for beta_i in betas:
        ty = gb_pdf(ts, beta_i)
        plt.plot(ts, ty)
        print(ty.sum()/100.)
    plt.show()


def generate_by_pair(s_true, beta_true, n_pairs):
    str = 0


def generate_by_ratio(s_true, beta_true, repeated_comps, known_pairs_ratio):
    str = 0


def gen_s(s_gen_func, n_items, s_true):
    if s_gen_func == 'ra':  # 'random':
        s_true = np.sort(np.random.normal(loc=1.0, size=n_items))
    elif s_gen_func == 'ge':  # 'geometric':
        po = np.arange(1., 2 * n_items + 1, 2.) - n_items
        po = po / 2 / n_items
        s_true = np.log(np.power(10, po))
    elif s_gen_func == 'ar':  # 'arithmetic':
        s_true = np.arange(n_items) * (1. / n_items)
    elif s_gen_func == 'pa':  # 'passed-in':
        if not s_true:
            raise ValueError
    else:
        raise NotImplementedError
    return s_true


def gen_betas(beta_gen_func, beta_gen_params, n_judges, beta_true):
    if beta_gen_func == 'ma':  # 'manual':
        assert len(beta_gen_params) == n_judges
        beta_true = np.array(beta_gen_params)
    if beta_gen_func == 'sh':  # 'shrink':
        beta_true = np.random.random(size=n_judges) / beta_gen_params
    if beta_gen_func == 'ex':  # exponential:
        beta_true = np.random.exponential(beta_gen_params, size=n_judges)
    elif beta_gen_func == 'po':  # 'power':
        beta_true = np.power(beta_gen_params, -1. * np.arange(0, n_judges))
    elif beta_gen_func == 'be':  # 'beta':
        beta_true = np.random.beta(beta_gen_params[0], beta_gen_params[1], size=n_judges)
    elif beta_gen_func == 'ga':  # 'gamma':
        beta_true = np.random.gamma(beta_gen_params[0], beta_gen_params[1], size=n_judges)
    elif beta_gen_func == 'ne':  # 'negative':
        n_positive = min(int(np.ceil(n_judges * 0.7)), n_judges - 1)
        n_negative = n_judges - n_positive
        beta_true = np.array([beta_gen_params] * n_positive + [-beta_gen_params] * n_negative)
    elif beta_gen_func == 'pa':  # 'passed-in':
        if not beta_true:
            raise ValueError
    else:
        print('ERR beta_gen_func ', beta_gen_func)
        raise NotImplementedError
    return beta_true


def adjust_ground_truth(s_true, beta_true):
    # # TODO: move this part to estimation rather than generation
    # to make estimation easier, since we are using the first beta to rescale the params
    # this value shouldn't be too large or small. using mode.
    # just a test, not used now
    # beta_idx = np.argsort(np.abs(beta_true))
    # ii = beta_idx[n_judges // 2]
    # beta_true[ii], beta_true[0] = beta_true[0], beta_true[ii]

    # rescale to make s_0 = 0, beta_0 = 1, for easy comparison between experiments
    s_true -= s_true[0]
    # s_true /= s_true.sum()  # not used anymore, it will result equivalent scale of beta, but not compensated
    s_ratio = 1. / beta_true[0]
    s_true = s_true * s_ratio
    beta_true = beta_true * s_ratio

    # randomly set 1/3 to negative beta
    # beta_true[0:len(beta_true)//3] *= -1.

    # print('after adjustment ', '\ns', s_true, '\nbeta', beta_true)
    return s_true, beta_true


def sample_one_pair(s_true, beta_k, i, j, k, count_mat):
    if beta_k < 0:
        s_j = s_true[i] + np.random.gumbel(0.5772 * beta_k, -beta_k)
        s_i = s_true[j] + np.random.gumbel(0.5772 * beta_k, -beta_k)
    else:
        s_i = s_true[i] + np.random.gumbel(-0.5772 * beta_k, beta_k)
        s_j = s_true[j] + np.random.gumbel(-0.5772 * beta_k, beta_k)
    if s_i > s_j:
        count_mat[k][j][i] += 1.
    else:
        count_mat[k][i][j] += 1.


def visualize_count_mat(count_mat, save_path, s_true=None, sort_mat=False):
    n_items = count_mat.shape[1]
    # sort matrix columns and rows so that in increasing order to be inspected by visualizer
    if sort_mat:
        assert s_true is not None
        s_seq = np.argsort(s_true)
        s_true = s_true[s_seq]
        new_count_mat = np.zeros(count_mat.shape)
        for k in range(count_mat.shape[0]):
            for j in range(count_mat.shape[1]):
                seqq = s_seq[j]
                for i in range(count_mat.shape[1]):
                    if count_mat[k][seqq][i]:
                        new_count_mat[k][j][i] = count_mat[k][seqq][i]
        count_mat *= 0.
        for k in range(count_mat.shape[0]):
            for j in range(count_mat.shape[1]):
                seqq = s_seq[j]
                for i in range(count_mat.shape[1]):
                    if new_count_mat[k][i][seqq]:
                        count_mat[k][i][j] = new_count_mat[k][i][seqq]

    population_img = np.zeros((3, n_items, n_items), dtype=np.double)
    population_img[2] += count_mat

    individual_imgs = list()
    for k in range(count_mat.shape[0]):
        individual_imgs.append(count_mat[k] / np.max(np.max(count_mat[k])))
    individual_imgs.append(population_img.transpose(1, 2, 0) / np.max(np.max(population_img)))

    show_images(individual_imgs[:25], cols=5, save_path=save_path)


def show_images(images, cols=1, titles=None, save_path=None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_data(data_seed=None,
                  n_items=10, n_judges=10, n_pairs=200,
                  beta_gen_params=30, beta_gen_func='sh', s_gen_func='ar',
                  visualization=False, save_path=None,
                  gen_by_pair=False, known_pairs_ratio=0.1, repeated_comps=32,
                  beta_true=None, s_true=None):

    if not data_seed:
        data_seed = int(time.time() * 10e7) % 2**32
    np.random.seed(data_seed)

    data_cnt = None
    if beta_gen_func == 'ds':
        ds = conv_data.ReadingLevelDataset()
        count_mat = ds.count_mat
        s_true = ds.s_true
        data_cnt = ds.data_cnt

        n_pairs = np.sum(np.sum(np.sum(count_mat)))
        n_items = count_mat.shape[1]
        n_judges = count_mat.shape[0]
        beta_true = np.ones(n_judges)
    else:
        s_true = gen_s(s_gen_func, n_items, s_true)
        beta_true = gen_betas(beta_gen_func, beta_gen_params, n_judges, beta_true)
        # print('ground truth s', s_true)
        # print('ground truth beta', beta_true)
        if not isinstance(s_true, np.ndarray):
            s_true = np.array(s_true)
        if not isinstance(beta_true, np.ndarray):
            beta_true = np.array(beta_true)

        s_true, beta_true = adjust_ground_truth(s_true, beta_true)

        # known_pairs_ratio = 0.1  # d/n sampling probability
        # repeated_comps = 32  # k number of repeated comparison
        assert not (gen_by_pair and n_pairs == -1)

        # TODO: np.ones is could act as a regularization, or 1.0 0.1 0.01, also have this approx in algo_func.py
        count_mat = np.zeros((n_judges, n_items, n_items), dtype=np.double)
        for k, beta_k in enumerate(beta_true):

            if gen_by_pair:
                # same pair repeat
                for i in range(n_items):
                    for j in range(n_items):
                        if i == j:
                            continue
                        # TODO: this is for each judge providing the ratio of inputs
                        if np.random.random() <= known_pairs_ratio:
                            for _ in range(repeated_comps):
                                sample_one_pair(s_true, beta_k, i, j, k, count_mat)
                                n_pairs += 1
            else:
                # random pair
                for _ in range(n_pairs):
                    # ensure that generate two different items for comparison
                    i = 0
                    j = 0
                    while i == j:
                        i = np.random.randint(0, len(s_true))  # TODO: can try other dist.
                        j = np.random.randint(0, len(s_true))
                        sample_one_pair(s_true, beta_k, i, j, k, count_mat)

    data_pack = Dict()
    data_pack.n_items = n_items
    data_pack.n_judges = n_judges
    data_pack.n_pairs = n_pairs
    data_pack.s_true = s_true
    data_pack.beta_true = beta_true
    data_pack.count_mat = count_mat
    data_pack.data_cnt = data_cnt

    if visualization:
        visualize_count_mat(count_mat, save_path)

    return data_pack
