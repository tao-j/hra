import numpy as np
import time

#from bokeh.plotting import *
import matplotlib.pyplot as plt

import cv2
import PIL
from IPython import display

from addict import Dict

def generate_data(data_seed=None, 
                  n_items=10, n_judges=10, n_pairs=200,
                  shrink_b=30, beta_gen_func='shrink',
                  visualization=False):
    
    if not data_seed:
        data_seed = int(time.time() * 10e7) % 2**32
    np.random.seed(data_seed)
    
    s = np.sort(np.random.normal(loc=1.0, size=n_items))
    s -= s[0]
    s /= s.sum()
    print('ground truth s', s)
    if beta_gen_func == 'manual':
        assert len(shrink_b) == n_judges
        betas = np.array(shrink_b)
    if beta_gen_func == 'shrink':
        betas = np.random.random(size=n_judges) / shrink_b
    elif beta_gen_func == 'power':
        betas = np.power(shrink_b, -1. * np.arange(0, n_judges))
    elif beta_gen_func == 'xi':
        assert shrink_b < len(betas)
        betas = [0.00001, 0.0001, 0.0002, 0.0005,
                 0.001, 0.005,
                 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                 0.1, 0.5, 1.0]
        betas = np.ones(n_judges) * betas[shrink_b]
    elif beta_gen_func == 'beta':
        betas = np.random.beta(shrink_b[0], shrink_b[1], size=n_judges)
    elif beta_gen_func == 'negative':
        n_positive = np.ceil(n_judges * 0.7)
        n_negative = n_judges - n_positive
        betas = np.array([shrink_b] * n_positive + [shrink_b] ** n_negative)
    print('ground truth beta', betas)

    # gumble distribution
    def gb_cdf(x, beta):
        mu = -0.5772*beta
        return np.exp(-np.exp(-(x-mu)/beta) )

    def gb_pdf(x, beta):
        mu = -0.5772*beta
        return 1. / beta * np.exp(-(x-mu)/beta - np.exp(-(x-mu)/beta) )

    # ts = np.arange(-.618, .618, 0.01)
    # for beta_i in betas:
    #     ty = gb_pdf(ts, beta_i)
    #     plt.plot(ts, ty)
    #     print(ty.sum()/100.)
    # plt.show()

    data = []
    judge_imgs = []
    total_img = np.zeros((3, n_items, n_items))

    for k, beta_i in enumerate(betas):
        data_img = np.zeros((n_items, n_items))
        for _ in range(n_pairs):
            # ensure that generate two different items for comparison
            i = 0
            j = 0
            while i == j:
                i = np.random.randint(0, len(s)) # try normal dist.
                j = np.random.randint(0, len(s))

            s_i = s[i] + np.random.gumbel(-0.5772*beta_i, beta_i)
            s_j = s[j] + np.random.gumbel(-0.5772*beta_i, beta_i)
            if s_i > s_j:
                data.append((i, j, k))
                data_img[i][j] += 1. # rgb
            else:
                data.append((j, i, k))
                data_img[j][i] += 1. # rgb
        total_img[2] += data_img
        judge_imgs.append(data_img)
        
    judge_imgs.append(total_img.transpose(1, 2, 0))
    
    data_pack = Dict()
    data_pack.n_items = n_items
    data_pack.n_judges = n_judges
    data_pack.n_pairs = n_pairs
    data_pack.s = s
    data_pack.betas = betas    
    if visualization:
        show_images(judge_imgs, cols=1)
        print(data_pack, len(data))
    data_pack.data = data
    return data_pack


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
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
    plt.show()
