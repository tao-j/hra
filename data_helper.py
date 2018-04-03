import numpy as np

#from bokeh.plotting import *
import matplotlib.pyplot as plt

import cv2
import PIL
from IPython import display

import multiprocessing as mp

def generate_data(data_seed=3838, n_items=10, n_judges=10, n_pairs=200, shrink_b=30, beta_gen_func='shrink'):
    np.random.seed(data_seed)
    s = np.sort(np.random.normal(loc=1.0, size=n_items))
    s -= s[0]
    s /= s.sum()
    print('ground truth s', s)
    # betas = np.random.beta(beta_a, beta_b, size=n_judges)
    if beta_gen_func == 'shrink':
        betas = np.random.random(size=n_judges) / shrink_b
    elif beta_gen_func == 'power':
        betas = np.power(shrink_b, -1. * np.arange(1, 1+n_judges))
    elif beta_gen_func == 'x':
        betas = np.ones(n_judges) * shrink_b
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
            i = np.random.randint(0, len(s)) # try normal
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
    show_images(judge_imgs, cols=1)
    print(len(data), n_items, n_judges, n_pairs)
    return [data, n_items, n_judges, n_pairs, s, betas]

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