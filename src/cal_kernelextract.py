import os
import copy
import numpy as np
from PIL import Image
import glob
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA

from load_imgs import *

def cal_white_pca(imgs):

    return imgs

if __name__ == '__main__':
    path = '../img_dif/imgs_layer6-8.pickle'

    with open(path, mode='rb') as f:
        list_imgs = pickle.load(f)


    imgs = list_imgs.imgs_float
    imgs = np.array(imgs)
    size_imgs = imgs.shape

    #reshape
    imgs = np.reshape(imgs,[size_imgs[0], size_imgs[1]*size_imgs[2]*size_imgs[3]])
    #normalize
    imgs_norm = (imgs - np.mean(imgs))/np.mean(imgs)

    #whitenening

    pca = PCA(25)
    pca.fit(imgs_norm)
