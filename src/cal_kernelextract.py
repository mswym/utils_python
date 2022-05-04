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
    path = '../img_dif_cut/cut_imgs_layer6-8_256.pickle'
    num_components = 25

    with open(path, mode='rb') as f:
        imgs = pickle.load(f)

    imgs = np.array(imgs)

    size_imgs = imgs.shape

    #reshape
    imgs = np.reshape(imgs,[size_imgs[0], size_imgs[1]*size_imgs[2]*size_imgs[3]])
    #normalize
    mean_imgs = np.mean(imgs)
    std_imgs = np.std(imgs)
    imgs_norm = (imgs - mean_imgs)/std_imgs

    #whitenening
    pca = PCA(num_components)
    pca.fit(imgs_norm)

    print('cumulated percentage is...')
    print(np.cumsum(pca.explained_variance_ratio_))

    print(pca.components_.shape) #25*768
    imgs_white = np.dot(pca.components_, imgs_norm.T)

    ica = FastICA(num_components, max_iter=1000)
    ica.fit(imgs_white.T)
    Uica = ica.mixing_
    print(Uica.shape) #pcanum x icanum

    kernel = np.dot(pca.comsponents_.T,Uica[:, 0])
    kernel = np.reshape(kernel,[size_imgs[1], size_imgs[2], size_imgs[3]])
    kernel = (kernel*std_imgs)+mean_imgs