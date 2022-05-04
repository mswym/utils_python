import os
import copy
import numpy as np
from PIL import Image
import glob
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA, DictionaryLearning

from load_imgs import *

def sample_roi(imgs, size_cut, num_patches_per_img):

    roi = []
    for ind_img in range(imgs.shape[0]):
        for ind_patches in range(num_patches_per_img):
            x_start = np.random.randint(0, imgs.shape[2] - size_cut - 1)
            y_start = np.random.randint(0, imgs.shape[1] - size_cut - 1)
            roi.append(np.reshape(imgs[ind_img, y_start: y_start + size_cut, 
                                  x_start: x_start + size_cut, :],
                                  [size_cut, size_cut, 3]))
    return np.array(roi)

def cal_norm(imgs):
    mean_imgs = np.mean(imgs)
    std_imgs = np.std(imgs)
    imgs_norm = (imgs - mean_imgs)/std_imgs

    return imgs_norm, mean_imgs, std_imgs

def cal_ica(imgs, num_components):
    ica = FastICA(num_components)
    ica.fit(imgs)

    return ica

def cal_dictionary(imgs, num_components):
    dict_learner = DictionaryLearning(n_components=num_components, transform_algorithm='lasso_lars')
    dict_learner.fit(imgs)

    return ica

def cal_sequential_sample(path):
    with open(path, mode='rb') as f:
        imgs = pickle.load(f)

    imgs = np.array(imgs)
    imgs = imgs[0:1000, :, :, :]
    size_imgs = imgs.shape

if __name__ == '__main__':
    path = '../img_dif170/imgs_layer6-8_170.pickle'
    num_components = 100
    num_patches_per_img = 1
    size_cut = 16

    with open(path, mode='rb') as f:
        imgs = pickle.load(f)
        imgs = np.array(imgs)

    imgs = sample_roi(imgs, size_cut, num_patches_per_img)
    size_imgs = imgs.shape
    #reshape
    imgs = np.reshape(imgs,[size_imgs[0], size_imgs[1]*size_imgs[2]*size_imgs[3]])
    #normalize
    imgs_norm, mean_imgs, std_imgs = cal_norm(imgs)

    #ica
    ica = cal_ica(imgs_norm, num_components)
    # sparse = cal_dictionary(imgs_norm, num_components)
    #reshape and rescale
    kernels = np.reshape(ica.components_,[num_components, size_imgs[1], size_imgs[2], size_imgs[3]])
    kernels = kernels - np.min(kernels)
    kernels = kernels/np.max(kernels)
