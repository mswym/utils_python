import os
import copy
import numpy as np
from PIL import Image
import glob
import pickle
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.decomposition import PCA, FastICA, DictionaryLearning

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

    return dict_learner

def cal_sequential_sample(path):
    with open(path, mode='rb') as f:
        imgs = pickle.load(f)

    imgs = np.array(imgs)
    imgs = imgs[0:1000, :, :, :]
    size_imgs = imgs.shape

def make_panels(model, num_components, size_imgs):
    kernels = np.reshape(model.components_, [num_components, size_imgs[1], size_imgs[2], size_imgs[3]])
    kernels = kernels - np.min(kernels)
    kernels = kernels / np.max(kernels)
    kernels = 128*(kernels - np.mean(kernels)) + 128
    kernels[np.where(kernels<0)] = 0
    kernels[np.where(kernels>255)] = 255
    kernels = kernels.astype('uint8')
    kernels = kernels.transpose(0, 3, 1, 2)
    kernels = torch.from_numpy(kernels)

    kernels = torchvision.utils.make_grid(kernels, nrow=10)
    kernels = kernels.to('cpu').detach().numpy().transpose(1, 2, 0).copy()

    return kernels

if __name__ == '__main__':
    path = '../img_dif340/imgs_layer6-8_340.pickle'
    num_components = 100
    num_patches_per_img = 10 #8000 in total
    size_cut = 32
    type_model = 'ica'
    fname_save = type_model + 'output_340.png'

    with open(path, mode='rb') as f:
        imgs = pickle.load(f)
        imgs = np.array(imgs)

    imgs = sample_roi(imgs, size_cut, num_patches_per_img)
    size_imgs = imgs.shape
    #reshape
    imgs = np.reshape(imgs,[size_imgs[0], size_imgs[1]*size_imgs[2]*size_imgs[3]])
    #normalize
    imgs_norm, mean_imgs, std_imgs = cal_norm(imgs)

    if type_model=='ica':
        model = cal_ica(imgs_norm, num_components)
    elif type_model=='sparse':
        model = cal_dictionary(imgs_norm, num_components)
    #reshape and rescale
    panels = make_panels(model)

    panels = panels.astype('uint8')
    panels = Image.fromarray(panels)
    panels.save(fname_save)

