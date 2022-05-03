import os
import copy
import numpy as np
from PIL import Image
import glob
import pickle
import matplotlib.pyplot as plt


class ReadImgDir:
    def __init__(self,
                 path_dir: str,
                 flag_crop: bool = False,
                 size_crop: int = 32,
                 flag_resize: bool = False,
                 scale_resize: float = 0.25):
        self.fname_list_img = glob.glob(path_dir)
        self.fname_list_img.sort()
        self.num_img = len(self.fname_list_img)
        self.flag_crop = flag_crop
        self.size_crop = size_crop
        self.flag_resize = flag_resize
        self.scale_resize = scale_resize
        self.imgs = []
        self.imgs_float = []
        self.labels = []
        print('num of images is ' + str(self.num_img))
        self.read_imgs_fromdir()

    def read_imgs_fromdir(self):

        for i_list in range(self.num_img):
            print(self.fname_list_img[i_list])
            img = Image.open(self.fname_list_img[i_list])
            img = np.array(img)
            if self.flag_crop:
                mask_crop = np.zeros(img.shape)
                pos_center = [img.shape[0] / 2, img.shape[1] / 2]
                mask_crop[round(pos_center[0] - (self.size_crop / 2)):round(pos_center[0] + (self.size_crop / 2)),
                round(pos_center[1] - (self.size_crop / 2)):round(pos_center[1] + (self.size_crop / 2))] = 1
                img = img * mask_crop
            img = Image.fromarray(img.astype('uint8'))

            if self.flag_resize:
                width, height = img.size
                img = img.resize((int(width*self.scale_resize), int(height*self.scale_resize)))

            self.imgs.append(img)
            self.labels.append(os.path.basename(self.fname_list_img[i_list]))



    def cal_mean_std(self):

        mean_img = []
        std_img = []
        if len(img.shape) == 2:
            mean_img.append(np.mean(img))
            std_img.append(np.std(img))
        else:
            mean_img.append(np.mean(img, 2))
            std_img.append(np.std(img, 2))

    def divide_imgs_col(self, num_divide: int = 4, flag_dif: bool = True):
        # this divide each image in the horizontal direction with the number of num_divide.
        # this function will update self.imgs and self.num_img
        # for a specific project I put a function to take a difference from the first image in each cutting.
        imgs = []
        labels = []
        size_img = np.array(self.imgs[0]).shape
        cut_shape = int(size_img[1]/num_divide)
        count = 0
        for l, img in enumerate(self.imgs):
            img = np.array(img)
            img = img.astype('float')
            for i in range(num_divide):
                img_div = copy.deepcopy(img[:, i * cut_shape: i * cut_shape + cut_shape, :])
                if flag_dif:
                    if i == 0:
                        img_1 = copy.deepcopy(img_div)
                    else:
                        count = count + 1
                        img_tmp = img_div - img_1 + 128
                        self.imgs_float.append(img_tmp)
                        img_tmp[np.where(img_tmp < 0)] = 0
                        img_tmp[np.where(img_tmp > 255)] = 255
                        print('min: ' + str(np.min(img_tmp)) + 'max:' + str(np.max(img_tmp)))
                        img_tmp = img_tmp.astype('uint8')
                        img_tmp = Image.fromarray(img_tmp)
                        imgs.append(img_tmp)
                        label, ext  = os.path.splitext(self.labels[l])
                        labels.append(label + '_' + str(i) + ext)
                else:
                    img_tmp = img_tmp.astype('uint8')
                    img_div = Image.fromarray(img_div)
                    imgs.append(img_div)
                    label, ext = os.path.splitext(self.labels[l])
                    labels.append(label + '_' + str(i) + ext)

        self.imgs = imgs
        self.labels = labels
        self.num_img = self.num_img * num_divide

    def save_img(self, path_save_dir):
        for i, img in enumerate(self.imgs):
            label, ext = os.path.splitext(self.labels[i])
            img.save(path_save_dir+label+'.png', format='PNG')

    def crop_img(self, img, size_crop):
        h_size_crop = int(size_crop/2)
        size_img = img.shape
        pos_center_x = int(size_img[1]/2)
        pos_center_y = int(size_img[0]/2)
        img = img[pos_center_y-h_size_crop:pos_center_y+h_size_crop,
                            pos_center_x-h_size_crop:pos_center_x+h_size_crop,:]
        return img

    def crop_imgs(self, size_crop):
        h_size_crop = int(size_crop/2)
        size_img = self.imgs_float[0].shape
        pos_center_x = int(size_img[1]/2)
        pos_center_y = int(size_img[0]/2)
        self.imgs = []
        imgs_new_float = []
        for img in self.imgs_float:
            tmp = img[pos_center_y-h_size_crop:pos_center_y+h_size_crop,
                            pos_center_x-h_size_crop:pos_center_x+h_size_crop,:]
            imgs_new_float.append(tmp)
            tmp = tmp.astype('uint8')
            tmp = Image.fromarray(tmp)
            self.imgs.append(tmp)
        self.imgs_float = imgs_new_float


if __name__ == '__main__':
    path = '../layer6-8/*.jpg'
    path_save = '../img_dif/'
    path_save2 = '../img_dif_crop/crop_'

    fname_save = 'imgs_layer6-8.pickle'

    size_crop = 256

    list_imgs = ReadImgDir(path, flag_resize = False,
                 scale_resize = 0.5)

    #divide each image
    #list_imgs.divide_imgs_col()
    #list_imgs.save_img(path_save2)

    #crop image
    #list_imgs.crop_imgs(size_crop)
    #list_imgs.save_img(path_save2)
    with open(path_save + fname_save, mode='wb') as f:
            pickle.dump(list_imgs, f)



#    with open(path_save + fname_save, mode='rb') as f:
#            list_imgs = pickle.load(f)

