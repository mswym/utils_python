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
                 size_crop: int = 256,
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
        self.size_img = []
        self.labels = []
        print('num of images is ' + str(self.num_img))
        #self.read_imgs_fromdir()

    def read_imgs_fromdir(self):

        for i_list in range(self.num_img):
            print(self.fname_list_img[i_list])
            img = Image.open(self.fname_list_img[i_list])

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

    def divide_imgs_col(self, num_divide: int = 4, flag_dif: bool = True, path_save_dir: str =''):
        # this divide each image in the horizontal direction with the number of num_divide.
        # this function will update self.imgs and self.num_img
        # for a specific project I put a function to take a difference from the first image in each cutting.
        # imgs = []
        labels = []

        for label in self.fname_list_img:
            img = Image.open(label)
            if self.flag_resize:
                width, height = img.size
                img = img.resize((int(width*self.scale_resize), int(height*self.scale_resize)))
            width, height = img.size
            cut_shape = int(width/num_divide)
            img = np.array(img)
            img = img.astype('float')
            for i in range(num_divide):
                img_div = copy.deepcopy(img[:, i * cut_shape: i * cut_shape + cut_shape, :])
                if flag_dif:
                    if i == 0:
                        img_1 = copy.deepcopy(img_div)
                    else:
                        img_tmp = img_div - img_1 + 128
                        self.imgs_float.append(img_tmp)
                        print('min: ' + str(np.min(img_tmp)) + 'max:' + str(np.max(img_tmp)))
                        img_tmp[np.where(img_tmp < 0)] = 0
                        img_tmp[np.where(img_tmp > 255)] = 255
                        img_tmp = img_tmp.astype('uint8')
                        img_tmp = Image.fromarray(img_tmp)
                        #imgs.append(img_tmp)
                        label_body, ext  = os.path.splitext(os.path.basename(label))
                        labels.append(label_body + '_' + str(i) + ext)
                        self.save_img(path_save_dir, img_tmp, label_body + '_' + str(i) + ext)
                else:
                    img_tmp = img_tmp.astype('uint8')
                    img_tmp = Image.fromarray(img_tmp)
                    #imgs.append(img_div)
                    label_body, ext = os.path.splitext(os.path.basename(label))
                    labels.append(label_body + '_' + str(i) + ext)
                    self.save_img(path_save_dir, img_tmp, label_body + '_' + str(i) + ext)

        #self.imgs = imgs
        self.size_img = np.array(img_tmp).shape
        self.labels = labels
        self.num_img = self.num_img * num_divide

    def save_img(self, path_save_dir, img, label):
        label, ext = os.path.splitext(label)
        img.save(path_save_dir+label+'.png', format='PNG')

    def save_imgs(self, path_save_dir):
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

    def open_img(self, path, label):
        label_body, ext = os.path.splitext(label)
        return Image.open(path + label_body + '.png')

    def crop_imgs(self, size_crop, path, path_save):
        h_size_crop = int(size_crop/2)
        pos_center_x = int(self.size_img[1]/2)
        pos_center_y = int(self.size_img[0]/2)
        for label in self.labels:
            print(label)
            img = self.open_img(path, label)
            img = img.crop((pos_center_x-h_size_crop,pos_center_y-h_size_crop,
                     pos_center_x+h_size_crop,pos_center_y+h_size_crop))
            self.save_img(path_save, img, label)

    def cut_imgs(self, size_cut, path_save):

        imgs_cut = []
        num_height = int(list_imgs.size_img[0]/size_cut)
        num_width = int(list_imgs.size_img[1] / size_cut)
        count = 0
        for img in self.imgs_float:
            for h in range(num_height):
                for w in range(num_width):
                    count += 1
                    tmp = img[h*size_cut:h*size_cut+size_cut,
                            w*size_cut:w*size_cut+size_cut, :]
                    imgs_cut.append(tmp)
                    tmp = tmp.astype('uint8')
                    tmp = Image.fromarray(tmp)
                    self.save_img(path_save, tmp, str(count))
        self.imgs_float = imgs_cut

if __name__ == '__main__':
    path = '../layer6-8/*.jpg'
    path_save = '../img_dif170/'
    path_save3 = '../img_dif_cut/cut_'

    fname_save = 'class_layer6-8.pickle'
    fname_save2 = 'imgs_layer6-8_170.pickle'

    size_cut = 16

    list_imgs = ReadImgDir(path, flag_resize = True,
                 scale_resize = 0.166)

    #divide each image
    list_imgs.divide_imgs_col(path_save_dir=path_save)

    with open(path_save + fname_save, mode='wb') as f:
        pickle.dump(list_imgs, f)

    with open(path_save + fname_save2, mode='wb') as f:
        pickle.dump(list_imgs.imgs_float, f)


    #with open(path_save + fname_save, mode='rb') as f:
    #    list_imgs = pickle.load(f)

    list_imgs.cut_imgs(size_cut, path_save3)

    with open(path_save3 + fname_save, mode='wb') as f:
        pickle.dump(list_imgs, f)

    with open(path_save3 + fname_save2, mode='wb') as f:
        pickle.dump(list_imgs.imgs_float, f)




