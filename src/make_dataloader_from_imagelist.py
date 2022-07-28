from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#prepared this for TensorFlow Hub demos because I could not use tf.data.Dataset

def make_ds_from_imglist(path_imglist, size_batch, size_img=None):
  imgs = []
  batch = []

  for i,path in enumerate(path_imglist):
    img = tf.keras.utils.load_img(
      path,
      grayscale=False,
      color_mode='rgb',
      target_size=size_img,
      interpolation='nearest'
    )

    batch.append(np.array(img).astype('float32'))

    if i==len(path_imglist)-1 or len(batch) == size_batch:
      batch = np.array(batch) / (255 / 2) - 1  # 0-255 to -1 to 1 #for pretrained bigGAN
      imgs.append(batch)
      batch = []

    ##only available for eager mode
    # dataset = tf.data.Dataset.from_tensor_slices(imgs)
    # dataset = dataset.batch(size_batch)
    # list(dataset.as_numpy_iterator())

  return imgs

def make_ds_from_veclist(path_veclist, size_batch):
  vecs = []
  batch = []

  for i,vec in enumerate(path_veclist):
    batch.append(np.array(vec).astype('float32'))

    if i==len(path_veclist)-1 or len(batch) == size_batch:
      batch = np.array(batch) / (255 / 2) - 1  # 0-255 to -1 to 1 #for pretrained bigGAN
      vecs.append(batch)
      batch = []

  return vecs

if __name__ == '__main__':
  size_img = (256,256)
  size_batch = 10