import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# function to get a list of file of a given extension, both the absolute path and the filename
def get_file_list(path, ext='', queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]


def get_data():
    tra_img_dir = './data/DRIVE/training/images'
    tra_msk_dir = './data/DRIVE/training/mask'
    tra_lbl_dir = './data/DRIVE/training/1st_manual'

    tra_imgs = sorted(get_file_list(tra_img_dir, 'tif')[0])
    tra_msks = sorted(get_file_list(tra_msk_dir, 'gif')[0])
    tra_lbls = sorted(get_file_list(tra_lbl_dir, 'gif')[0])

    return tra_imgs, tra_msks, tra_lbls


def show_image(idx, imgs, msks, lbls):
    img = np.asarray(Image.open(imgs[idx]))
    msk = np.asarray(Image.open(msks[idx]))
    lbl = np.asarray(Image.open(lbls[idx]))
    img_g = img[:,:,1].squeeze().astype(float)
    plt.subplot(1,3,1)
    plt.imshow(img); plt.title('RGB image {}'.format(idx+1))
    plt.subplot(1,3,2)
    plt.imshow(msk, cmap='gray'); plt.title('Mask {}'.format(idx+1))
    plt.subplot(1,3,3)
    plt.imshow(lbl, cmap='gray'); plt.title('Manual annotation {}'.format(idx+1))
    plt.show()

