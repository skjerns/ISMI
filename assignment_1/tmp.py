# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from PIL import Image
import os
import dicom
from IPython import display
import time
import sys
from mpl_toolkits.mplot3d import Axes3D
import copy
matplotlib.rcParams['figure.figsize'] = (20, 12)
import scipy.signal
# function that reads all files in a directory
def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]
# load the first scan from LIDC-IDRI
scan_path = './data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'

# read all dicom files in the folder
dcm_files = sorted(get_file_list(scan_path, 'dcm')[0])
n_slices = len(dcm_files)

# read one slice and print dicom information
slice_idx = 0
ct_slice_dicom = dicom.read_file(dcm_files[slice_idx])


# convert the slice to a numpy array
ct_slice_numpy = ct_slice_dicom.pixel_array
dims = ct_slice_numpy.shape

# print scan information
print ('{} dicom files of dimension {} found'.format(n_slices, dims))
scan = np.zeros((dims[0], dims[1], n_slices))

for f in range(n_slices):
    ds = dicom.read_file(dcm_files[f])
    scan[:,:,ds.InstanceNumber-1] = ds.pixel_array
        
def gaussian_2d(sigma_mm, voxel_size):
    x_vec = np.arange(-sigma_mm*3.3,sigma_mm*3.3,voxel_size[0]) # *2.69 SD should cover most of the area
    y_vec = np.arange(-sigma_mm*3.3,sigma_mm*3.3,voxel_size[1])
    xx,yy = np.meshgrid(x_vec,y_vec)
    kernel = (1/(2*np.pi*sigma_mm**2)) * np.exp(-(xx**2+yy**2)/(2*sigma_mm**2))
    return kernel, xx, yy 

def laplacian_of_gaussian(g):
    gxx = np.gradient(np.gradient(g,axis=0),axis=0)
    gyy = np.gradient(np.gradient(g,axis=1),axis=1)
    LoG = gxx+gyy
    return LoG,gxx,gyy

# compute LoG
#g,x,y = gaussian_2d(6.0, [0.7, 0.7])
#assert(g is not None),"g can not be None"
#assert(x is not None),"x can not be None"
#assert(y is not None),"y can not be None"
#
#LoG,gxx,gyy = laplacian_of_gaussian(g)
#assert(LoG is not None),"LoG can not be None"
#assert(gxx is not None),"gxx can not be None"
#assert(gyy is not None),"gyy can not be None"
#
##visualize the filters
#plt.subplot(1,4,1)
#plt.imshow(g)
#plt.subplot(1,4,2)
#plt.imshow(gxx)
#plt.subplot(1,4,3)
#plt.imshow(gyy)
#plt.subplot(1,4,4)
#plt.imshow(LoG)
#plt.show()


def get_thorax(ct_slice_numpy):
    thorax = (ct_slice_numpy > 500)
    thorax = scipy.ndimage.morphology.binary_fill_holes(thorax)
    label, num_label = scipy.ndimage.label(thorax)
    size = np.bincount(label.ravel())
    biggest_label = size[1:].argmax() + 1
    thorax = (label == biggest_label)
    return thorax    
# implement trachea seed point detection

def trachea_one_slice(ct_slice):
    ct_slice[~get_thorax(ct_slice)]=0
    # apparently the trachea is from 10-25mm, so we take that/2 as our sigma
    sigmas = np.arange(5,14)                                                  
    xy_act = np.zeros((len(sigmas),3))
    for i in np.arange(len(sigmas)): 
        # voxel size is fixed in this example. We could also read it from the Dicom:{Pixel Spacing}.
        g,_,_ = gaussian_2d(sigmas[i],(0.7,0.7))                               
        LoG,_,_ = laplacian_of_gaussian(g) 
        # multiply by sigma^2 to achieve a normalized version of the laplacian
        LoG = LoG * sigmas[i]**2                                               
        # using 'same' to have same dimensions later
        conv = scipy.signal.fftconvolve(ct_slice, LoG*-1 , mode='same')    
        # get "highest" activation pixel, in our case of inverted LoG that is the min.
        activation = np.min(conv)                                              
        (x,y) = np.unravel_index(conv.argmin(), conv.shape)
        xy_act[i,:] = [x,y,activation]
        
    # highest activation at x,y
    idx = np.argmin(xy_act,axis=0)[2]
    best_x, best_y = xy_act[idx,0:2]
    
    return [best_x, best_y, activation]


def trachea_seed_point_detection(scan):
    xyz_act = np.zeros((scan.shape[-1],3))
    for i in range(scan.shape[2]):
        sys.stdout.write('.')
        xyz_act[i,:] = trachea_one_slice(scan[:,:,i])
    idx = np.argmin(xyz_act,axis=0)[2]
    best_x, best_y = xyz_act[idx,0:2]

    return xyz_act# best_x, best_y, idx  # coordinates of the selected seed point (you can also call it (i,j,k))

a = trachea_seed_point_detection(scan);
#x,y,z = trachea_seed_point_detection(scan);
#best_ct = scan[:,:,z]
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#circ = plt.Circle((y, x), radius=8, fc='none', ec='r',lw=5)
#plt.imshow(best_ct, cmap='gray')
#ax.add_patch(circ)
#plt.show()
#print('Found at x: {},y: {} in image: {}'.format(x,y,z))
for i in range(scan.shape[2]):
    print(i)
    plt.imshow(scan[:,:,i], cmap='gray')
    plt.show()