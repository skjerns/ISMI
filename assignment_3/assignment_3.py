
# coding: utf-8

# # Airway segmentation in chest CT scans

# In this assignment, we are going to develop a system for automatic segmentation of airways in chest CT scans. The input data is a 3D volume of the human thorax. An example of *coronal* slice of this 3D volume is shown in Figure 1a. The output of this assignment is also a 3D volume containing binary values for each voxel, indicating whether it belongs to an airway or not. The output airway segmentation should be a single **connected component**. An example of this output is shown as a coronal 2D overlay in Figure 1b and as a 3D volume in Figure 1c.
# 
# <img src="./figures/exampleCT_and_airwaySegmentation_small.png" alt="example coronal slice of an chest CT and an example airway segmentation" style="width:1000px;height:300px;">
# <I>Figure 1. An example of coronal slice of a 3D chest CT scan (a) and an example of output of an airway segmentation algorithm (b and c). The example airway segmentation is shown as a red overlay in (b) and as a 3D rendering in (c). The blue arrows in (b) show some examples of an airway wall.</I>
# <br>
# <br>
# The main goal of airway segmentation is to differentiate the airway lumen from the surrounding lung tissue (also called <I>lung parenchyma</I>). Both appear as dark voxels on CT but are separated by the airway wall which appear as a brighter tube surrounding the airway lumen. The blue arrows in Figure 1b give some examples of an airway wall. For the larger airways, the thick airway walls provide a clear separation between the airway lumen and the surrounding lung parenchyma. However, this separation becomes less pronounced for smaller airways because of limitations in resolution and partial volume effects, which often results in a segmentation that leaks into the lung parenchyma as is shown in Figure 2b and 2c. 
# 
# <img src="./figures/exampleCT_and_airwaySegmentation_small_leaks.png" alt="example coronal slice of an chest CT and an example airway segmentation" style="width:1000px;height:300px;">
# <I>Figure 2. An example of the same coronal slice as shown in Figure 1, but with an airway segmentation that leaked into the lung parenchyma (b and c). The example leaky airway segmentation is shown as a red overlay in (b) and as a 3D rendering in (c).</I>
# <br>
# <br>
# A common issue when segmenting airways is **how to define a good trade-off between increasing the airway tree length and reducing the amount of leaks**. In most methods, this trade-off is determined by optimising the parameters that influence the tree length and the amount of leaks.
# 
# 

# ## Clinical background
# 
# Airway segmentation in CT plays an important role in the analysis of pulmonary diseases. A robust method to automatically segment an airway tree in CT is especially relevant for quantifying airway changes such as thickening of the airway wall and changes in lumen diameter. Quantifying these airway changes may be key for improving the diagnosis and treatment planning for pulmonary diseases involving airway pathology such as chronic obstructive pulmonary disease (COPD), cystic fibrosis, or bronchiectasis. In addition, segmented airways can aid in segmenting other pulmonary structures such as lobes, segments, and pulmonary arteries and veins.

# ## Data
# 
# For the development and evaluation of your airway segmentation method, we are going to use data from an internationally recognised challenge for airway segmentation (EXACT'09, http://image.diku.dk/exact/). The goal of EXACT'09 is to compare algorithms that extract the airway tree from chest CT scans using a common data set of 20 chest CT scans. From EXACT09, we selected 10 cases for training and 10 cases for test.
# You can download the data from this link: https://surfdrive.surf.nl/files/index.php/s/IuZCAf2K8U092Su (pwd: ismi2017).
# In the evaluation of your method, we will we focus on the following measurements:
# 
# - airway tree length (in mm)
# - percentage of detected airway tree length (as compared to the reference standard)
# - volume of leaked voxels (in mm<sup>3</sup>)
# - false positive rate (percentage of leaked voxels as compared to correctly segmented voxels)
# 
# In order to develop your method, a set of 10 chest CT scans with reference airway segmentations are made available (training/tuning set). These reference airway segmentations will give you a good idea on what you may expect from your segmentation algorithm, and may be used to train/tune your algorithm.

# ## Tasks for this assignment
# 
# This assignement consist of two mandatory tasks and one optional taks:
# 
# 1. develop an algorithm for **automatic airway segmentation** based on connected component [mandatory]
# 2. submit the results to **challenger** (the mini-challenge framework, now it has a name!) [mandatory]
# 3. improve the performance of your algorithm by combining your solution with other known techniques [optional]
# 
# ### 1. Develop an algorithm for automatic airway segmentation
# 
# * Use (double) thresholding techniques, find the optimal value of threshold per scan
# * Use connected components, find the best neighborhood strategy empirically
# * Use the trachea seed point to detect the connected component that belong to airways
# * Implement some explosion control technique, you can use the data from training set for that
# 
# ### 2. Submission to the mini-challenge framework
# 
# The output of this assignment must be the airway segmentation for the 10 test cases from the EXACT09 challenge, provided together with this notebook.
# In order to generate and submit the results to challenger:
# * Run your airway segmentation algorithm on the 10 test images
# * Save the segmentation output (3D matrix) as an ITK file to a local folder. Segmented airway voxels should have a value different from zero. 
# * Submit the results to the framework using the submission cell (at the end of this notebook)
# 
# ### 3. Improve airway segmentation
# 
# Use your knowledge on filtering, texture descriptors and morphology to improve the result of your airway segmentation method. Your improvement can be based on some post-processing of the result obtained with connected components, or a new solution. Be creative!
# 
# ### Additional information
# * Notebook submission deadline: **Monday February 20th at 23:59**.
# * Send the notebook via e-mail to the teaching assistants:
#   * Mehmet: *mehmet.dalmis@radboudumc.nl*
#   * Freerk: *freerk.venhuizen@radboudumc.nl*

# ## Initialization
# We first initialize variables and function that will be used in the notebook to develop the airway segmentation method.

# In[1]:

# import libraries needed for this assignment
import SimpleITK as sitk
import os
import dicom
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
get_ipython().magic(u'matplotlib inline')
from IPython import display
from challenger import submit_results


# Convenience function to read list of files in a directory (also used in previous assignments)

# In[2]:

def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]


# Define paths to the data used in this assignment and extract list of filenames. Adjust the 'data_folder' path to match your local settings.

# In[3]:

# set the path to the folder with the data you downloaded from SURFDrive
data_folder = 'c:\ismi'

# training images
tra_img_dir = os.path.join(data_folder, 'training' , 'images')
# training annotations
tra_ann_dir = os.path.join(data_folder, 'training', 'annotations')
# test images
tes_img_dir = os.path.join(data_folder, 'test', 'images')

tra_img_fls = get_file_list(tra_img_dir, 'dcm')[0] # dicom files
tra_ann_fls = get_file_list(tra_ann_dir, 'mhd')[0] # ITK files
tes_img_fls = get_file_list(tes_img_dir, 'dcm')[0] # dicom files


# Define the path for your output results. By default, it is a 'result' folder in the directory of this notebook. Feel free to change it if you like. In this folder, you will have to look for your results when you want to visualize them.

# In[4]:

# set output folder (required to submit result to the mini-challenge)
result_output_folder = os.path.join('./results')
if not(os.path.exists(result_output_folder)):
    os.makedirs(result_output_folder)


# Convenience function to load a file in ITK format and return both an ITK and a numpy version of it. Having both the ITK file format and the numpy format is handy, because ITK files contain information about the scan, like spacing (voxel size) etc. However, using numpy format is convenient when operations on matrixes have to be applied.

# In[5]:

def load_itk_file(file_name):
    ''' load itk image file. '''
    img_itk = sitk.ReadImage(file_name)
    img_np = sitk.GetArrayFromImage(img_itk)
    return img_itk, img_np


# Convenience function to save files in ITK format. Although other formats are available in ITK, in this assignment we want to use *mhd* files, which is the format supported by the mini-challenge evaluation script. Note that the image parameters *spacing*, *origin* and *direction* are used in the function.

# In[6]:

def save_itk_file(img, filename, img_params, use_compression=True):
    ''' Save input image 'img' in ITK format mhd. 
        It requires a filename to save the input image, and
        a set of ITK parameters, like spacing, origin and direction
        to properly store the image. If the input image is not
        in numpy format, img_params are ignored.
    '''
    if not filename.endswith('mhd'):
        print ('filename should have extension .mhd!')
        return False
    # check if the input variable is a numpy array
    if type(img).__module__ == np.__name__:
        itk_img = sitk.GetImageFromArray(img) # convert it
        itk_img.SetSpacing(img_params['spacing'])
        itk_img.SetOrigin(img_params['origin'])
        itk_img.SetDirection(img_params['direction'])
    else:
        itk_img = img
    w = sitk.ImageFileWriter()
    w.SetUseCompression(use_compression)
    w.SetFileName(filename)
    w.Execute(itk_img)
    return True


# ## Visualization with MeVisLab 
# 
# A nice way to visualize images and annotations is by using MeVisLab. You can download MeVisLab for your operating system (it supports Windows, Linux and Mac OS) from this link: http://www.mevislab.de/download/.
# MeVisLab comes with the ITK library installed, which you can use to load both the CT scan, the reference standard and your segmentation result, and inspect it by scrolling the scan in 2D or in 3D. These are two examples of how you can visualize your segmentation result in MeVisLab:
# 
# <img src="./figures/mevislab_overlay_example.png" alt="MeVisLab overlay">
# 
# To scroll up and down in the 2D visualization, just use the weel of your mouse, or press it to do a quick scrolling.
# To zoom in and out in the 3D view, use the combination of keys "CTRL + central mouse button".
# You will find a file called **network_visualization.mlab** in the folder of this assignment (which you also received via Blackboard), open it to load the network in the example and use it to visualize your results.
# 
# Try to open a few training cases and overlay the provided airway annotations. This will be useful to get familiar with the kind of images you will have to process and with the appearance of airways in CT scans.
# 
# #### MeVisLab tips & tricks
# * In MeVisLab, you can change the window level (look-up-table) used to visualize image intensities by clicking the right button of your mouse and moving the cursor around. You will see that the gray levels change. This may be useful to get an optimal visualization of the lungs and to visualize airways.
# * In the network provided, we included a module called **OrthoSwapFlip**. Double click on the module and check the options available. One option is called *View Direction*. Change the value in the menu and check how the view changes in the window.
# * You can switch from voxel coordinates to world coordinates in the View2D module by doing: 
#    
#    *click on the right button > Show Window > Settings > Appearance [TAB] > Current Pos. in > Voxel/World.*
# 
# 
# * Now we are using MeVisLab to visualize the provided manual annotations overlayed on training images. You can imagine that later the same network can be used to inspect test cases, and to overlay the segmentation results provided by your algorithm. 

# ### About CT scans
# Typically, CT scans are acquired from the bottom part of the body to the top part, and during the acquisition slices are stored.
# For this reason, the slice with index zero in the CT scan actually corresponds to the last slice, if you see the scan from top to bottom. This also means that the lung that you see on the left is actually the right lung, and the other way around. It is important to keep these things in mind while developing algorihtms for medical image analysis!

# ## 1. Airway segmentation with explosion control

# ### Extract airway volume statistics

# One of the problems of airway segmentation is that airways can easily leak into the lung parenchyma, since the intensity of the parenchyma is also low, and airway walls are sometimes thin. For this reason, during the segmentation procedure, knowing what is the **volume of airway** that we can expect can help prevent explosion of airway segmentation into the parehcnyma. In order to do that, we can get statistics on airway volume from the training set provided with this assignment.
# 
# Implement a function that computes the mean volume, in **mm<sup>3</sup>**, of an airway tree using data from the training set. Use all the scans in the training set to compute the mean volume.
# 
# **NOTE:** You can get the *spacing* information from ITK images using the function **GetSpacing()** of the image object itself.

# In[7]:

# function to compute the mean volume
def get_mean_volume_mm3(file_lst):
    ''' 
        Compute the mean volume (in mm^3) of airway tree from a given set.
        The input is a list of ITK files of annotated airway trees.
    '''
    mean_volume_mm3 = []
    for filename in file_lst:
        annotation_itk, annotation_np = load_itk_file(filename)
        mean_volume_mm3.append( np.sum(annotation_np) * np.prod(annotation_itk.GetSpacing()) )
    print(np.min(mean_volume_mm3))
    print(np.max(mean_volume_mm3))
    print(np.median(mean_volume_mm3))
    return np.mean(mean_volume_mm3)


# #### Question
# We asked to compute the **mean** volume from the training set. Do you think that other statistics would be useful to implement an *explosion control* algorihtm? Why?

# How about the median and the min/max? E.g. if we are above the max, we know that we overdid it.

# **NOTE.** If you think that other statistics are needed, feel free to expand the *get_mean_volume_mm3()* function to include more statistics, or to implement more functions.
# 
# Now we can compute (at least) the mean volume using the function we have defined.

# In[8]:

# extract list of airway annotation files
tra_ann_list = get_file_list(tra_ann_dir, 'mhd')[0]

# compute the mean airway volume using the function you have implemented
mean_volume_mm3 = get_mean_volume_mm3(tra_ann_list)
print('Mean volume = {} mm^3'.format(mean_volume_mm3))


# Now that we have computed the mean volume, we can use it to implement some kind of explosion control in our airway segmentation algorithm. We have to segment airways in all test cases. Therefore, for each case, the tasks are the following:
# 
# * load the scan
# * pick a seed point in the trachea
# * do connected components segmentation, based on threshold(s)
# * use the seed point to select the connected component related to airways
# * in your algorithm, use the learned mean volume to pick the optimal threshold that prevents (limits) leakage
# 
# #### About trachea seed point
# In order to develop your algorihtm, you will need the coordinates of a seed point placed in the trachea.
# This is related to what we developed in the first assignment!
# However, due to noise in CT scans, good seed points are typically defined close to the **carina point**. 
# The seed point produced by the algorithm that you developed in the first assignment may give a different location, since it was not optimized for that (we simply did not ask for that).
# 
# To solve this problem, we manually picked seed points for the test cases of this assignment, which you can find in the dictionary defined in the next cell.

# In[9]:

# seed points dictionary (z, y, x)
seed_points = {}
seed_points['1.0.000.000000.0.00.0.0000000000.0000.0000000000.000'] = (327, 249, 245)
seed_points['1.2.276.0.28.3.0.14.4.0.20090213134050413'] = (258, 263, 251)
seed_points['1.2.276.0.28.3.0.14.4.0.20090213134114792'] = (251, 260, 248)
seed_points['1.2.392.200036.9116.2.2.2.1762676169.1080882991.2256'] = (303, 286, 249)
seed_points['1.2.840.113704.1.111.2004.1131987870.11'] = (202, 235, 251)
seed_points['1.2.840.113704.1.111.2296.1199810886.7'] = (358, 331, 247)
seed_points['1.2.840.113704.1.111.2296.1199810941.11'] = (336, 335, 248)
seed_points['1.2.840.113704.1.111.4400.1131982359.11'] = (227, 200, 252)
seed_points['1.3.12.2.1107.5.1.4.50585.4.0.7023259421321855'] = (287, 273, 258)
seed_points['2.16.840.1.113669.632.21.3825556854.538251028.390606191418956020'] = (180, 257, 238)


# **NOTE.** In the training set, a region growing algorithm based on seed points was used to build the reference segmentation. This means that the volume of trachea above the seed point is not considered in the computation of the mean volume. You will have to take this into account when developing the explosion control part of your algorithm! 

# #### Question
# Are the seed-point coordinates given in voxel-coordinates or world-coordinates? What is the difference between world-coordinates and voxel-coordinates?

# The seed points seem to be in voxel coordinates (All values between 0-512). Voxelcoordinates are defined just within in the image, having the origin within the image space. No negative numbers are possible. World coordinates have their origin defined differently, for instance relative to some reference point. They can be negative.

# ### Auxiliary functions

# In the following cells, we propose templates of code that you can use to implement functions that will be necessary to run the main airway segmentation algorithm, with explosion control. The idea is that you implement simple functions that can do specific tasks, and then these functions will be used in the last cells of the notebook to define the main function for airway segmentation, and to run the main script that segments airways in the test set. Let's get started!

# Implement a function to get parameters from the image in ITK format. The parameters we need to extract are:
#   * origin
#   * spacing
#   * direction

# In[10]:

def get_scan_params(img_itk):
    # >>> YOUR CODE HERE, Replace 'None's with your code <<<
    img_params = {}
    img_params['origin'] = img_itk.GetOrigin()
    img_params['spacing'] = img_itk.GetSpacing()
    img_params['direction'] = img_itk.GetDirection()
    return img_params


# #### Question
# What is the data format of the *spacing* object, and what is the meaning of its values?

# The format is a tuple and contains 3 values, namely the spacing in X,Y and Z direction setting the relation of voxel to mm
# 

# Implement a function that computes the volume of a voxel in mm<sup>3</sup> given the spacing vector

# In[11]:

def get_voxel_volume_mm3(spacing):
    # >>> YOUR CODE HERE, Replace 'None's with your code <<<
    voxel_volume_mm3 = np.prod(spacing)
    return voxel_volume_mm3


# Based on the result of the connected component algorithm, we will extract the segmentation of airways as the connected component that contains the seed point provided. Since images can be very noisy, the seed point may not belong to any connected component, but still very close to it. For this reason, given a seed point, we will have to check for labels available in its neighborhood, and pick the one of the closest component. 
# 
# The following function implements this idea by considering voxels of connected components in a cube of size (2d+1)x(2d+1)x(2d+1) centered on the seed point. By default, the size of the cube is 21 voxels, but you can tune this parameter and see if this affects your results. We still don't have the volume of connected components, but we will extract it later in the main script. For the time being, think of it as a 3D matrix in numpy, where each connected structure in 3D is assigned a different label (a different number >0).

# In[12]:

def get_seed_point_label(conn_comps, seed_point, d=10):
    ''' 
        Detect the closest label in a (2d+1) x (2d+1) x (2d+1) cube 
        centered on the seed point.
    '''
    distances = []
    labels = []
    for dx in range(-d, d):
        for dy in range(-d, d):
            for dz in range(-d, d):
                label = conn_comps[seed_point[0] + dz,
                                   seed_point[1] + dy,
                                   seed_point[2] + dx]
                if label != 0:
                    labels.append(label)
                    distances.append(np.sqrt(float(dx)**2 + float(dy)**2 + float(dz)**2.))
    if len(labels) == 0:
        return 0
    else:
        return labels[np.argmin(np.array(distances))]


# ### Main airway segmentation function
# 
# Implement a function that performs airway segmentation of a given CT scan in numpy array format based on connected components. The function needs at least the following inputs:
# 
# * img_np: CT scan in numpy array format
# * seed_point: coordinates of the seed point in 3D, oriented as (z, y, x)
# * thresholds: list of threshold(s) you want to apply to extract a binarized version of the image, to use as input for the connected component algorithm
# 
# In the development of this function, consider that:
# * each threshold will generate a certain volume, which we will have to compare with the mean volume learned from the training set, to implement explosion control; keep track of the volumes in a list
# * the z-coordinate of the seed point defines the last slice to consider in your volume, because you want to discard a part of the trachea and make your volume comparable with the one learned from the training set.
# 
# We have defined the structure of this (long) function, and already implemented some of its parts. Fill in the missing lines of code. This function will be used later in the main script to run airway segmentation on the entire test set.
#from skimage import segmentation
#markers[~img_np] = -1                                 
##                                        
##                                        
###                                        
#import scipy 
#import skimage                                        
#                                        
#thresholds = np.arange(-1000,-500,100)                                     
#volumes = []
#filenameann = "C:\\ISMI\\training\\annotations\\scan01.mhd"
#filenameimg = "C:\\ISMI\\training\\images\\scan01.dcm"
#seed_point = (453,268,255)
#_, img_anno = load_itk_file(filenameann)
#img_itk, img_np = load_itk_file(filenameimg) 
#img_params = get_scan_params(img_itk)
#
#    # get voxel volume
#voxel_volume_mm3 = get_voxel_volume_mm3(img_params['spacing'])   
#it = 1
#nn = 4
## try all thresholds and compute airway volume for each of them
#for threshold in thresholds:
#    print('-----------------------------------------')
#    print('processing threshold {}'.format(threshold))
#
#    # extract a binarized version of the image by applying the threshold(s)
#    img_binary = scipy.ndimage.binary_erosion( img_np < threshold ,iterations=it)
#
#
#    # run connected-components algorithm on the binary image and label each region,
#    # you can use one of the functions that we have seen in the last lecture
#    # or that we used in the first assignment
##        conn_comps, nfeats = scipy.ndimage.label(img_binary) 
#    conn_comps = skimage.morphology.label(img_binary, neighbors = nn)
#    
#    
#    print('-> found {} connected components'.format(np.max(conn_comps)))     
#
#    # get the label of the seed point 
#    seed_point_label = get_seed_point_label(conn_comps, seed_point)
#
#    
#    print('-> seed point label = {}'.format(seed_point_label))
#
#    if seed_point_label != 0:
#        
#        # extract airways as the connected component that has the label of the seed point
#        airways = scipy.ndimage.binary_dilation(conn_comps == seed_point_label,iterations=it)
#
#        # remove voxels in the upper part of the trachea, to make the volume
#        # compatible with what measured in the training set
#        airways[seed_point[0]:] = 0 # CT scans are acquired from bottom to top
#
#        # compute the airway volume in mm^3
#        airway_volume_mm3 = np.sum(airways) * voxel_volume_mm3
#
#        # add the computed volume to the list
#        volumes.append(airway_volume_mm3)
#        print('--> airway volume = {} mm^3'.format(airway_volume_mm3))
#    else:
#        volumes.append(np.inf)
#        print('--> airway volume = 0 mm^3')
#
## visualize volume trend
#plt.plot(thresholds, volumes)
#plt.xlabel('HU')
#plt.ylabel('mm^3')
#plt.plot(thresholds, mean_volume_mm3*np.ones((len(thresholds, ))), '--r')
#plt.show()
#
## find optimal threshold, which minimizes the "explosion" of segmentation in the parenchyma
#optimal_threshold = thresholds[(np.abs(volumes-mean_volume_mm3)).argmin()]
#
#print('optimal threshold {} found'.format(optimal_threshold))
#
## Apply the optimal threshold to the scan to get airway segmentation.
## Basically, you have to repeat steps that you have implemented in 
## the previous 'for loop' already, but this time you don't need to remove voxels 
## in the upper part of the trachea, the evaluation algorithm will handle this
#print('extracting airways...')
#img_binary = scipy.ndimage.binary_erosion( img_np < threshold ,iterations=it)
#conn_comps = skimage.morphology.label(img_binary, neighbors = nn)
#seed_point_label = seed_point_label = get_seed_point_label(conn_comps, seed_point)
#airways_segmentation = np.array(scipy.ndimage.binary_dilation(conn_comps == seed_point_label,iterations=it),dtype=np.int32)
#                          
#FP = np.sum(airways_segmentation == (img_anno == 0))/       float(np.sum(airways_segmentation)          )            
#                                        
#stop                                 
#                                        
                                        
# In[ ]:
import scipy 
import skimage

it = 1
nn = 4

def get_airway_segmentation(img_np, seed_point, thresholds, voxel_vol_mm):
    ''' Airway segmentation in CT scan based on connected components and explosion control.'''
    
    # initialize list of volumes
    volumes = []
    
    # try all thresholds and compute airway volume for each of them
    for threshold in thresholds:
        
        if len(volumes)>0 and (volumes[-1]>mean_volume_mm3 or volumes[-1]==mean_volume_mm3*50):
            volumes.append(mean_volume_mm3*50)
            print('yes')
        else:
                  
            print('-----------------------------------------')
            print('processing threshold {}'.format(threshold))
    
            # extract a binarized version of the image by applying the threshold(s)
            img_binary =  img_np < threshold 
    
    
            # run connected-components algorithm on the binary image and label each region,
            # you can use one of the functions that we have seen in the last lecture
            # or that we used in the first assignment
    #        conn_comps, nfeats = scipy.ndimage.label(img_binary) 
            conn_comps = skimage.morphology.label(img_binary, neighbors = nn)
            
            
            print('-> found {} connected components'.format(np.max(conn_comps)))     
    
            # get the label of the seed point 
            seed_point_label = get_seed_point_label(conn_comps, seed_point)
        
            
            print('-> seed point label = {}'.format(seed_point_label))
    
            if seed_point_label != 0:
                
                # extract airways as the connected component that has the label of the seed point
                airways = conn_comps == seed_point_label
                # remove voxels in the upper part of the trachea, to make the volume
                # compatible with what measured in the training set
                airways[seed_point[0]:] = 0 # CT scans are acquired from bottom to top
    
                # compute the airway volume in mm^3
                airway_volume_mm3 = np.sum(airways) * voxel_vol_mm
    
                # add the computed volume to the list
                volumes.append(airway_volume_mm3)
                print('--> airway volume = {} mm^3'.format(airway_volume_mm3))
            else:
                volumes.append(0)
                print('--> airway volume = 0 mm^3')
        

    # visualize volume trend
    plt.plot(thresholds, volumes)
    plt.xlabel('HU')
    plt.ylabel('mm^3')
    plt.plot(thresholds, mean_volume_mm3*np.ones((len(thresholds, ))), '--r')
    plt.show()

    # find optimal threshold, which minimizes the "explosion" of segmentation in the parenchyma
    optimal_threshold = thresholds[(np.abs(volumes-mean_volume_mm3)).argmin()]

    print('optimal threshold {} found'.format(optimal_threshold))

    # Apply the optimal threshold to the scan to get airway segmentation.
    # Basically, you have to repeat steps that you have implemented in 
    # the previous 'for loop' already, but this time you don't need to remove voxels 
    # in the upper part of the trachea, the evaluation algorithm will handle this
    print('extracting airways...')
    img_binary =  img_np < optimal_threshold
    conn_comps = skimage.morphology.label(img_binary, neighbors = nn)
    seed_point_label = seed_point_label = get_seed_point_label(conn_comps, seed_point)
    airways_segmentation = np.array(conn_comps == seed_point_label,dtype=np.int32)
    
    return airways_segmentation


# #### Question
# What kind of connectivity did you choose for the connected component algorithm and why?

# *Your answer here.*

# ### Main script to run airway segmentation on the test set
# Now that you have implemented all the functions necessary to extract airways from a given CT scan, you can run your algorithm on the entire test set and submit the results.
# But first you have to define the threshold (or the thresholds) you want to use in the connected component algorithm. Since the test script expects a list of values, even if you want to use only one value, put it in a list. 

# In[ ]:

# threshold(s) to consider
thresholds = np.arange(-1000,-500,1)





# #### Question
# What is the theoretically expected intensity value (in Hounsfield unit) for voxel in the airway lumen? What is a plausible range of values that you can test for airway segmentation?

# *Your answer here.*

# In[ ]:

# extract and save airway segmentation from test cases
for test_img_filename in tes_img_fls:

    # extract scan id from filename, will be used to save the result image
    scan_id = os.path.splitext(ntpath.basename(test_img_filename))[0]
    print('processing scan id {}'.format(scan_id))

    # load ITK image in ITK and numpy format
    img_itk, img_itk_np = load_itk_file(test_img_filename) 
    print('scan loaded with size {}'.format(img_itk_np.shape))                     
    img_np = img_itk_np
    # extract scan parameters
    img_params = get_scan_params(img_itk)

    # get voxel volume
    voxel_volume_mm3 = get_voxel_volume_mm3(img_params['spacing'])   

    # get seed point coordinates from the defined dictionary
    seed_point = seed_points[scan_id]



    # airway segmentation using the function that you have implemented
    airways = get_airway_segmentation(img_itk_np, seed_point, thresholds, voxel_volume_mm3)
    
    # visualize slice with seed point
    plt.subplot(1,2,1)

    plt.imshow(img_itk_np[seed_point[0]].squeeze(), cmap='gray')
    plt.scatter(seed_point[2], seed_point[1], c='r')
    plt.subplot(1,2,2)
    plt.imshow(airways[seed_point[0]].squeeze(), cmap='gray')
    plt.scatter(seed_point[2], seed_point[1], c='r')
    plt.show()
    
    # save to disk
    print('saving segmentation result to disk')
    outputfile = os.path.join(result_output_folder, scan_id+'.mhd')
    save_itk_file(airways, outputfile, img_params)
    print('done!')


# One of the output produced by the functions that we have prepared is a plot of the trend of airway volume as a function of the threshold(s) that you have picked. Based on this plot, you may think of narrowing the search interval for your thresholds, and try to further fine-tune the threshold (and other parameters, like the connectivity, etc.) and try to improve the results based on plain connected components segmentation.

# ## 2. Submitting results to "challenger"
# The airway segmentation results for the test set have been stored in the result folder that you defined at the beginning of the notebook (result_output_folder).
# Before submitting the results, you may want to have a look at your results by using the MeVisLab network provided. You can use the network to load the test scan and to show the segmentation results in overlay. This can give you a feeling what to expect from the evaluation, and can also give some feedback how to improve your method, apart from the numerical results that you will get from **challenger**.

# In[ ]:

user = {'username': 'S.Kern', 'password' : '5CCN6PW2'} # enter you username and password
description = {'notes' : 'o0n4 finestgrain'}

submit_results (user, result_output_folder, description)


# ## 3. Improving airway segmentation (optional)
# 
# Now that you have segmented airways using the connected component algorithm, you can improve the airway segmentation by using other (additional) techniques. For this task, you can use all the knowledge that you have on *filtering*, *texture analysis*, *classification* and *morphology*.
# There is no specific limitation to what you can do here, pick the technique (or the techniques) that you think is the most appropriate, and try to improve your results. **Be creative!**
