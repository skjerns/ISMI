


################### EXPLANATION
# The current version of this script has 13% FP with 75mm length and 46% coverage
# I consider this a good trade-off between the measures. We gain 50% more length
# for only 1%-point less FP in comparrison with the previous approach.
#
# I updated the original script with the following features:
#  1) I use +one SD of the median to control for explosion. The mean
#     just seemed to be too conservative
#  2) A dilation on the labelled image is used with a structuring element
#     having only entries in the vertical direction This trick is used in
#     accordance with the paper: https://vlebb.leeds.ac.uk/bbcswebdav/orgs/SCH_Computing/FYProj/reports/1011/Parker.pdf
#     With this structuring element we should be able to preserve the vertical
#     nature of the airway while filling missing values on the way.
#     In practice this approach did not improve my work that's why I'm using 3) instead
#  3) Holes are filled on the final image using fill_holes
#  4) The best way to control for explosion would be to check the tree paths
#     and see if there are 'bottlenecks' connecting two large regions.
#     To implement this algorithm would have required a bit more time though.
#
###############










# import libraries needed for this assignment

import SimpleITK as sitk
import os
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from challenger import submit_results



def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]


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

    return np.median(mean_volume_mm3)+np.std(mean_volume_mm3)


# extract list of airway annotation files
tra_ann_list = get_file_list(tra_ann_dir, 'mhd')[0]

# compute the mean airway volume using the function you have implemented
mean_volume_mm3 = get_mean_volume_mm3(tra_ann_list)
print('Mean of mean, max and median volume = {} mm^3'.format(mean_volume_mm3))

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


# In[10]:

def get_scan_params(img_itk):
    # >>> YOUR CODE HERE, Replace 'None's with your code <<<
    img_params = {}
    img_params['origin'] = img_itk.GetOrigin()
    img_params['spacing'] = img_itk.GetSpacing()
    img_params['direction'] = img_itk.GetDirection()
    return img_params

# In[11]:

def get_voxel_volume_mm3(spacing):
    # >>> YOUR CODE HERE, Replace 'None's with your code <<<
    voxel_volume_mm3 = np.prod(spacing)
    return voxel_volume_mm3

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
        ############ skip if we are above threshold, but fill list to be able to plot it nicely
        if len(volumes)>0 and (volumes[-1]>mean_volume_mm3 or volumes[-1]==mean_volume_mm3*4):
            volumes.append(mean_volume_mm3*50)
            print('yes')
        else:
                  
            print('-----------------------------------------')
            print('processing threshold {}'.format(threshold))
            selem = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,1,0],[0,1,0],[0,1,0]],[[0,0,0],[0,0,0],[0,0,0]]])
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
#                airways = skimage.morphology.binary_dilation(conn_comps,selem = selem)
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
    airways_segmentation =  conn_comps == seed_point_label
#    airways_segmentation = skimage.morphology.binary_dilation(airways_segmentation,selem=selem)
    airways_segmentation = np.array(scipy.ndimage.morphology.binary_fill_holes(airways_segmentation),dtype=np.int32)
    
    return airways_segmentation



# threshold(s) to consider
thresholds = np.arange(-980,-500,1)



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


# In[ ]:

user = {'username': 'S.Kern', 'password' : '5CCN6PW2'} # enter you username and password
description = {'notes' : 'o0n4 finestgrain mmm closeing'}

submit_results (user, result_output_folder, description)
