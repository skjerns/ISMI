
# coding: utf-8

# In[1]:

# import libraries needed for this assignment
import os
import numpy as np
from math import floor
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
get_ipython().magic(u'matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10, 6)
from sklearn.neighbors import KNeighborsClassifier
import scipy.signal
from challenger import submit_results


# In[2]:

# function to get a list of file of a given extension, both the absolute path and the filename
def get_file_list(path,ext='',queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]


# Point to the local copy of training data

# In[3]:

tra_img_dir = './data/DRIVE/training/images'
tra_msk_dir = './data/DRIVE/training/mask'
tra_lbl_dir = './data/DRIVE/training/1st_manual'

tra_imgs = sorted(get_file_list(tra_img_dir, 'tif')[0])
tra_msks = sorted(get_file_list(tra_msk_dir, 'gif')[0])
tra_lbls = sorted(get_file_list(tra_lbl_dir, 'gif')[0])

# In[6]:
from skimage.filters import gabor_kernel

def gauss_filter(sigma, x0=0.0, y0=0.0):
    x_vec = np.arange(-sigma*3, sigma*3, 1.0) 
    y_vec = np.arange(-sigma*3, sigma*3, 1.0)
    xx,yy = np.meshgrid(x_vec,y_vec)
    
    g = (1/(2*np.pi*sigma**2)) * np.exp(-(xx**2+yy**2)/(2*sigma**2))
    
    gx  = np.gradient(g,axis=0)
    gxx = np.gradient(g,axis=0)
    gxy = np.gradient(g,axis=1)
    
    gy  = np.gradient(g,axis=1)
    gyy = np.gradient(g,axis=1)
    
    return g, gx, gxx, gxy, gy, gyy



def get_gabors():
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels


# In[9]:
from scipy.ndimage.filters import sobel
from skimage import feature

def extract_features(img, sigmas, n_features=51):
    """
        Computes features from a given input image with given sigmas.
        Output of this function is a 3 dimensional numpy array containing
        the different computed features for the given input image.
    """    
    dims = img.shape # dimensions of the image
    
    features = np.zeros((dims[0], dims[1], n_features)) # each feature map has the same size as the input image
    
    # the first feature we use is the pixel intensity in the green channel itself
    img_g = img[:,:,1] #I just assume it follows the RGB convention and not GBR or BGR...
    features[:,:,0] = img_g

            
    gabors = get_gabors()        
            
    # >>> YOUR CODE STARTS HERE <<<
    i = 1
    for s in sigmas:
        gfilters = gauss_filter(s)
        for gf in gfilters:
            features[:,:,i] = scipy.signal.fftconvolve(img_g, gf, mode='same') ;i+=1

    for gabor in gabors:
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gabor, mode='same') ;i+=1
                
    features[:,:,i] = sobel(img_g, axis=0) ;i+=1
    features[:,:,i] = sobel(img_g, axis=1) ;i+=1
    features[:,:,i] = sobel(img_g, axis=0)+sobel(img_g, axis=1) ;i+=1
    features[:,:,i] = feature.canny(img_g, sigma=.3)
    # >>> YOUR CODE ENDS HERE <<<  
        
    return features


# In[10]:
smpl = 0
for idx in range(len(tra_msks)):
    smpl += np.sum(np.asarray(Image.open(tra_msks[idx]))==255)
    
# set the parameters for your CAD system here
n_samples_per_class_per_image = 10000 # how many positive/negative pixels per image in the training set?
n_classes = 2           # how many classes in this problem?
sigmas = [1,2,4,8,16]   # what values of sigma?
n_features = 51         # how many features?

# define training data and labels
x_train = np.zeros((n_classes * n_samples_per_class_per_image * len(tra_imgs), n_features))  
y_train = np.zeros((n_classes * n_samples_per_class_per_image * len(tra_imgs), 1))

# In[11]:

def get_random_indexes(msk, n_idxs):
    """ 
        Returns rows and columns of user-defined positive and negative indexes.
        The variable msk is a binary map in numpy format.
    """
    pos_idxs = np.array(np.where(msk > 0))
    neg_idxs = np.array(np.where(msk == 0))
    n_pos = pos_idxs.shape[1] # number of positives found in the mask
    n_neg = neg_idxs.shape[1] # number of negatives found in the mask
    n_min = min(n_neg, min(n_idxs, n_pos))
    rnd_idxs_pos = range(n_pos)
    np.random.shuffle(rnd_idxs_pos)
    rnd_idxs_neg = range(n_neg)
    np.random.shuffle(rnd_idxs_neg)
    return pos_idxs[:, rnd_idxs_pos[:n_min]], neg_idxs[:, rnd_idxs_neg[:n_min]] 

# In[27]:

tmp_array = np.empty((0,n_features))
y = np.empty(0)
if 'feats_calculated' not in vars():
    features_tra = np.zeros([len(tra_imgs), 584, 565, n_features])
    
for f in range(len(tra_imgs)):
     
    # load training image and annotation
    img = np.asarray(Image.open(tra_imgs[f]))
    lbl = np.asarray(Image.open(tra_lbls[f]))
                
    # extract features from the given images
    print('extracting features for image ' + str(f+1))
    if 'feats_calculated' not in vars(): # only calculate them if we did not already, save computation
        features_tra[f,:,:,:] = extract_features(img, sigmas, n_features) # implement the extract_features function defined above!

    # extract random position of samples     
    p_idx, n_idx = get_random_indexes(lbl, n_samples_per_class_per_image)    
    
    pos_feat = features_tra[f,p_idx[0,:],p_idx[1,:],:]
    neg_feat = features_tra[f,n_idx[0,:],n_idx[1,:],:]
    pos_lbl = lbl[p_idx[0,:],p_idx[1,:]]
    neg_lbl = lbl[n_idx[0,:],n_idx[1,:]]
    
    tmp_array = np.append(tmp_array, pos_feat, axis=0)
    tmp_array = np.append(tmp_array, neg_feat, axis=0) # we append the array
    y = np.append(y, neg_lbl)
    y = np.append(y, pos_lbl)

    
feats_calculated = True    

x_train = tmp_array[:,:] # I copy the array to check for right dimensions
y_train = y[:]   
ranges = np.ptp(x_train,axis=0)


# In[17]:

def normalization(x_train):
    """
        Normalization of x_train
    """ 
    # >>> YOUR CODE STARTS HERE <<<
    meanV = np.mean(x_train, axis = 0) # vector of mean values
    stdV = np.std(x_train, axis = 0)  # vector of standard deviation values
    x_train_norm = (x_train-meanV)/stdV
    # >>> YOUR CODE ENDS HERE <<<
    
    return x_train_norm, meanV, stdV

x_train_norm, meanV, stdV = normalization(x_train)


# For convenience, you can save your training data, which you can load later and use for testing purposes without the need to rebuild it every time you run an experiment. You may want to define a flag to enable/disable training, or just execute some cells in this notebook instead of executing all cells. You may want to add more varables that you think will be necessary to test new samples.

# In[18]:

# save training data to disk in numpy format
np.savez('./data/training_data.npz', x_train=x_train, y_train=y_train, 
         x_train_norm=x_train_norm, meanV=meanV, stdV=stdV)

# In[22]:
from sklearn.ensemble import RandomForestClassifier
# load training data, define a nearest-neighbour classifier and train it.
K = int(np.sqrt(len(y_train))) / n_classes # define the K parameter
K = K / n_classes * n_classes + 1        # assure that uneven neigbours are used

print("Using Random Forest")

npz = np.load('./data/training_data.npz')    
#neigh = KNeighborsClassifier(n_neighbors=K, n_jobs=-1) # use multithreading 
#neigh.fit(npz['x_train_norm'], npz['y_train'].ravel())
import time
start = time.time()
print("Starting fit. Takes around 250 seconds on my PC.")
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1)
forest.fit(npz['x_train_norm'], npz['y_train'].ravel())
print("Fitting took {} seconds".format("%.f" % (time.time()-start)) )

# In[23]:

# test images
tes_img_dir = './data/DRIVE/test/images'
tes_msk_dir = './data/DRIVE/test/mask'

tes_imgs = sorted(get_file_list(tes_img_dir, 'tif')[0])
tes_msks = sorted(get_file_list(tes_msk_dir, 'gif')[0])

result_output_folder = './data/DRIVE/test/results'
if not(os.path.exists(result_output_folder)):
    os.mkdirs(result_output_folder)


# #### Question
# 
# Do you think that you have to apply some kind of normalization to test data as well? In case you do, what do you think is the best strategy? Will you compute statistics (mean, std) on the test set, or will you use the ones from the training set? Why?

# We need to standardize it so that the values follow the same distribution as the train set. the kNN is sensitive to the absolutes of values (not scale-invariant) as it calculates the distances on the un-normalized input space. If we now have a few outliers in the test set that are not present in the training set we shift all values of the test set further than the ones of the train set. This will induce a bias.

# In[24]:

# define vectors of mean value and standard deviation for the test set here
mean_test = meanV
std_test = stdV


# In[25]:

def normalization_test(x_test, meanV, stdV):
    """
        Normalization of the test data
    """ 
    eps = np.finfo(float).eps    
    x_test_post = (x_test - meanV)/(stdV + eps) 
    
    return x_test_post


# In the next cell we loop over all the images in the test set and do the following for every image:
# 
# * Extract features for every pixel in the image
# * Apply normalization
# * Classify every pixel
# * Save the output to disk
#     
# In the classification step, the output of the classifier can be:
# * the predicted label of the test sample classified
# * a likelihood value (pseudo-probability), based on processing distance measures
# 
# In order to optimize the performance of our system, we would like to obtain some kind of probability for each pixel, which we can later post-process by applying a threshold, which we will have to optimize. By changing the threshold, the amount of pixel classified as vessel and as background will change.

# In[30]:

# To optimize the probability threshold we classify each image of the train set 
# and see at which average threshold we get the highest accuracy.
import sklearn
# classify all images in the train set
thresholds = np.arange(0,255) 
accs = np.zeros([255,20])

for f in range(len(tra_imgs)):
    
    # load test image and mask
    img = np.asarray(Image.open(tra_imgs[f])) # don't actually need to load this. just to keep to code intact.
    msk = np.asarray(Image.open(tra_msks[f]))
    lbl = np.asarray(Image.open(tra_lbls[f]))      
    
    ns = img.shape[0] * img.shape[1] #number of samples is ALL pixels in the image
    x_test = np.zeros((ns, n_features))
    
    # compute features
    features = features_tra[f,:,:,:]    # that's why I save all calculated features from before, no need to recalculate
    for k in range(features.shape[2]):
        x_test[:,k] = features[:,:,k].flatten()
    
    # normalize
    x_test_norm = normalization_test(x_test, mean_test, std_test)
    
    print('searching for best threshold in train image {}'.format(f))
    p_test = forest.predict_proba(x_test_norm)
    thresholds = np.array(np.unique(p_test)*255, dtype='int32')[:-1]
    for thres in thresholds:
        p_test_reshaped = p_test[:,0].reshape(img.shape[0], img.shape[1]) * msk
        final_output = (p_test_reshaped > thres) * 255 # Threshold the probabilitymap to obtain the final result            
        final_output = scipy.ndimage.binary_closing(final_output, iterations=1)
        accs[thres, f] = sklearn.metrics.f1_score(final_output.flatten()>0,lbl.flatten()>0)
#        accs[thres, f] = np.mean(final_output==lbl)
optimal_threshold = np.mean(np.argmax(np.mean(accs,axis=1)))


print("Optimal threshold found at {}".format(optimal_threshold))


#%%
####################
    

# classify all images in the test set
for f in range(len(tes_imgs)):
    
    # load test image and mask
    img = np.asarray(Image.open(tes_imgs[f]))
    msk = np.asarray(Image.open(tes_msks[f]))
            
    ns = img.shape[0] * img.shape[1] #number of samples is ALL pixels in the image
    x_test = np.zeros((ns, n_features))
    
    # compute features
    print ('extraction features for image ' + str(f+1))
    features = extract_features(img, sigmas, n_features)    
    for k in range(features.shape[2]):
        x_test[:,k] = features[:,:,k].flatten()
    
    # normalize
    x_test_norm = normalization_test(x_test, mean_test, std_test)

    print('labeling pixels with nearest-neighbor...')
    p_test = forest.predict_proba(x_test_norm)
    
    p_test_reshaped = p_test[:,0].reshape(img.shape[0], img.shape[1]) * msk
    final_output = (p_test_reshaped > optimal_threshold) * 255 # Threshold the probabilitymap to obtain the final result
#    final_output=scipy.ndimage.binary_fill_holes(final_output)
    final_output=scipy.ndimage.binary_closing(final_output, iterations=1)               
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(p_test_reshaped)
    plt.subplot(1,3,3)
    plt.imshow(final_output)
    plt.show()
    
    im = Image.fromarray(final_output.astype('uint8'))
    im.save(os.path.join(result_output_folder, str(f+1) + "_mask.png"))
    


# ### Submit your result
# 
# After processing all the images in the test set you can upload your result to the challenge website and see how well you performed compared to your collegues! You can submit as often as you want, only the best result counts.
# 
# Enter your username and password in the cell below to submit your result. You can add a description for your own reference. This description will be also shown in the website, which you can use as a reference to keep track of your the development of your system.
# 
# You should have received your username and password by email. Otherwise, TAs will give you a temporary spare account.

# In[ ]:

user = {'username': 'S.Kern', 'password' : '5CCN6PW2'} # enter you username and password
description = {'notes' : 'hungarian 10k samples'}

submit_results (user, os.path.abspath(result_output_folder), description)


# **Check your result!** http://ismi17.diagnijmegen.nl/

# ## 2. Improve your results! [optional]
# 
# Try to improve your results and resubmit.
# 
# A few ideas to improve the system are:
# 
# * Design a segmentation system based on morphology, without using pixel classification.
# * Improve the performance of your existing system by:
#   * Using more features (Local Binary Patterns, Gabor filters, use rotated derivative of Gaussian filters, etc.). Get creative!
#   * Postprocessing to improve the results using morphological filtering
#   * Using more training samples
#   * etc.
# 
# 
