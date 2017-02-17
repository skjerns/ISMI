import numpy as np
from math import floor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.signal


########## KERNELS #######################

def gauss_filter(sigma):
    sigma_ = 1.5
    x_vec = np.arange(-sigma_*3, sigma_*3, 1.0)
    y_vec = np.arange(-sigma_*3, sigma_*3, 1.0)
    xx, yy = np.meshgrid(x_vec, y_vec)
    kernel = (1/(2*np.pi*sigma**2)) * np.exp(-(xx**2+yy**2)/(2*sigma**2))
    return kernel


def get_gaus_deriv(g):
    gx  = np.gradient(g,axis=0)
    gxx = np.gradient(g,axis=0)
    gxy = np.gradient(g,axis=1)

    gy  = np.gradient(g,axis=1)
    gyy = np.gradient(g,axis=1)
    gyx = np.gradient(g,axis=0)
    return gx, gy, gxx, gyy, gxy, gyx


# code to test and visualize the gaussian kernel
def visualize_gaussian_kernel(sigma):
    """
        Visualizes the Gaussian kernel defined above for a given sigma.
    """
    gaussian_kernel = gauss_filter(sigma) #calculate the Gaussian filter kernel

    fig = plt.figure()
    ax = Axes3D(fig)
    x_dim, y_dim = gaussian_kernel.shape
    x,y = np.meshgrid(range(x_dim), range(y_dim))
    offset_x = floor(x_dim/2)
    offset_y = floor(y_dim/2)

    x = x - float(offset_x)
    y = y - float(offset_y)
    #ax.plot_surface(x, y, gaussian_kernel, antialiased=True, cmap=cm.jet, linewidth=0)
    ax.plot_surface(x, y, gaussian_kernel, rstride=1, cstride=1, cmap=cm.jet)
    plt.show()


######### FEATURE EXTRACTION ######################

def extract_features(img, sigmas, n_features=1):
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

    # >>> YOUR CODE STARTS HERE <<<
    i = 1
    for s in sigmas:
        g = gauss_filter(s)
        gx, gy, gxx, gyy, gxy, gyx = get_gaus_deriv(g)
        features[:,:,i] = scipy.signal.fftconvolve(img_g, g, mode='same') ;i+=1
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gx, mode='same') ;i+=1
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gy, mode='same') ;i+=1
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gxx, mode='same') ;i+=1
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gyy, mode='same') ;i+=1
        features[:,:,i] = scipy.signal.fftconvolve(img_g, gxy, mode='same') ;i+=1
    # >>> YOUR CODE ENDS HERE <<<

    return features


#######################################################

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


def normalization_test(x_test, meanV, stdV):
    """
        Normalization of the test data
    """
    eps = np.finfo(float).eps
    x_test_post = (x_test - meanV)/(stdV + eps)

    return x_test_post