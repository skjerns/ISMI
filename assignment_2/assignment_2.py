# import libraries needed for this assignment
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from challenger import submit_results


from data import get_data, show_image
from preprocessing import visualize_gaussian_kernel


# get and visualize dataset
tra_imgs, tra_msks, tra_lbls = get_data()

# for i in range(len(tra_imgs)):
#     show_image(i, tra_imgs, tra_msks, tra_lbls)


# visualization of a gaussina filter
sigma = 2
visualize_gaussian_kernel(sigma)

exit()


# set the parameters for your CAD system here
n_samples_per_class_per_image = 100 # how many positive/negative pixels per image in the training set?
n_classes = 2           # how many classes in this problem?
sigmas = [1, 2, 4, 8, 16]   # what values of sigma?
n_features = 31         # how many features?

# define training data and labels
x_train = np.zeros((n_classes * n_samples_per_class_per_image * len(tra_imgs), n_features))
y_train = np.zeros((n_classes * n_samples_per_class_per_image * len(tra_imgs), 1))


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

    pos_feat = features[f,p_idx[0,:],p_idx[1,:],:]
    neg_feat = features[f,n_idx[0,:],n_idx[1,:],:]
    pos_lbl = lbl[p_idx[0,:],p_idx[1,:]]
    neg_lbl = lbl[n_idx[0,:],n_idx[1,:]]

    tmp_array = np.append(tmp_array, pos_feat, axis=0)
    tmp_array = np.append(tmp_array, neg_feat, axis=0) # we append the array
    y = np.append(y, pos_lbl)
    y = np.append(y, neg_lbl)

feats_calculated = True

x_train[:,:] = tmp_array[:,:] # I copy the array to check for right dimensions
y_train[:,0] = y[:]
ranges = np.ptp(x_train,axis=0)


x_train_norm, meanV, stdV = normalization(x_train)


# save training data to disk in numpy format
np.savez('./data/training_data.npz', x_train=x_train, y_train=y_train,
         x_train_norm=x_train_norm, meanV=meanV, stdV=stdV)



# load training data, define a nearest-neighbour classifier and train it.
K = int(np.sqrt(len(y_train))) / n_classes # define the K parameter
K = K / n_classes * n_classes + 1        # assure that uneven neigbours are used

print("Using {}-Nearest-Neighbours".format(K))

npz = np.load('./data/training_data.npz')
neigh = KNeighborsClassifier(n_neighbors=K, n_jobs=-1) # use multithreading
neigh.fit(npz['x_train_norm'], npz['y_train'].ravel())



# test images
tes_img_dir = './data/DRIVE/test/images'
tes_msk_dir = './data/DRIVE/test/mask'

tes_imgs = sorted(get_file_list(tes_img_dir, 'tif')[0])
tes_msks = sorted(get_file_list(tes_msk_dir, 'gif')[0])

result_output_folder = './data/DRIVE/test/results'
if not(os.path.exists(result_output_folder)):
    os.mkdirs(result_output_folder)


# To optimize the probability threshold we classify each image of the train set
# and see at which average threshold we get the highest accuracy.

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
    p_test = neigh.predict_proba(x_test_norm)
    for thres in thresholds:
        p_test_reshaped = p_test[:,0].reshape(img.shape[0], img.shape[1]) * msk
        final_output = (p_test_reshaped > thres) * 255 # Threshold the probabilitymap to obtain the final result
        accs[thres, f] = np.mean(final_output==lbl)
optimal_threshold = np.mean(np.argmin(np.mean(accs,axis=1)))

print("Optimal threshold found at {}".format(optimal_threshold))



#####################


# classify all images in the test set
for f in range(len(tes_imgs)):

    # load test image and mask
    img = np.asarray(Image.open(tes_imgs[f]))
    msk = np.asarray(Image.open(tes_msks[f]))

    ns = img.shape[0] * img.shape[1] #number of samples is ALL pixels in the image
    x_test = np.zeros((ns, n_features))

    # compute features
    print ('extraction features for image ' + str(f+1))
    features = extract_features(img, sigmas, 31)
    for k in range(features.shape[2]):
        x_test[:,k] = features[:,:,k].flatten()

    # normalize
    x_test_norm = normalization_test(x_test, mean_test, std_test)

    print('labeling pixels with nearest-neighbor...')
    p_test = neigh.predict_proba(x_test_norm)

    p_test_reshaped = p_test[:,0].reshape(img.shape[0], img.shape[1]) * msk
    final_output = (p_test_reshaped > optimal_threshold) * 255 # Threshold the probabilitymap to obtain the final result

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(p_test_reshaped)
    plt.subplot(1,3,3)
    plt.imshow(final_output)
    plt.show()

    im = Image.fromarray(final_output.astype('uint8'))
    im.save(os.path.join(result_output_folder, str(f+1) + "_mask.png"))


user = {'username': 'S.Kern', 'password' : '5CCN6PW2'} # enter you username and password
description = {'notes' : 'testing result submission system'}

submit_results (user, os.path.abspath(result_output_folder), description)

#  2. Improve your results! [optional]
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
