###########################
#
#  Here I am trying to create a CNN for classification of RAW data
#  The following methods have been tried and used:
#   1. Various CNN architectures from a vast amount of papers
#        2-5 convolutional layers
#        3-7 kernel size
#        [16,32,64] kernel number
#        locally connected layers
#   2. Dropout was used in addition to a L2 regularization
#   3. 
#
#   3. Because there are not a lot of training samples, data augmentation was used
#      -From the least frequent classes, all images have been added in their mirror
#       version. 
#      -Different amount of slices have been introduced as new training samples
#      -Different views were extracted for the test set and a winner-takes-all
#       voting was used to make the predictions more consistent.
#
#######################  IDAS
#
#  Because 3D ConvNets are often very applied yet I had the idea to use an RNN
#  and feed each slice into network in a sequential matter
#  This turned out to be a bit too complex, as I did not find out how to reset
#  the gradients after each 64 images.
#
#######################  CONCLUSION
#
#  I am a bit surprised that the CNN performed so bad. I have read many different
#  papers, some of them even performing the same task as here and when implementing the
#  exact same architecture it did not improve my results. I have no clue why
#  but I don't have more time left to spend on this assignment :(
#
#  Additionally I am surprised by the large differences between the validation
#  and the test set. Sometimes my results were more than 5% better than on test
#
#######################################
#


import numpy as np
import os
import sklearn
import sklearn.neighbors
import matplotlib.pyplot as plt
from IPython import display
#get_ipython().magic(u'matplotlib inline')
import itertools
import time
import random
import lasagne
import collections
import theano
from theano import tensor as T
import scipy
data_dir = 'c:\\assignment_5'# define here the directory where you have your data, downloaded from SURFDrive
noduleTypes = ["Solid", "Calcified", "PartSolid", "GroundGlassOpacity", "SolidSpiculated"]
n_classes = len(noduleTypes)
print('Classification problem with {} classes'.format(n_classes))
def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],                         [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]
    
def get_ortogonal_patches(x, r=0):
    views = list()
    indeces = [32];
#    indeces = np.arange(0,64)
    for idx in indeces:
        if r==0:
            views.append(x[idx,:,:])
            views.append(x[:,:,idx])
            views.append(x[:,idx,:])
        elif r==1:
            views.append(np.fliplr(x[idx,:,:]))
#            views.append(np.fliplr(x[:,:,idx]))
#            views.append(np.fliplr(x[:,idx,:]))
        elif r==2:
            views.append(np.flipud(x[idx,:,:]))
#            views.append(np.flipud(x[:,:,idx]))
#            views.append(np.flipud(x[:,idx,:]))
    return views

def z_score(arr1,arr2):
    conc = np.vstack([arr1,arr2])
    zarrays = scipy.stats.zscore(conc,axis=0)
    return zarrays[:len(arr1),:,:], zarrays[len(arr1):,:,:]

   

npz = np.load(os.path.join(data_dir, 'test', 'test_set.npz') )
#x_test = npz['x']
nodule_ids_test = npz['nodule_ids']

trainX = list()
trainY = list()

src_dir = os.path.join(data_dir, "training", "nodules")
i = -1;


for noduleType in noduleTypes:
    i += 1
    nodules_dir = os.path.join(src_dir, noduleType)
    npzs = get_file_list(nodules_dir, 'npz')
    for idx1, f in enumerate(range(len(npzs[0]))):
        file_path = npzs[0][f]
        filename = npzs[1][f]
        # axes are oriented as (z, x, y)
        npz = np.load(file_path)
        views = get_ortogonal_patches(npz['data'])
        trainX.append(views)
        trainY.append(i)
#        if i==2 or i==4:
#            views = get_ortogonal_patches(npz['data'], r=1)
#            trainX.append(views)
#            trainY.append(i)
#            views = get_ortogonal_patches(npz['data'], r=2)
#    #        for view in views:
#            trainX.append(views)
#            trainY.append(i)
            
numviews = len(views)
  
trainX = np.array(trainX, dtype=np.float32)
trainY = np.array(trainY, dtype=np.int32)

testX  = list()
src_dir = os.path.join(data_dir, "test", "nodules")   
for idx in nodule_ids_test:
    file_path = os.path.join(src_dir, idx)+'.npz'
    npz = np.load(file_path)
    views = get_ortogonal_patches(npz['data'])
#    for view in views:
    testX.append(views)
testX = np.array(testX)   

trainX, testX = z_score(trainX, testX )       
shape_cube = npz['data'].shape
print 'Size of our cubes',shape_cube

# convenience function that returns the axial, coronal and sagitaal view given a cube (in numpy format) containing a nodule


# function to split training set into training and validation subsets
def split_training_validation_datasets(x, y, val_percentage=0.3, val_balanced=True):
    """
    Derive a training and a validation datasets from a given dataset with
    data (x) and labels (y). By default, the validation set is 30% of the
    training set, and it has balanced samples across classes. When balancing,
    it takes the 30% of the class with less samples as reference.
    """
    if val_balanced == False:
        all_idx = np.arange(len(y))
        val_idx = np.random.choice(all_idx,int(len(y)*(1-0.3)), replace=False)
        tra_idx = np.delete(all_idx,val_idx)
    else:
        all_idx = np.arange(len(y))
        val_idx=[]
        nitems = int((collections.Counter(y).most_common()[-1][1]) * 0.3)
        for c in np.unique(y):
            
            val_idx.append(np.random.choice(np.where(y==c)[0], nitems, replace=False))
                
        val_idx = np.vstack(val_idx).flatten()
        tra_idx = np.delete(all_idx,val_idx, axis=0)

    
    x_validation = x[val_idx,:]
    y_validation = y[val_idx]
    x_train = x[tra_idx,:]
    y_train = y[tra_idx]


    return x_train, y_train, x_validation, y_validation
from collections import Counter

def voting(array, items):
    assert(len(array)%items == 0)
    new_array = list()
    
    for i in np.arange(len(array)/items):
        classes = array[i*items:(i+1)*items]
        maxclass = Counter(classes).most_common(1)[0][0]
        new_array.append(maxclass)
        print((i+1)*3)
    
    return np.array(new_array)
    
# In[12]:


trainX, trainY, valX, valY = split_training_validation_datasets(trainX, trainY, val_balanced=True)
print "Total number of cases in train: {}".format(collections.Counter(trainY))
print "Total number of cases in valid: {}".format(collections.Counter(valY))

from sklearn.utils import shuffle
trainX, trainY = shuffle(trainX,trainY) 

#trainX = np.expand_dims(trainX, axis=1)
#valX = np.expand_dims(valX, axis=1)
#testX = np.expand_dims(testX, axis=1)
# In[16]:

def plot_confusion_matrix(conf_mat, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix
    """
    plt.figure()
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
   
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j], horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# input tensors for data and targets
input_var = T.tensor4('inputs')
target_var = T.dmatrix('targets')


data_size = trainX.shape
n_classes = len(noduleTypes)
print data_size
print n_classes


def label_to_one_hot(y):
    '''
    Convert labels into "one-hot" representation
    '''
    n_values = np.max(y) + 1
    y_one_hot = np.eye(n_values)[y]
    
    return y_one_hot


# Now we can apply the function that converts labels:

# In[23]:

# training
print 'Number of samples in training set',trainY.shape
trainY = label_to_one_hot(trainY)
print trainY.shape

# validation
print 'Number of samples in validation set',valY.shape
valY = label_to_one_hot(valY)
print valY.shape
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax


# In[24]:
def build_neural_network(data_size, n_classes):
    network = lasagne.layers.InputLayer(shape=(None,data_size[1], data_size[2], data_size[3]), input_var=input_var)
    network = lasagne.layers.Conv2DLayer( network, num_filters=32, filter_size=(3, 3),  nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()) 
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer( network, num_filters=32, filter_size=(3, 3),  nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()) 
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer( network, num_filters=32, filter_size=(3, 3),  nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()) 
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.LocallyConnected2DLayer(network, num_filters= 16, filter_size=(7,7))
    dense = lasagne.layers.DenseLayer( network, num_units=512, W=lasagne.init.GlorotUniform(),      nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.dropout(dense,p = 0.6)
    network = lasagne.layers.DenseLayer( network, num_units=n_classes,W=lasagne.init.GlorotUniform(),      nonlinearity=lasagne.nonlinearities.softmax)
    return network


# In[20]:
start = time.time()

# get the network
network = build_neural_network(data_size, n_classes)

# get the prediction during training
prediction = lasagne.layers.get_output(network)
from lasagne.regularization import regularize_layer_params_weighted, l2
# define the (data) loss
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
layers = {network.input_layer.input_layer: 0.01}
l2_reg = regularize_layer_params_weighted(layers, l2)
loss = loss + l2_reg
# In[25]:

# extract the parameters we want to optimize
params = lasagne.layers.get_all_params(network, trainable=True)
#updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
updates = lasagne.updates.adadelta(loss,params)

# get the prediction on the validation set during training
val_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)
val_loss = val_loss.mean()
l2_reg = regularize_layer_params_weighted(layers, l2)
val_loss = loss + l2_reg

# compute the (mean) accuracy
val_acc  = T.mean(T.eq(T.argmax(val_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)
train_fn  = theano.function([input_var, target_var], loss, updates=updates, name='train')
val_fn    = theano.function([input_var, target_var], [val_loss, val_acc]  , name='validation')
get_preds = theano.function([input_var]            , val_prediction       , name='get_preds')



n_epochs   = 100
batch_size = 64 # adapt this value based on the memory of your GPU
n_mini_batch_training = data_size[0]/batch_size # number of training mini-batches given the batch_size

# lists where we will be storing values during training, for visualization purposes
tra_losses = []
val_losses = []
val_accs   = []
    
    # we want to save the parameters that give the best performance on the validation set
    # therefore, we store the best validation accuracy, and save the parameters to disk
#best_val_acc = 0
#fig = plt.figure(figsize=(10, 5))
#plt.ion()
best_val_acc = 0
# loop over the number of epochs
plt.close('all')
fig = plt.figure(figsize=(10, 5))
for epoch in xrange(n_epochs):
    # training
    cum_tra_loss = 0.0 # cumulative training loss
    for b in range(n_mini_batch_training-1):
        x_batch = trainX[b*batch_size:(b+1)*batch_size,:].astype(np.float32) # extract a mini-batch from x_train
        y_batch = trainY[b*batch_size:(b+1)*batch_size,:] # extract labels for the mini-batch
        mini_batch_loss = train_fn(x_batch, y_batch)
        cum_tra_loss += mini_batch_loss
        
    # validation
    val_loss, val_acc = val_fn(valX.astype(np.float32), valY.astype(np.float32))
    # if the accuracy improves, save the network parameters
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # save network
        params = lasagne.layers.get_all_param_values(network)
        np.savez('./nn_params.npz', params=params)
    
    tra_loss = cum_tra_loss/n_mini_batch_training # final training loss for this epoch
    
    # add to lists
    tra_losses.append(tra_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    tra_loss_plt, = plt.plot(range(len(tra_losses)), tra_losses, 'b')
    val_loss_plt, = plt.plot(range(len(val_losses)), val_losses, 'g')
    val_acc_plt, = plt.plot(range(len(val_accs)), val_accs, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend([tra_loss_plt, val_loss_plt, val_acc_plt], 
               ['training loss', 'validation loss', 'validation accuracy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Best validation accuracy = {:.2f}%'.format(100. * best_val_acc))
    plt.pause(0.001)
print(best_val_acc)


    # ## Classification: validation set
    # Now we can use the trained network to classify the validation set, and check that the performance corresponds to the best value obtained during training. We can compute the accuracy and also visualize the confusion matrix, to get a feeling how well we are doing.
    

#%%
# load the network architecture again
network = build_neural_network(data_size, n_classes) # define the network again, in case you start here
npz = np.load('./nn_params.npz') # load stored parameters
lasagne.layers.set_all_param_values(network, npz['params']) # set parameters

# compile the function again, using the (re)loaded network
val_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
get_preds = theano.function([input_var]            , val_prediction       , name='get_preds')

# classify validation set
prediction = get_preds(valX.astype(np.float32))
y_validation_auto = np.argmax(prediction, axis=1)
conf_mat_nn  = sklearn.metrics.confusion_matrix(np.argmax(valY, axis=1), y_validation_auto)
#acc_nn = sklearn.metrics.accuracy_score(valY, y_validation_auto)
#print('Accuracy on validation set: {:.2f}%'.format(100. * acc_nn))
plot_confusion_matrix(conf_mat_nn, classes=noduleTypes,
                      title='Confusion matrix: Neural Network classifier (True label vs. Predicted label)')
    
testY=np.array([])

prediction = get_preds(testX)
test_prediction = np.argmax(prediction, axis=1)
#test_prediction = voting(test_prediction,numviews)

# classify test set
n_test_samples = nodule_ids_test.shape[0]
h_csv = open('./nn_results.csv', 'w')
h_csv.write('nodule_id, label\n')
for n in range(n_test_samples):
#    test_sample = testX[n:n+1,:].astype(np.float32)
    nodule_id   = nodule_ids_test[n]
    y = test_prediction[n]
    testY = np.append(testY,y)
    h_csv.write('{}, {}\n'.format(nodule_id, y+1))
h_csv.close()

# In[42]:

import challenger
sub = raw_input('Submit? Y/N: ')
if sub.lower() == 'y':
    challenger.submit_results({'username': 'S.Kern',
                           'password': '5CCN6PW2'},
                          "nn_results.csv",
                          {'notes': 'more RAW'})
# ### Teaching Assistants:
# 
# Send us an email for questions. Remember to send your assignment before Monday midnight.
# 
# - Gabriel Humpire: g.humpiremamani@radboudumc.nl
# - Peter Bandi: peter.bandi@radboudumc.nl
