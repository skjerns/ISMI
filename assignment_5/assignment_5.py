##############
#  This script is used for hyperparameter search.
#  I use the following architecture
#  0. Input Layer (ReLu)
#  1. Dense Layer (ReLu)
#  2. Dropout Layer
#  3. Dense Layer (ReLu)
#  4. Dropout Layer
#  5. Output Layer (softmax)
#
#  I use the HyperOpt library for a parameter search.
#  It uses Bayesian statistics to achieve the best outcome using 1000 samples
#  I am optimizing the following parameters:
#  Number of neurons per layer (10-120)
#  Percentage of dropout (0-0.9)
#  Batch size (32,64,128)
################ RESULTS
#  The best architecture seems to be with 90 Neurons using 0.3 Dropout.
#  I get around 56% on the test set and 60% on the validation set. 
#  I cannot explain the quite big difference between validation and test
#
#
# import libraries
import numpy as np
import os
import sklearn
import sklearn.neighbors
import matplotlib.pyplot as plt
import itertools
import lasagne
import theano
from theano import tensor as T


# In[2]:

data_dir = 'c://assignment_5'# define here the directory where you have your data, downloaded from SURFDrive


noduleTypes = ["Solid", "Calcified", "PartSolid", "GroundGlassOpacity", "SolidSpiculated"]
n_classes = len(noduleTypes)
print('Classification problem with {} classes'.format(n_classes))


# In[4]:

# convenience function
def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],                         [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]

def get_ortogonal_patches(x):
    dims = x.shape
    axial = x[32,:,:]
    coronal =  x[:,:,32]
    sagittal=  x[:,32,:]
    print(dims)
    return axial, coronal, sagittal


# load training data (given features)
npz = np.load(os.path.join(data_dir, 'training', 'training_set.npz'))
x_trainr = npz['x']
y_trainr = npz['y']
y_trainr = y_trainr.astype(np.int32)  # convert it to int, which they actually are.
print x_trainr.shape
print y_trainr.shape              


# In[8]:

# load test data (given features)
npz = np.load(os.path.join(data_dir, 'test', 'test_set.npz') )
x_test = npz['x']
nodule_ids_test = npz['nodule_ids']
print x_test.shape

# normalize training data
x_mean = np.mean(x_trainr, axis=0)
x_std  = np.std(x_trainr, axis=0)
x_trainr = (x_trainr - x_mean)/x_std

# normalize test data
x_test = (x_test - x_mean)/x_std



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
        val_idx = np.random.choice(all_idx,int(len(y)*(1-0.3)))
        tra_idx = np.delete(all_idx,val_idx)
    else:
        all_idx = np.arange(len(y))
        val_idx=[]
        nitems = int((collections.Counter(y).most_common()[-1][1]) * 0.3)
        for c in np.unique(y):
            val_idx.append(np.random.choice(np.where(y==c)[0], nitems))
        val_idx = np.vstack(val_idx).flatten()
        tra_idx = np.delete(all_idx,val_idx)


    x_validation = x[val_idx,:]
    y_validation = y[val_idx]
    x_train = x[tra_idx,:]
    y_train = y[tra_idx]


    return x_train, y_train, x_validation, y_validation


# In[12]:

import collections

print "Total number of cases per class: {}".format(collections.Counter(y_trainr))
x_train, y_train, x_validation, y_validation = split_training_validation_datasets(x_trainr, y_trainr,val_balanced=False)
print "Total number of cases in train: {}".format(collections.Counter(y_train))
print "Total number of cases in valid: {}".format(collections.Counter(y_validation))

# shuffle training dataset
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train,y_train) # shuffle them using the same RNG


def plot_confusion_matrix(conf_mat, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix
    """
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
input_var  = T.fmatrix('input')
target_var = T.dmatrix('targets')


data_size = x_train.shape
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


# In[23]:

# training
print 'Number of samples in training set',y_train.shape
y_train_one_hot = label_to_one_hot(y_train)
print y_train_one_hot.shape

# validation
print 'Number of samples in validation set',y_validation.shape
y_validation_one_hot = label_to_one_hot(y_validation)
print y_validation_one_hot.shape

# check number of samples per class
print np.sum(y_train_one_hot, axis=0)
print np.sum(y_validation_one_hot, axis=0)

from hyperopt import STATUS_OK
import time
# In[20]:
start = time.time()
def NN(params):
    neurons, dropout, batch_size, epoch = params
    print neurons, dropout, batch_size
    neurons = int(neurons)
    # define neural network with 1 hidden layer
    
    def build_neural_network(data_size, n_classes, neurons,dropout):
        network = lasagne.layers.InputLayer(shape=(None, data_size[1]),
                                         input_var=input_var)
        network = lasagne.layers.DenseLayer(network, num_units=neurons,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        network = lasagne.layers.DropoutLayer(network, p=dropout)
        network = lasagne.layers.DenseLayer(network, num_units=neurons,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        network = lasagne.layers.DropoutLayer(network, p=dropout/3)
        network = lasagne.layers.DenseLayer(
            network, num_units=n_classes,W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.softmax)
        
        return network
    
    
    # ### Loss function
    # Now that the architecture is defined, we move to the second component of the learning framework, the **loss function**. In order to do that, we first have to define a function that, given the network, gets the predicted probability for a given input sample.
    # Lasagne offers a function for that, 'get_output()'. Since we are dealing with a multi-class classification problem, categorical cross-entropy seems a reasonable choice.
    
    # In[21]:
    
    # get the network
    network = build_neural_network(data_size, n_classes, neurons, dropout)
    
    # get the prediction during training
    prediction = lasagne.layers.get_output(network)
    
    # define the (data) loss
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # In[25]:
    
    # extract the parameters we want to optimize
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.adadelta(loss,params)
    

    
# get the prediction on the validation set during training
    val_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)
    val_loss = val_loss.mean()

# compute the (mean) accuracy
    val_acc  = T.mean(T.eq(T.argmax(val_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)
    train_fn  = theano.function([input_var, target_var], loss, updates=updates, name='train')
    val_fn    = theano.function([input_var, target_var], [val_loss, val_acc]  , name='validation')
    get_preds = theano.function([input_var]            , val_prediction       , name='get_preds')
    

    
    n_epochs   = epoch
    batch_size = batch_size # adapt this value based on the memory of your GPU
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
    for epoch in xrange(n_epochs):

        
        # training
        cum_tra_loss = 0.0 # cumulative training loss
        for b in range(n_mini_batch_training-1):
            x_batch = x_train[b*batch_size:(b+1)*batch_size,:].astype(np.float32) # extract a mini-batch from x_train
            y_batch = y_train_one_hot[b*batch_size:(b+1)*batch_size,:] # extract labels for the mini-batch
            mini_batch_loss = train_fn(x_batch, y_batch)
            cum_tra_loss += mini_batch_loss
            
        # validation
        val_loss, val_acc = val_fn(x_validation.astype(np.float32), y_validation_one_hot.astype(np.float32))
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
    print(best_val_acc)
    return {'loss':-best_val_acc, 'status': STATUS_OK}
    # ## Classification: validation set
    # Now we can use the trained network to classify the validation set, and check that the performance corresponds to the best value obtained during training. We can compute the accuracy and also visualize the confusion matrix, to get a feeling how well we are doing.
    
    # In[30]:
################################## Optimization routine
from hyperopt import hp   

space = (
     hp.quniform('neurons',10,120,1),
     hp.uniform('dropout',0,0.8),
     hp.choice('batch_size',[8,16,32,64,128]),
     hp.choice('epoch',[100,250]),
    )

from hyperopt import fmin, tpe
best = fmin(NN, space, algo=tpe.suggest, max_evals=1000)
print(best)
print('This took {} minutes'.format((time.time()-start)/60.0))
