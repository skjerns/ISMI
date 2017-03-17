####################################
#
# My current setup gets ~89% on the test set which I think is quite good
#
# I used a Conv-net with the structure with the following building block:
# 1. Conv2D 128@(3,3) + BN + Dropout 
# 2. Conv2D 128@(3,3) + BN 
# 3. MaxPool2D(2,2)
# This structure was repeated 3 times.
# The output of the CNN block was 128x4x4 = 3k
# The output got flattened, then
# I finished with 2 FC layers with 1024 and 512 neurons with Dropout
#
# 1. I also implemented some (relatively) weak L2 norm which helped to counter the overfit slightly.
# 2. Data augmentation boosted me around 4% up.
# 3. The Gaussian input layer did not help, I ditched it
# 4. I used LeakyRectify just to try it and it did help me with p=0.33
#
####################################
# import libraries
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
#get_ipython().magic(u'matplotlib inline')
#matplotlib.rcParams['figure.figsize'] = (20, 12)
from tqdm import tnrange, tqdm_notebook
from IPython import display
import time
import challenger



# In[2]:

dataset_dir = 'C:\\ISMI\\cifar-10-batches-py\\'

# load one batch, reshape data
f = open(os.path.join(dataset_dir, 'data_batch_1'), 'rb') # example for batch_1
cifar_batch = pickle.load(f)
f.close()

# print original data shape (matrix format)
print cifar_batch['data'].shape
print len(cifar_batch['labels'])

# shape after re-arranging data shape (tensor format)
reshaped_cifar_batch = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype(np.float32)
print reshaped_cifar_batch.shape


# Before you define your datasets, it is useful to check the distribution of labels across batches of CIFAR10, in case some batches have skewed distributions of labels.
# In order to do that, you can use visualize the histogram of labels using the function *hist()* of the matplotlib library:
# 
#     plt.hist()

# In[5]:

batch_1 = pickle.load(open(os.path.join(dataset_dir, 'data_batch_1'), 'rb'))
batch_2 = pickle.load(open(os.path.join(dataset_dir, 'data_batch_2'), 'rb'))
batch_3 = pickle.load(open(os.path.join(dataset_dir, 'data_batch_3'), 'rb'))
batch_4 = pickle.load(open(os.path.join(dataset_dir, 'data_batch_4'), 'rb'))
batch_5 = pickle.load(open(os.path.join(dataset_dir, 'data_batch_5'), 'rb'))
plt.subplot(1,5,1); plt.hist(batch_1['labels']); plt.title('batch_1')
plt.subplot(1,5,2); plt.hist(batch_2['labels']); plt.title('batch_2')
plt.subplot(1,5,3); plt.hist(batch_3['labels']); plt.title('batch_3')
plt.subplot(1,5,4); plt.hist(batch_4['labels']); plt.title('batch_4')
plt.subplot(1,5,5); plt.hist(batch_5['labels']); plt.title('batch_5')


# #### Question
# Do you think that you have to organize batches in a specific way in order to make a training and a validation set? Do you think that your choice would change the performance significantly?

# The labels are distributed relatively evenly, that means we can just use a leave-one-out approach and use 4 batches for training while taking one for validation. Because the labels are not perfectly even I suggest using a multiclass F1-Score (or how about Kohens-Kappa?) for evaluation instead of Accuracy.

# Now that you have decided how to distribute batches in your training and validation datasets, you can implement a function that builds and returns the datasets. You will be using this function in your experiments later.

# In[9]:

# convenience function that builds and returns data sets from CIFAR10

def label_to_one_hot(y):
    '''
    Convert labels into "one-hot" representation
    '''
    return y
#    n_values = np.max(y) + 1
#    y_one_hot = np.eye(n_values)[y]
#    
#    return y_one_hot


def load_data(val_id=-1):
    batches = list()
    for i in np.arange(1,6):
        batches.append(pickle.load(open(os.path.join(dataset_dir, 'data_batch_' + str(i)), 'rb')))
    idx = np.arange(0,5)
    if val_id == -1: val_id = np.random.choice(idx,1)[0]
    print ('taking fold {}'.format(val_id))
    idx = np.delete(idx, val_id)
    
    train_x = list()
    train_y = list()
    for i in idx:
        train_x.append((batches[i]['data'].reshape(-1, 3, 32, 32) / 255.).astype(np.float32))
        train_x.append((batches[i]['data'].reshape(-1, 3, 32, 32) / 255.).astype(np.float32)[:,:,:,::-1])
        train_y.append(label_to_one_hot(batches[i]['labels']))
        train_y.append(label_to_one_hot(batches[i]['labels']))
    # make training set
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y).flatten()

    # make validation set
    val_x = (batches[val_id]['data'].reshape(-1, 3, 32, 32) / 255.).astype(np.float32)
    val_y = label_to_one_hot(batches[val_id]['labels'])
    val_y = np.array(val_y)
    # load test set
    test_batch = pickle.load(open(os.path.join(dataset_dir, 'test_batch'), 'rb'))
    tes_x = (test_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype(np.float32)
    tes_filenames = test_batch['filenames']

    # load labels CIFAR10
    f = open(os.path.join(dataset_dir, 'batches.meta'), 'rb')
    cifar_dict = pickle.load(f)
    label_to_names = {k:v for k, v in zip(range(10), cifar_dict['label_names'])}
    f.close()

    print '-----------------------------------------------------'
    print label_to_names
    print '-----------------------------------------------------'
    print '> shape training set tensor: {}'.format(train_x.shape)
    print '> length training labels: {}'.format(len(train_y))
    print '-----------------------------------------------------'
    print '> shape validation set tensor: {}'.format(val_x.shape)
    print '> length training labels: {}'.format(len(val_y))
    print '-----------------------------------------------------'
    print '> shape test set tensor: {}'.format(tes_x.shape)
    
    return train_x, train_y, val_x, val_y, tes_x, tes_filenames


# # Build your network
# In the followign tasks you have to fill in the body of the function that returns the output layer of your network and
# define the training, validation and test *functions*.
# Note that these tasks were part of the assignment last week as well.

# ## Task 1: build a simple convolutional network
# ### Network
# Define your network builder function. 
# We build a convolutional network that contains:
# 
# 1. input layer
# 2. convolutional layer
# 3. max pooling layer
# 4. convolutional layer
# 5. max pooling layer
# 6. fully-connected layer(s)
# 7. soft-max layer
# 
# ### Hint
# 
# 1. Select the number of convolutional and max pooling layers and choose the filter size so, that the input image is shrinked to 5x5 before the fully connected layers.
# 2. Use at least one fully connected layer between the last convolutional layer and the output fully connected layer with softmax nonlinearity.

# In[21]:

# Build your network

# Define your network builder function. You can assume the (1, 3, 32, 32) input size.
#
def build_network(input_tensor):
    """
    Define the network layers.
    
    Args:
        input_tensor (theano.tensor.ftensor4): Input tensor.
    
    Returns:
        lasagne.layers.Layer: Output layer.
    """
    
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_tensor)
#    network = lasagne.layers.GaussianNoiseLayer(network)
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.DropoutLayer(network, p=0.3)
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.DropoutLayer(network, p=0.3)
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.DropoutLayer(network, p=0.3)
    network = lasagne.layers.Conv2DLayer( network, num_filters=128, filter_size=(3, 3),pad='same',  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.33), W=lasagne.init.HeUniform()) 
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print network.output_shape
    network = lasagne.layers.FlattenLayer(network)
    network = lasagne.layers.DropoutLayer(network, p=0.3)
    network = lasagne.layers.DenseLayer( network, num_units=1024, W=lasagne.init.HeUniform(),      nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.DropoutLayer(network, p=0.6)
    network = lasagne.layers.DenseLayer( network, num_units=512, W=lasagne.init.HeUniform(),      nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.DropoutLayer(network, p=0.6)
    network = lasagne.layers.DenseLayer( network, num_units=10,W=lasagne.init.GlorotUniform(),      nonlinearity=lasagne.nonlinearities.softmax)
    return network



# ### Functions
# Define your **training**, **validation** and **evaluation** functions. By default, **use_l2_regularization** is set to False, and l2_loss is set to 0.0. You will have to modify this later, but it is just fine for the time being.

# In[23]:
from lasagne.regularization import regularize_layer_params_weighted, l2
def training_function(network, input_tensor, target_tensor, learning_rate, use_l2_regularization=False, l2_lambda=0.01):
    """
    Training function.
    
    Args:
        network (lasagne.layers.Layer): Output layer.
        input_tensor (theano.tensor.ftensor4): Input tensor.
        target_tensor (theano.tensor.ivector): Target tensor.
        
    Returns:
        function: Network update function. It accepts [input_tensor, target_tensor] tensors as input and outputs
           [loss, l2_loss, accuracy] values.
    """
    
    # Get the network output and calculate metrics.
    #
    layers = {network.input_layer.input_layer.input_layer: l2_lambda, network.input_layer.input_layer.input_layer.input_layer: l2_lambda}
    network_output = lasagne.layers.get_output(network)
    l2_penalty = regularize_layer_params_weighted(layers, l2)
    loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() + l2_penalty
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), target_tensor), dtype=theano.config.floatX)

    # Get the network parameters and the update function.
    #
    network_params  = lasagne.layers.get_all_params(network, trainable=True)
    weight_updates  = lasagne.updates.adam(loss, network_params)

    # Construct the training function.
    #
    return theano.function([input_tensor, target_tensor], [loss, accuracy], updates=weight_updates)


# In[24]:

def validate_function(network, input_tensor, target_tensor):
    """
    Validation function.
    
    Args:
        network (lasagne.layers.Layer): Output layer.
        input_tensor (theano.tensor.ftensor4): Input tensor.
        target_tensor (theano.tensor.ivector): Target tensor.
        
    Returns:
        function: Network validation function. It accepts [input_tensor, target_tensor] tensors as input and outputs
           [loss, accuracy] values.
    """

    # Get the network output and calculate metrics.
    #
    network_output = lasagne.layers.get_output(network, deterministic = True)
    loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() # + l2_loss
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), target_tensor), dtype=theano.config.floatX)
    
    # Construct the validation function.
    #
    return theano.function([input_tensor, target_tensor], [loss, accuracy],allow_input_downcast=True)


# In[25]:

def evaluate_function(network, input_tensor):
    """
    Evaluation function.
    
    Args:
        network (lasagne.layers.Layer): Output layer.
        input_tensor (theano.tensor.ftensor4): Input tensor.
        
    Returns:
        function: Network evaluation function. It accepts [input_tensor] tensor as input and outputs the network
           prediction [prediction].
    """
    
    # Get the network output and calculate metrics.
    #
    network_output = lasagne.layers.get_output(network, deterministic=True)
    
    # Construct the evaluation function.
    #
    return theano.function([input_tensor], network_output)


# ### Training
# Now you can define a function that does trains the convnet by updating parameters for each mini-batch. This function will have to inculde the two main steps that we implemented last week: (1) a pass over the training set, to update the parameters, and (2) a pass over the validation set, to check the performance. This will be repeated *n_epochs* time.
# During training/validation, you will have to store the loss and accuracy values, in order to visualize them after each epoch in a plot that shows the learning curves. This is useful to monitor the training procedure.
# Note that all these steps have been implemented in the previous assignment, you can reuse a lot of that code!

# In[26]:
import sklearn
def train_convnet(network,
                  train_x,
                  train_y,
                  validation_x,
                  validation_y,
                  n_epochs,
                  network_name,
                  training_fn,
                  validation_fn,
                  training_batch_size,
                  validation_batch_size,
                  plot_curves=True):
    """
    Train the given network.
    
    Args:
        network (lasagne.layers.Layer): Output layer.
        train_x (numpy.ndarray): Training images.
        train_y (numpy.ndarray): Training labels.
        validation_x (numpy.ndarray): Validation images.
        validation_y (numpy.ndarray): Validation labels.
        n_epochs (int): Number of epochs.
        network_name (str): Name used to identify experiment.
        traing_fn (function): Training function.
        validation_fn (function): Validation function.
        training_batch_size (int): Training batch size.
        validation_batch_size (int): Validation batch size.
        plot_curves (bool): Plot curves flag.
    """
    print training_batch_size
    print len(train_y)
    n_batch_train = len(train_y)/training_batch_size # number of training mini-batches given the batch_size
    n_batch_val   = len(val_y)/validation_batch_size 
    # lists where we will be storing values during training, for visualization purposes
    tra_losses = []
    tra_accs   = []
    val_losses = []
    val_accs   = []
    
    # we want to save the parameters that give the best performance on the validation set
    # therefore, we store the best validation accuracy, and save the parameters to disk
    best_val_acc = 0
    # loop over the number of epochs
    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    for epoch in xrange(n_epochs):
        # training
        print epoch
        train_x, train_y = sklearn.utils.shuffle(train_x,train_y)
        cum_tra_loss = 0.0 # cumulative training loss
        cum_tra_acc = 0.0
        start = time.time()
        for b in range(n_batch_train-1):
            x_batch = train_x[b*training_batch_size:(b+1)*training_batch_size,:].astype(np.float32) # extract a mini-batch from x_train
            y_batch = train_y[b*training_batch_size:(b+1)*training_batch_size] # extract labels for the mini-batch
            mini_batch_loss, mini_batch_acc = training_fn(x_batch, y_batch)
            cum_tra_loss += mini_batch_loss
            cum_tra_acc += mini_batch_acc
        cum_tra_loss /= float(n_batch_train)
        cum_tra_acc /= float(n_batch_train)
        
#        # validation
        cum_val_loss = 0.0
        cum_val_acc = 0.0
        for b in range(n_batch_val-1):
            x_batch = validation_x[b*validation_batch_size:(b+1)*validation_batch_size,:].astype(np.float32) # extract a mini-batch from x_train
            y_batch = validation_y[b*validation_batch_size:(b+1)*validation_batch_size] # extract labels for the mini-batch
            val_loss, val_acc = validation_fn(x_batch, y_batch)
            cum_val_loss += val_loss
            cum_val_acc  += val_acc
        cum_val_acc  /= float(n_batch_val)
        cum_val_loss /= float(n_batch_val)
        print 'Took {:.1f} min. Now validating'.format((time.time()-start)/60)
#        cum_val_loss, cum_val_acc = validation_fn(validation_x.astype(np.float32), validation_y.astype(np.float32))
        # if the accuracy improves, save the network parameters
        
        if cum_val_acc > best_val_acc:
            best_val_acc = cum_val_acc
            # save network
            params = lasagne.layers.get_all_param_values(network)
            with open('./'+ network_name +'.npz', 'w') as f:
                np.savez(f, params=params)

        # add to lists
        tra_losses.append(cum_tra_loss)
        tra_accs.append(cum_tra_acc)
        val_losses.append(cum_val_loss)
        val_accs.append(cum_val_acc)
        print('Train: {:.1f}  Val: {:.1f}, Best-Val:{:.1f} '.format( cum_tra_acc*100, cum_val_acc*100, best_val_acc*100))
        plt.subplot(1,2,1)
        tra_loss_plt = plt.plot(range(len(tra_losses)), tra_losses, 'b')
        val_loss_plt = plt.plot(range(len(val_losses)), val_losses, 'g')
        plt.legend([tra_loss_plt[0],val_loss_plt [0]], ['Training loss', 'Validation loss'], loc='center right', bbox_to_anchor=(1, 0.5))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Losses')
        plt.subplot(1,2,2)
        tra_acc_plt  = plt.plot(range(len(tra_accs)), tra_accs, 'b')
        val_acc_plt  = plt.plot(range(len(val_accs)), val_accs, 'g')
        plt.legend([tra_acc_plt[0],val_acc_plt [0]], ['Training acc', 'Validation acc'], loc='center right', bbox_to_anchor=(1, 0.5))
        plt.xlabel('epoch')
       
                  
        plt.title('Accuracy (Best-Val {:.2f}%)'.format(100. * best_val_acc))
        plt.pause(0.001)


        # Your code. Hint: you can copy your solution from the last assignment.
        pass


# ## Task 2: Train your network
# Now that you have defined all the parameters and the functions necessary to train and validate your network, use this cell to run your exepriment. Define a *network_name*, which will be used to (1) save the parameters of the trained network to disk and (2) save a csv file to submit to challenger. Since you will be running several experiments and reusing the same cell (copy-paste) several times, having a name for the network used in each experiment is handy!

# In[ ]:

# Train your network

# Load data.
#
train_x, train_y, val_x, val_y, tes_x, tes_filenames = load_data()

# Define parameters
#
tra_batch_size = 200
val_batch_size = 200
n_epochs = 100
learning_rate = 0

network_name = 'network_task_4'

# Define the symbolic input X and the symbolic target y.
#
inputs = T.tensor4('X')
targets = T.ivector('y')

# Build the network.
#
network = build_network(input_tensor=inputs) # THIS FUNCTION MAY CHANGE IN FUTURE EXPERIMENTS!

# Define functions.
#
train_network = training_function(network=network, input_tensor=inputs, target_tensor=targets, learning_rate=learning_rate)
validate_network = validate_function(network=network, input_tensor=inputs, target_tensor=targets)

# Train the network.
#
train_convnet(network,
              train_x,
              train_y,
              val_x,
              val_y,
              n_epochs,
              network_name,
              training_fn=train_network,
              validation_fn=validate_network,
              training_batch_size=tra_batch_size,
              validation_batch_size=val_batch_size,
              plot_curves=True)


# ### Test and submit results to challenger
# Now you can run the network on the test set, get the predicted labels, and submit the results to challenger for evaluation. Use the following code (copy-paste) also for future tasks.

# In[ ]:

# Test and submit to challenger

# laod data (if not done before)
train_x, train_y, val_x, val_y, tes_x, tes_filenames = load_data()

# indicate the name of the network for this test
network_name = 'network_task_4'

# initialize the network used in this experiment (this may change)
network = build_network(inputs)
npz = np.load('./'+network_name+'.npz') # load stored parameters
lasagne.layers.set_all_param_values(network, npz['params']) # set parameters

# initialize tensor variable
# (in case you run this cell after training and after restarting the notebook)
#inputs = T.tensor4('X')

# compile evaluate_function()
evaluate_network = evaluate_function(network=network, input_tensor=inputs)

# classify the test set
test_predictions = list()
for b in np.arange(100):
    batch_x = tes_x[b*100:(b+1)*100,:].astype(np.float32)
    test_predictions.append( evaluate_network(batch_x))
test_predictions = np.vstack(test_predictions)
tes_y = np.argmax(test_predictions, axis=1)

# write csv files with outputs
ho = open('./results_{}.csv'.format(network_name), 'w')
ho.write('filename, label\n')
for filename, label in zip(tes_filenames, tes_y):
    ho.write('{}, {}\n'.format(filename, label))
ho.close()
sub = raw_input('Submit? Y/N: ')
if sub.lower() == 'y':
# submit to challenger
    challenger.submit_results({'username': 'S.Kern',
                           'password': '5CCN6PW2'},
                          'results_{}.csv'.format(network_name),
                          {'notes': 'Augmented Convolution L2'})


