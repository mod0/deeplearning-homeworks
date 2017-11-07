from __future__ import print_function
import sys
import copy
import numpy as np
import numpy.random as nr
import sklearn.metrics as skm
import matplotlib.pyplot as plt

###
# Dropouts with Relu Activation
###

def loaddata(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows = 1)
    return data

def sgd(no_of_layers, units_per_layer, dropout_probability, shuffle_order_training, train_data, shuffle_order_testing, test_data, w, dw, z, dz, a, y, eta, maxiter, early_termination):
    previous_weights             = None
    downscaled_weights           = None
    testing_accuracy             = []
    training_accuracy            = []
    total_testing_loss_fn_value  = []      
    total_training_loss_fn_value = []
    best_testing_loss            = 0

    # Keep a testing mask handy for early termination
    testing_mask = generate_masks_for_layers(no_of_layers, units_per_layer, np.zeros(no_of_layers))
    
    for epoch in xrange(maxiter):
        print("Stochastic Gradient Descent: Running epoch " + str(epoch))

        # Go one example at a time in random order and correct the weights
        for i in shuffle_order_training:
            # Generate masks for each layer
            masks = generate_masks_for_layers(no_of_layers, units_per_layer, dropout_probability)
            
            # Do the forward pass and the backward pass for
            # the example at the order[i]
            forward_prop(i, train_data, w, z, a, y, masks)
            backward_prop(i, train_data, w, dw, z, dz, a, y, masks)
            update_weights(w, dw, eta)
        print('\n')
        
        downscaled_weights = copy.deepcopy(w)
        for i, p in enumerate(dropout_probability):
            downscaled_weights[i] *= (1.0 - p)

        # Use the final set of weights at the epoch to compute the testing and training loss
        total_training_loss = 0.0    
        for i in shuffle_order_training:
            forward_prop(i, train_data, downscaled_weights, z, a, y, testing_mask)
            # Compute training loss
            total_training_loss += distance_lossfn(y, label_to_ytrue(len(y), train_data[i, -1]))

        total_testing_loss = 0.0    
        for i in shuffle_order_testing:
            # Forward propagate the y for test data
            forward_prop(i, test_data, downscaled_weights, z, a, y, testing_mask)
            # Compute testing loss
            total_testing_loss += distance_lossfn(y, label_to_ytrue(len(y), test_data[i, -1]))

        print("Total training loss in this epoch: " + str(total_training_loss))
        print("Total testing loss in this epoch: " + str(total_testing_loss))

        # Add to total loss fn value
        total_testing_loss_fn_value.append(total_testing_loss)
        total_training_loss_fn_value.append(total_training_loss)
        
        # Find the accuracy using current weights on train data
        training_accuracy.append(get_accuracy(shuffle_order_training, train_data, downscaled_weights, z, a, y, testing_mask))
        
        # Find the accuracy using current weights on test data
        testing_accuracy.append(get_accuracy(shuffle_order_testing, test_data, downscaled_weights, z, a, y, testing_mask))

        # Stop once testing loss begins to increase again
        if len(total_testing_loss_fn_value) == 1:
            # Save previous weights
            previous_weights  = copy.deepcopy(downscaled_weights)
            best_testing_loss = total_testing_loss_fn_value[0]
        elif (total_testing_loss < best_testing_loss):
            previous_weights  = copy.deepcopy(downscaled_weights)
            best_testing_loss = total_testing_loss
        elif early_termination:
            break

        w = downscaled_weights
        
    # Pack weights and exit
    model = dict()
    model['w']                                             = previous_weights
    model['training_accuracy']                             = training_accuracy
    model['testing_accuracy']                              = testing_accuracy
    model['total_training_loss_fn_value']                  = total_training_loss_fn_value
    model['total_testing_loss_fn_value']                   = total_testing_loss_fn_value

    return model

def generate_masks_for_layers(L, units, probs):
    """
    L     : Number of layers including the input
    units : Number of units in each layer
    prob  : the probability with which the units drop in some layer
    """
    masks = []
    
    for l in xrange(L):
        mask = (nr.rand(units[l]) >= probs[l])
        masks.append(mask)

    return masks

def doplots(title, data, label, xlabel, ylabel, semilogy = False):
    plt.figure(title)
    if not semilogy:
        plt.plot(data, label = label)
    else:
        plt.semilogy(data, label = label, basey = 2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('auto')
    plt.legend()
    plt.show()

def get_accuracy(shuffle_order, data, w, z, a, y, masks):
    pred_label = []
    true_label = data[:, -1]
    # first get training accuracy
    for i in shuffle_order:
        forward_prop(i, data, w, z, a, y, masks)
        pred_label.append(y_to_label(y))

    return (1.0 * sum(pred_label == true_label[shuffle_order]))/len(true_label)

def get_precision_recall_fscore(shuffle_order, data, w, z, a, y, masks):
    pred_label = []
    true_label = data[:, -1]
    # first get training accuracy
    for i in shuffle_order:
        forward_prop(i, data, w, z, a, y, masks)
        pred_label.append(y_to_label(y))

    return skm.precision_recall_fscore_support(true_label[shuffle_order], pred_label)
    
def forward_prop(example_id, data, w, z, a, y, masks):
    # Right now only making it work for the given MLP
    x         =  data[example_id,:-1]
    z[0][:]   =  w[0].dot(x * masks[0]) # Mask the input: effect is the input is non existant
    a[0][:]   =  relu(z[0])
    z[1][:]   =  w[1].dot(a[0] * masks[1]) # Mask the activation output: effect is node does not contribute
    a[1][:]   =  relu(z[1])
    z[2][:]   =  w[2].dot(a[1] * masks[2]) # Mask the activation output: effect is node does not contribute
    y[:]      =  softmax(z[2])


def backward_prop(example_id, data, w, dw, z, dz, a, y, masks):
    # Right now only making it work for the given MLP
    x             =  data[example_id,:-1]
    label         =  int(data[example_id, -1])
    ytrue         =  label_to_ytrue(len(y), label)
    L             =  distance_lossfn(y, ytrue, disp = True)
    L_g           =  distance_lossfn_grad(y, ytrue)
    dy_dz3        =  softmax_jac(z[2])
    dz[2][:]      =  L_g.dot(dy_dz3)
    dz[1][:]      =  relu_grad(z[1]) * (w[2].T.dot(dz[2]))
    dz[0][:]      =  relu_grad(z[0]) * (w[1].T.dot(dz[1]))
    dw[2][:,:]    =  np.outer(dz[2], a[1] * masks[2]) # Multiply the masked value
    dw[1][:,:]    =  np.outer(dz[1], a[0] * masks[1]) # Multiply the masked value
    dw[0][:,:]    =  np.outer(dz[0], x * masks[0])    # Multiply the masked value

    
def update_weights(w, dw, eta):
    for i in xrange(len(w)):
        w[i] = w[i] - eta * dw[i]
        
        
def distance_lossfn(y, ytrue, disp=False):
    lossfn = ((y - ytrue)**2).sum()
    if disp:
        sys.stdout.write('\rLoss: ' + str(lossfn) + ' ' * 20)
        sys.stdout.flush()
    return lossfn


def distance_lossfn_grad(y, ytrue):
    return 2 * (y - ytrue)


def label_to_ytrue(ny, label):
    label_ytrue = np.zeros(ny)
    label_ytrue[int(label)] = 1
    return label_ytrue


def y_to_label(y):
    return np.argmax(y)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    epsilon = 1e-8
    return  np.ceil(np.maximum(0, z)/(z + epsilon))

def softmax(z):
    # Numerically stable version of softmax:
    # Since we sum a bunch of exponentiated quantities, its possible for an overflow/underflow/nan
    # We subtract the maximum value from each entry and then compute the softmax on this
    # Note: This still preserves the ratio for each entry and is equivalent to
    # multiplying and dividing by the corresponding constant.
    # Citation: Eli Bendersky's website: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    maxz = np.max(z)
    newz = z - maxz
    expz = np.exp(newz)
    return expz/expz.sum()


def softmax_jac(z):
    # First compute the softmax
    Sz   = softmax(z)
    # Compute the outer product of softmax with itself
    Sz_g = np.outer(Sz, -Sz)
    # Fill zeros along the diagonals
    np.fill_diagonal(Sz_g, 0)
    # Now take the sum along rows
    Sz_g_row_sum = -np.sum(Sz_g, axis = 1)
    # Now set the diagonal to be the values in the sum vector above
    Sz_g = Sz_g + np.diag(Sz_g_row_sum)
    return Sz_g



def initialize(trainfile = 'train_data.txt', testfile = 'test_data.txt', eta = 0.01, maxiter = 30, early_termination = False):
    train_data = loaddata('train_data.txt')
    print("Loaded training data of shape: " + str(train_data.shape) + str(type(train_data)))

    test_data = loaddata('test_data.txt')
    print("Loaded testing data of shape: "+ str(test_data.shape) +  str(type(test_data)))
    
    params                              = dict()
    params['train_data']                = train_data
    params['test_data']                 = test_data
    params['w']                         = [nr.normal(size=(3, 2)), nr.normal(size=(3,3)), nr.normal(size=(2,3))]      
    params['dw']                        = [np.zeros((3, 2)), np.zeros((3,3)), np.zeros((2,3))]                       
    params['z']                         = [np.zeros(3), np.zeros(3), np.zeros(2)]
    params['dz']                        = [np.zeros(3), np.zeros(3), np.zeros(2)]
    params['a']                         = [np.zeros(3), np.zeros(3)]
    params['eta']                       = eta
    params['y']                         = np.zeros(2)
    params['shuffle_order_training']    = np.arange(train_data.shape[0])
    params['shuffle_order_testing']     = np.arange(test_data.shape[0])
    params['maxiter']                   = maxiter

    params['no_of_layers']              = 3
    params['units_per_layer']           = [2, 3, 3]
    params['dropout_probability']       = [0, 0, 1.0/3]

    params['early_termination']         = early_termination

    # randomize the order in which the examples will be accessed
    nr.shuffle(params['shuffle_order_training'])
    nr.shuffle(params['shuffle_order_testing'])
    
    return params
