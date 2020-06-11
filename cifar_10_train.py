from conv_helper_functions import *
from Linear_helper_functions import *
from helper_functions import *

import numpy as np
from matplotlib import pyplot as plt

"""## **Importing the data**"""

from keras.datasets import cifar10

(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

print('Train: X=%s, y=%s' % (Xtrain.shape, Ytrain.shape))
print('Test: X=%s, y=%s' % (Xtest.shape, Ytest.shape))

#plot the dataset
#from matplotlib import pyplot
#for i in range(9):
#	pyplot.subplot(330 + 1 + i)
#	pyplot.imshow(Xtrain[i])
#pyplot.show()

print("Thetype of the input", type(Xtrain))
print("The shape of the input vector", Xtrain.shape)
print("The size of one single image", Xtrain[1, :, :, :].shape)
print("Thetype of the input", type(Ytrain))
print("The shape of the input vector", Ytrain.shape)
print(Ytrain[0:8,:])

"""The dataset contains Xtrain in the form of an numpy ndarray
>
and the Ytrain in the from of numbered labels [0, 9]

## **Defining a small Batch Size for faster debugging**
# Training examples = 500
## Testing examples = 100
"""

batch_size = 100
test_size = 50

Xtrain.shape, Ytrain.shape
XTrain = Xtrain[0:batch_size, :,:,:]
XTest = Xtest[0:test_size, :,:,:]

YTrain = Ytrain[0:batch_size,:]
YTest = Ytest[0: test_size, :]

print('Train: X=%s, y=%s' % (XTrain.shape, YTrain.shape))
print('Test: X=%s, y=%s' % (XTest.shape, YTest.shape))

np.unique(YTrain)

YTrain = one_hot_encoding(YTrain)
YTest = one_hot_encoding(YTest)

"""## **Test Functions for Helper Functions**
1) 7 functions in linear_helper_functions are perfectly working
>
2) 8 functions in conv_helper_functions are perfectly working
>
3) 3 functions in helper_functions are perfectly working
"""



"""## **CIFAR10 MODEL DEFINATION**
def CIFAR10_model(X, Y, learning_rate = 0.001, num_iterations = 50, print_cost = False):
"""

#hparameters = initialize_hparameters()
#parameters = initialize_parameters()

def CIFAR10_model(X, Y, learning_rate = 0.001, num_iterations = 50, print_cost = False):
  
  #initialise parameters and hparameters
  hparameters = initialize_hparameters(3, 1, 1)
  parameters = initialize_parameters((3,3,3,32), (3,3,32,64), (3,3,64,128), (256, 86528), (10, 256))

  # Retrive parameters from parameters dictionary
  W1 = parameters['W1']
  W2 = parameters['W2']
  W3 = parameters['W3']
  W4 = parameters['W4']
  W5 = parameters['W5']
  
  b1 = parameters['b1']
  b2 = parameters['b2']
  b3 = parameters['b3']
  b4 = parameters['b4']
  b5 = parameters['b5']

  grads = {}
  costs = []
  m = X.shape[0]

  for i in range(0, num_iterations):

    # Forward Propagation - Conv1->relu->pool1->Conv2->relu->pool2->Conv3->relu->pool3->flatten->lin->relu->lin->sigmoid
    
    # Convolution Layers
    A1, cache1 = conv_forward(X, W1, b1, hparameters)
    A2, cache2 = relu(A1)
    A3, cache3 = pool_forward(A2, hparameters, mode = 'max')
    A4, cache4 = conv_forward(A3, W2, b2, hparameters)
    A5, cache5 = relu(A4)
    A6, cache6 = pool_forward(A5, hparameters, mode = 'max')
    A7, cache7 = conv_forward(A6, W3, b3, hparameters)
    A8, cache8 = relu(A7)
    A9, cache9 = pool_forward(A8, hparameters, mode = 'max')

    #Flatten the previous output
    A10, cache10 = flatten(A9)

    #Linear Layers
    A11, cache11 = linear_activation_forward(A10, W4, b4, activation='relu')
    A12, cache12 = linear_activation_forward(A11, W5, b5, activation='sigmoid')

    ## Compute Cost
    cost = compute_cost(A12, Y)

    ## initilize the derivative wrt to A12 
    dA12 = - (np.divide(Y.T, A12) - np.divide(1 - Y.T, 1 - A12))

    ## backward propagation

    #linear backward propagation
    dA11, dW5, db5 = linear_activation_backward(dA12, cache12, activation= 'sigmoid')
    dA10, dW4, db4 = linear_activation_backward(dA11, cache11, activation='relu')
    
    # Unflatten the array to suite the backward prop for gradietns
    dA9 = unflatten(dA10, cache10)

    # Convolution Backward Propagation
    dA8 = pool_backward(dA9, cache9, mode = 'max')
    dA7 = relu_backward(dA8, cache8)
    dA6, dW3, db3 = conv_backward(dA7, cache7)
    dA5 = pool_backward(dA6, cache6, mode = 'max' )
    dA4 = relu_backward(dA5, cache5)
    dA3, dW2, db2 = conv_backward(dA4, cache4)
    dA2 = pool_backward(dA3, cache3, mode ='max')
    dA1 = relu_backward(dA2, cache2)
    dA0, dW1, db1 = conv_backward(dA1, cache1)

    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2
    grads['dW3'] = dW3
    grads['db3'] = db3
    grads['dW4'] = dW4
    grads['db4'] = db4
    grads['dW5'] = dW5
    grads['db5'] = db5

    ##Update Parameters
    parameters = update_parameters(parameters, grads, learning_rate= learning_rate)

    ## Retrive new parameters W1, b1.... from parameters dictionary in above step
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
  
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']
    b4 = parameters['b4']
    b5 = parameters['b5']
    
    #print cost every 10 iterations
    if print_cost:
      print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
    costs.append(cost)

  plt.plot(np.squeeze(costs))
  plt.ylabel('cost')
  plt.xlabel('epoch (per hundred)')
  plt.title('Learning Rate:', 0.001)
  plt.show()

  return parameters



"""## **TRAIN THE NETWORK**"""

parameters = CIFAR10_model(XTrain, YTrain, learning_rate= 0.001, num_iterations= 2, print_cost= True)


"""## **poop backwards function (Dummy because pool_backward gives wierd error**"""

def poop_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev



"""## **Predicting the output from the trained parameters**"""

def predict(parameters, hparameters, X, Y):

  W1 = parameters['W1']
  W2 = parameters['W2']
  W3 = parameters['W3']
  W4 = parameters['W4']
  W5 = parameters['W5']
  
  b1 = parameters['b1']
  b2 = parameters['b2']
  b3 = parameters['b3']
  b4 = parameters['b4']
  b5 = parameters['b5']

  m = X.shape[0]

  # Forward Propagation - Conv1->relu->pool1->Conv2->relu->pool2->Conv3->relu->pool3->flatten->lin->relu->lin->sigmoid
    
  # Convolution Layers
  A1, cache1 = conv_forward(X, W1, b1, hparameters)
  A2, cache2 = relu(A1)
  A3, cache3 = pool_forward(A2, hparameters, mode = 'max')
  A4, cache4 = conv_forward(A3, W2, b2, hparameters)
  A5, cache5 = relu(A4)
  A6, cache6 = pool_forward(A5, hparameters, mode = 'max')
  A7, cache7 = conv_forward(A6, W3, b3, hparameters)
  A8, cache8 = relu(A7)
  A9, cache9 = pool_forward(A8, hparameters, mode = 'max')

  #Flatten the previous output
  A10, cache10 = flatten(A9)
  
  #Linear Layers
  A11, cache11 = linear_activation_forward(A10, W4, b4, activation='relu')
  A12, cache12 = linear_activation_forward(A11, W5, b5, activation='sigmoid')

  
  return A12
