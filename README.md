# CNN_from_scratch
The repo is an implmentation of CNN from scratch in python using Numpy for classification of the standard CIFAR10 dataset.

## Requirements
This projects requires python3+, numpy and matplotlib. For importing the CIFAR10 dataset one can directly download using this [link](https://www.cs.toronto.edu/~kriz/cifar.html) or you can use keras.datasets to import cifar10.
```python
from keras.datasets import cifar10
(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
print('Train: X=%s, y=%s' % (Xtrain.shape, Ytrain.shape))
print('Test: X=%s, y=%s' % (Xtest.shape, Ytest.shape))
```

## Contents
This repo contains the following files:
1. conv_helper_functions.py - Contains all the helper functions for implementing Convolution Layers from scratch.
2. Linear_helper_functions.py - Contains all the helper functions for implementing Linear Layers from scratch.
3. Helper_functions.py - Contains all the helper functions for building a model.
4. CIFAR_10_Train.py - Contains the training and plotting loss for the model.

The README is further divided into subsections to explain briefly different modules of Training

## Block Structure Used for creating the model ![alt text](https://github.com/carankt/CNN_from_scratch/blob/master/Block%20Structure.png)
The above Figure shows the generalised version of the training structure used for training the network. In this repo, I have created various helper functions to complete the task.

```python
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
```
In the forward propagation, the input vector X is passed through various forward blocks which outputs the new activation and cache. The  cache is used to carry out Back-propagation and the new activation is now passed further to the next block.
After the completion of forward propagation, the model will have cache for each block and the final activation output of the final layer. The final activation is then passed on to compute Cost, a scalar with respect to Y (The target variable). Followed by initialisation of derivative of last activation with respect to cost. In our case, the cost is cross entropy loss.![alt text](https://github.com/carankt/CNN_from_scratch/blob/master/Loss%20Function.png)

Using Simple calculus we can initialize the *dA12* as 
```python
dA12 = - (np.divide(Y.T, A12) - np.divide(1 - Y.T, 1 - A12))
```
For Back Propagating the derivative, the *dA12* is passed to corresponding backward propagation blocks that only saves the parameters necesarry for updating weights. 
```python
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
```
The gradients corresponding to the weights are saved in the grads dictionary and then passed through *update_parameters* function located in the helper_functions.py file. 
Finally all the updates parameters after the specified number of iterations are stored in a parameters dictionary (output of the model) and the corresponding cost for each iteration is plotted. 

## Output after training for two iterations
The aim of this repo is to demonstrate how one can implement a simple yet complex Conlovutional Network using numpy. This implementation, lacks speed and memory efficiency when compared with existing infrastructures like tensorflow, keras, pytorch, caffe and others. However, this repo is a good fit for a begineer who wants to test out basic concepts in a convolutional neural network.
![alt text](https://github.com/carankt/CNN_from_scratch/blob/master/Cost%20after%20two%20iterations.png)

We can see the cost is reducing after the first iteration. The functionalities of each individual blocks have also been tested using suitable test cases. Hence, we can say that the model is probably working fine.  

## Detailed Explanation to various helper functions used in the Repo
To make the README file look short and sweet, I have compiled the detailed explations on various helper function in the pdf. [Click here](https://github.com/carankt/CNN_from_scratch/blob/master/Detailed%20Explanation%20of%20helper%20functions%20used%20in%20this%20repo.pdf)

## References
1. Deep Learning Specialisation by Andrew Ng, Coursera
2. CS229 and CS230 - Stanford Engineering on Stanfordonline Youtube Channel
3. Stack overflow 
3. 
