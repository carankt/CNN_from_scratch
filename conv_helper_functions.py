# -*- coding: utf-8 -*-
"""Helper_Functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15G2a1aRO5RnhznSNF_9hQWcRhfVrHLWX
"""

import numpy as np

"""## **Zero Padding Function**"""

def zero_pad(X, pad):
  X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode = 'constant', constant_values = (0,0))
  return X_pad

"""## **Single Step Convolution**"""

def conv_single_step(a_slice_prev, W, b):
  s = np.multiply(a_slice_prev, W)
  Z = np.sum(s)
  Z = Z + float(b)
  return Z

"""## **Convolution Forward**"""

def conv_forward(A_prev, W, b, hparameters):

  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

  (f, f, n_C_prev, n_C) = W.shape

  stride = hparameters['stride']
  pad = hparameters['pad']

  n_H = int((n_H_prev - f + 2*pad)/stride) + 1
  n_W = int((n_W_prev - f + 2*pad)/stride) + 1

  Z = np.zeros((m, n_H, n_W, n_C))

  A_prev_pad = zero_pad(A_prev, pad)

  for i in range(m):
    a_prev_pad = A_prev_pad[i,:,:,:]
    for h in range(n_H):
      vert_start = h*stride
      vert_end = h*stride + f

      for w in range(n_W):
        horiz_start = w*stride
        horiz_end = w*stride + f

        for c in range(n_C):
          a_slice_prev = a_prev_pad[ vert_start:vert_end, horiz_start:horiz_end, :]

          weights = W[:,:,:,c]
          biases = b[:,:,:,c]
          Z[i,h,w,c] = conv_single_step(a_slice_prev, weights, biases)
  assert(Z.shape == (m,n_H, n_W, n_C))

  cache = (A_prev, W, b, hparameters)

  return Z, cache


"""## **Convolution Backward Pass**"""

def conv_backward(dZ, cache):

  (A_prev, W, b, hparameters) = cache 
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  (f,f,n_C_prev, n_C) = W.shape

  stride = hparameters['stride']
  pad = hparameters['pad']

  (m,n_H,n_W,n_C) = dZ.shape

  dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
  dW = np.zeros((f,f,n_C_prev, n_C))
  db = np.zeros((1,1,1,n_C))

  A_prev_pad = zero_pad(A_prev,pad)
  dA_prev_pad = zero_pad(dA_prev, pad)

  for i in range(m):
    a_prev_pad = A_prev_pad[i,:,:,:]
    da_prev_pad = dA_prev_pad[i,:,:,:]

    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):

          vert_start = h*stride
          vert_end = h*stride + f 
          horiz_start = w*stride
          horiz_end = w*stride + f

          a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

          da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[:,:,:,c]*dZ[i,h,w,c]
          dW[:,:,:,c] += a_slice*dZ[i,h,w,c]
          db[:,:,:,c] += dZ[i,h,w,c]

    dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]

  assert(dA_prev.shape == (m,n_H_prev, n_W_prev, n_C_prev))

  return dA_prev, dW, db

"""## **Pooling Backwards**

1) helper function to create mask


2) max pool backward
"""

def create_mask_from_window(x):
  mask = (x ==np.max(x))

  return mask

def distribute_value(dZ, shape):
  (n_H,n_W) = shape
  average = np.sum(dZ)/(n_H*n_W)
  a = np.ones((n_H,n_W))*average

  return a


"""## **Pool Forward Layer**"""

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (â‰ˆ2 lines)
            vert_start = h*stride
            vert_end = h*stride + f
            
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (â‰ˆ2 lines)
                horiz_start = w*stride
                horiz_end = w*stride +f
                
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (â‰ˆ1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end , horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    ### END CODE HERE ###
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache





def pool_backward(dA, cache, mode = "max"):
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