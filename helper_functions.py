# -*- coding: utf-8 -*-
"""Helper_Functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15G2a1aRO5RnhznSNF_9hQWcRhfVrHLWX
"""

import numpy as np

"""## **Unflattening the variable**"""

def unflatten(A_prev, cache):
  A = cache
  a_prev = A_prev.T
  A_unflat = a_prev.reshape(A.shape)
  return A_unflat


"""## **One Hot Encoding Function**"""

def one_hot_encoding(A):
    
  A = np.squeeze(A.T)
  a_onehot = np.zeros((A.size , A.max() + 1))
  a_onehot[np.arange(A.size), A] = 1
  return a_onehot

"""## **Initialise Parameters function**"""

def initialize_parameters(w1_s, w2_s, w3_s, w4_s, w5_s):
    
  """ w1_s/w2_s/w3_s is of the form (f,f,n_C_prev,n_C)
      w4_s is of the form (n_hidden_units, length of output from prev layer)
      w5_s is of the form (n_output, length of output from prev layer )

      b1_s/b2_s/b3_s is of the form (1,1,1,n_C)
      b4_s is of the form (n_hidden_units, 1)
      b5_s is of the form (n_output, 1)
  """
  np.random.seed(42)
    
  W1 = np.random.randn(w1_s[0], w1_s[1], w1_s[2], w1_s[3]) * 0.01
  b1 = np.zeros(shape=(1, 1, 1, w1_s[3]))

  W2 = np.random.randn(w2_s[0], w2_s[1], w2_s[2], w2_s[3] ) * 0.01
  b2 = np.zeros(shape=(1,1,1, w2_s[3]))
  
  W3 = np.random.randn(w3_s[0], w3_s[1], w3_s[2], w3_s[3]) * 0.01
  b3 = np.zeros(shape=(1,1,1, w3_s[3]))

  W4 = np.random.randn(w4_s[0], w4_s[1]) * 0.01
  b4 = np.zeros(shape=(w4_s[0], 1))

  W5 = np.random.randn(w5_s[0], w5_s[1]) * 0.01
  b5 = np.zeros(shape=(w5_s[0], 1))
    
  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3,
                "W4": W4,
                "b4": b4,
                "W5": W5,
                "b5": b5}
    
  return parameters


"""## **Initialize hparameter**"""

def initialize_hparameters(f, stride, pad):
  hparameters = {"f": f,
                 "stride": stride,
                 "pad": pad
                 }
  return hparameters
  
"""## **Update Parameters dictionary**"""

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters