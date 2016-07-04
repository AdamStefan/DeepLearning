import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.ones(num_filters)
    pad = (filter_size - 1) / 2
    stride = 1
    # self.params['W1Pad'] = pad
    # self.params['W1Stride'] = stride
    self.MaxPoolSize = 2
    self.MaxPoolStride = 2

    outputSizeWidthCl = int((input_dim[2] + 2 * pad - filter_size) / stride + 1)
    outputSizeHeightCl = int((input_dim[1] + 2 * pad - filter_size) / stride + 1)


    outputSizeWidthAfterPool = int((outputSizeWidthCl - self.MaxPoolSize)/ self.MaxPoolStride +1)
    outputSizeHeightAfterPool = int((outputSizeHeightCl - self.MaxPoolSize) / self.MaxPoolStride + 1)

    self.params['W2'] = weight_scale * np.random.randn(num_filters * outputSizeHeightAfterPool * outputSizeWidthAfterPool, hidden_dim)
    self.params['b2'] = np.ones(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.ones(num_classes)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores, cache1 = conv_forward_fast(X,W1,b1, conv_param)
    scores , cacheRelu1 = relu_forward(scores)
    scores , cachePool = max_pool_forward_fast(scores,pool_param)
    scores, cache2 = affine_forward(scores,W2,b2)
    scores, cacheRelu2 = relu_forward(scores)
    scores, cache3 = affine_forward(scores,W3,b3)

    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dx = softmax_loss(scores, y)

    loss += 0.5 * self.reg * (np.sum(W3 * W3) + np.sum(W2 * W2) + np.sum(W1 * W1))

    # conv - relu - 2x2 max pool - affine - relu - affine - softmax

    dx, dw3, db3 = affine_backward(dx, cache3)
    grads['W3'] = dw3 + W3 * self.reg
    grads['b3'] = db3

    dx  = relu_backward(dx, cacheRelu2)

    dx, dw2, db2 = affine_backward(dx, cache2)
    grads['W2'] = dw2 + W2 * self.reg
    grads['b2'] = db2

    dx = max_pool_backward_fast(dx, cachePool)

    dx = relu_backward(dx, cacheRelu1)

    dx, dw1, db1 = conv_backward_fast(dx, cache1)
    grads['W1'] = dw1 + W1 * self.reg
    grads['b1'] = db1
    
    return loss, grads
  
  
pass
