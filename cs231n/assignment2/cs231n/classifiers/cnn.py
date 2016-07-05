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


  def loss(self,X,y=None):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores, cache1 = conv_forward_fast(X, W1, b1, conv_param)
    scores, cacheRelu1 = relu_forward(scores)
    scores, cachePool = max_pool_forward_fast(scores, pool_param)
    scores, cache2 = affine_forward(scores, W2, b2)
    scores, cacheRelu2 = relu_forward(scores)
    scores, cache3 = affine_forward(scores, W3, b3)

    if y is None:
      return scores

    loss, grads = 0, {}

    loss, dx = softmax_loss(scores, y)

    loss += 0.5 * self.reg * (np.sum(W3 * W3) + np.sum(W2 * W2) + np.sum(W1 * W1))

    # conv - relu - 2x2 max pool - affine - relu - affine - softmax

    dx, dw3, db3 = affine_backward(dx, cache3)
    grads['W3'] = dw3 + W3 * self.reg
    grads['b3'] = db3

    dx = relu_backward(dx, cacheRelu2)

    dx, dw2, db2 = affine_backward(dx, cache2)
    grads['W2'] = dw2 + W2 * self.reg
    grads['b2'] = db2

    dx = max_pool_backward_fast(dx, cachePool)

    dx = relu_backward(dx, cacheRelu1)

    dx, dw1, db1 = conv_backward_fast(dx, cache1)
    grads['W1'] = dw1 + W1 * self.reg
    grads['b1'] = db1

    return loss, grads





class MultipleLayersConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32),
               convolutions=[{'filter_size': 3, 'num_filters': 32},{'filter_size': 3, 'num_filters': 32}],
               hiddenDimensions=[500, 500], num_classes=10, weight_scale=1e-3, reg=0.0,
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
    self.bn_params={}
    self.reg = reg
    self.dtype = dtype
    currentInput_channels = input_dim[0]
    currentInput_width = input_dim[2]
    currentInput_height = input_dim[1]

    self.MaxPoolSize = 2
    self.MaxPoolStride = 2
    self.Convolutions = convolutions
    self.HiddenDimensions = hiddenDimensions

    for idx,convolutionItem in enumerate(convolutions):
      itemIndex = str(idx + 1)
      filter_size = convolutionItem['filter_size']
      num_filters = convolutionItem['num_filters']

      pad = (filter_size - 1) / 2
      stride = 1

      self.params['Conv_W'+ itemIndex] = weight_scale * np.random.randn(num_filters,currentInput_channels, filter_size, filter_size)
      self.params['Conv_b' + itemIndex] = np.ones(num_filters)
      self.params['Conv_gamma' + itemIndex] = np.ones(num_filters)
      self.params['Conv_beta' + itemIndex] = np.zeros(num_filters)
      bn_param = {'mode': 'train'}
      self.bn_params['Conv_bn_param'+itemIndex] = bn_param

      currentInput_width_beforePooling = int((currentInput_width + 2 * pad - filter_size) / stride + 1)
      currentInput_height_beforePooling = int((currentInput_height + 2 * pad - filter_size) / stride + 1)

      currentInput_width = int((currentInput_width_beforePooling - self.MaxPoolSize)/ self.MaxPoolStride +1)
      currentInput_height = int((currentInput_height_beforePooling - self.MaxPoolSize) / self.MaxPoolStride + 1)
      currentInput_channels = num_filters

    currentInput_dim =  currentInput_channels * currentInput_width * currentInput_height

    for idx,hiddenDimmension in enumerate(hiddenDimensions):
      itemIndex = str(idx + 1)
      self.params['W'+itemIndex] = weight_scale * np.random.randn(currentInput_dim, hiddenDimmension)
      self.params['b'+itemIndex] = np.ones(hiddenDimmension)
      self.params['gamma' + itemIndex] = np.ones(hiddenDimmension)
      self.params['beta' + itemIndex] = np.zeros(hiddenDimmension)
      bn_param = {'mode': 'train'}
      self.bn_params['bn_param' + itemIndex] = bn_param
      currentInput_dim = hiddenDimmension

    itemIndex = str(len(hiddenDimensions) + 1)
    self.params['W' + itemIndex] = weight_scale * np.random.randn(currentInput_dim, num_classes)
    self.params['b' + itemIndex] = np.ones(num_classes)



    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)



  def loss(self,X, y=None):

    scores = X
    wSquaredSum = 0
    cachedData = {}

    for idx, convolutionItem in enumerate(self.Convolutions):
      itemIndex = str(idx + 1)
      W, b, gamma, beta , bn_param = self.params['Conv_W' + itemIndex], self.params['Conv_b' + itemIndex],self.params['Conv_gamma' + itemIndex],self.params['Conv_beta' + itemIndex],self.bn_params['Conv_bn_param' + itemIndex]
      filter_size = convolutionItem['filter_size']

      conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

      oParam = conv_param, pool_param, gamma, beta, bn_param
      scores, cache = conv_batch_relu_pool_forward(scores,W,b,oParam, reluType = 'LeakyRelu')
      cachedData['Conv_cache_'+ itemIndex] = cache
      wSquaredSum+=np.sum(W*W)

    for idx,hiddenDim in enumerate(self.HiddenDimensions):
      itemIndex = str(idx + 1)
      W, b, gamma, beta, bn_param = self.params['W' + itemIndex], self.params['b' + itemIndex], self.params[
        'gamma' + itemIndex], self.params['beta' + itemIndex], self.bn_params['bn_param' + itemIndex]

      oParam = gamma, beta, bn_param
      scores, cache = affine_batch_relu_forward(scores, W, b, oParam, reluType = 'LeakyRelu')
      cachedData['cache_' + itemIndex] = cache
      wSquaredSum += np.sum(W * W)

    itemIndex = str(len(self.HiddenDimensions) + 1)
    W, b = self.params['W' + itemIndex],self.params['b' + itemIndex]
    scores, cache = affine_forward(scores, W, b)


    if y is None:
      return scores

    loss, grads = 0, {}
    loss, dx = svm_loss(scores, y)
    loss += 0.5 * self.reg * (wSquaredSum)


    dx, dw, db = affine_backward(dx, cache)
    grads['W'+itemIndex] = dw + W * self.reg
    grads['b'+itemIndex] = db

    for idx in range(len(self.HiddenDimensions) - 1, -1, -1):
      itemIndex = str(idx + 1)
      W = self.params['W' + itemIndex]
      cache = cachedData['cache_' + itemIndex]
      dx, dw, db, dgamma, dbeta = affine_batch_relu_backward(dx, cache)
      grads['W' + itemIndex] = dw + W * self.reg
      grads['b' +itemIndex] = db
      grads['gamma' + itemIndex] = dgamma
      grads['beta' + itemIndex] = dbeta

    for idx in range(len(self.Convolutions) - 1, -1, -1):
      itemIndex = str(idx + 1)
      W = self.params['Conv_W' + itemIndex]
      cache = cachedData['Conv_cache_' + itemIndex]
      dx, dw, db, dgamma, dbeta = conv_batch_relu_pool_backward(dx, cache)
      grads['Conv_W' + itemIndex] = dw + W * self.reg
      grads['Conv_b' + itemIndex] = db
      grads['Conv_gamma' + itemIndex] = dgamma
      grads['Conv_beta' + itemIndex] = dbeta


    return loss, grads
  


pass
