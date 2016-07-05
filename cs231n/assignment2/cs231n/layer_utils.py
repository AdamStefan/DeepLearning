from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db



def conv_batch_relu_pool_forward(x, w, b, oParams, reluType = 'Standard'):
  conv_param, pool_params, gamma, beta, bn_param = oParams
  scores, cacheCnn = conv_forward_fast(x, w, b, conv_param)
  scores, cacheBatch =  spatial_batchnorm_forward(scores, gamma, beta, bn_param)


  if (reluType == 'LeakyRelu'):
    scores, cacheRelu = leakyRelu_forward(scores)
  else:
    scores, cacheRelu = relu_forward(scores)

  scores, cachePool = max_pool_forward_fast(scores, pool_params)
  cache = cacheCnn, cacheRelu, cachePool , reluType, cacheBatch

  return scores , cache


def conv_batch_relu_pool_backward(dout, cache):
  cacheCnn, cacheRelu, cachePool, reluType, cacheBatch = cache

  dx = max_pool_backward_fast(dout, cachePool)
  if (reluType == 'LeakyRelu'):
    dx = leakyRelu_backward(dx, cacheRelu)
  else:
    dx = relu_backward(dx, cacheRelu)

  dx, dgamma, dbeta = spatial_batchnorm_backward(dx,cacheBatch)
  dx, dw, db = conv_backward_fast(dx, cacheCnn)

  return dx, dw, db , dgamma, dbeta


def affine_batch_relu_forward(x, w, b, oParam, reluType = 'Standard'):
  gamma, beta, bn_param = oParam
  scores, cacheAffine = affine_forward(x,w,b)
  scores, cacheBatch = batchnorm_forward(scores,gamma,beta,bn_param)

  if (reluType == 'LeakyRelu'):
    scores, cacheRelu = leakyRelu_forward(scores)
  else:
    scores, cacheRelu = relu_forward(scores)

  cache = cacheAffine,cacheBatch,cacheRelu,reluType
  return scores, cache


def affine_batch_relu_backward(dout, cache):
  cacheAffine, cacheBatch, cacheRelu, reluType = cache
  if (reluType == 'LeakyRelu'):
    dx = leakyRelu_backward(dout,cacheRelu)
  else:
    dx = relu_backward(dout, cacheRelu)

  dx, dgamma, dbeta = batchnorm_backward(dx, cacheBatch)
  dx,dw,db = affine_backward(dx,cacheAffine)

  return dx,dw,db,dgamma,dbeta