import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)


  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  xInput = x.reshape(x.shape[0],-1)
   # out  = None
  out = xInput.dot(w) + b

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache

  xInput = x.reshape(x.shape[0], -1)
  dw = xInput.T.dot(dout)
  db = np.sum(dout, axis=0)
  dxOutput = dout.dot(w.T)
  dx = dxOutput.reshape(x.shape)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)

  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  tmp = np.zeros_like(dout)
  tmp[x>0] = 1
  dx = dout * tmp

  return dx


def leakyRelu_forward(x):
  out = np.maximum(0.01, x)

  cache = x
  return out, cache


def leakyRelu_backward(dout, cache):
  dx, x = None, cache
  tmp = np.ones_like(dout)
  tmp = tmp * 0.01
  tmp[x > 0] = 1
  dx = dout * tmp
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':

    sample_mean = np.mean(x,axis=0)
    sample_var = np.var(x,axis=0)
    xInput = (x - sample_mean)/(np.sqrt(sample_var + eps) )
    out = gamma * xInput + beta

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    cache = x, gamma, beta, eps

  elif mode == 'test':
    xInput = (x - running_mean) / (np.sqrt(running_var) + eps)
    out = gamma * xInput + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None

  x, gamma, beta, eps = cache

  sample_mean = np.mean(x, axis=0)
  sample_var = np.var(x, axis=0)

  coeff = (np.sqrt(sample_var + eps) )
  # print (coeff.shape)

  dx = (dout /coeff) * gamma
  N = x.shape[0]

  dldVar =  np.sum(dout * gamma *  (x - sample_mean) * (-0.5) * np.power((sample_var+ eps),-1.5),axis=0)
  dldMean = np.sum(-dout * gamma/coeff, axis=0) + dldVar * (-2/N) * np.sum(x-sample_mean,axis=0)
  dx = dx + (dldVar * 2 * (x - sample_mean) / N) + (dldMean / N)



  xInput = (x - sample_mean) / (np.sqrt(sample_var + eps) )

  dgamma = np.sum(xInput * dout, axis=0)
  dbeta = np.sum(dout,axis=0)


  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None

  x, gamma, beta, eps = cache
  N = x.shape[0]

  sample_mean_minusX = x- np.mean(x, axis=0)
  sample_var_plusEps = np.var(x, axis=0) + eps

  xInput = sample_mean_minusX / (np.sqrt(sample_var_plusEps))

  dgamma = np.sum(xInput * dout, axis=0)
  dbeta = np.sum(dout, axis=0)

  dx = (1. / N) * gamma * (sample_var_plusEps) ** (-1. / 2.) * (N * dout - np.sum(dout, axis=0)
                                                       - (sample_mean_minusX) * (sample_var_plusEps) ** (-1.0) * np.sum(dout * (sample_mean_minusX),
                                                                                                   axis=0))
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = (np.random.randn(*x.shape) <p)/p
    out = x * mask
  elif mode == 'test':
    # there is nothing to do
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = mask * dout
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  F = w.shape[0]  #F represents the number of Filters
  C = w.shape[1]  #C represents the number of Channel on Filter

  HH = w.shape [2] # splatial Height of filter
  WW = w.shape[3]  # splatial Width of filter

  N = x.shape[0]   #number of samples


  stride = conv_param['stride']
  pad = conv_param['pad']



  outputSizeWidth = int(((x.shape[3] + 2 * pad)  - WW) / stride + 1)
  outputSizeHeight = int(((x.shape[2]+ 2* pad) - HH) / stride + 1)

  out = np.zeros((N,F,outputSizeHeight,outputSizeWidth))
  xPadded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)


  for sample_index in range(N):
    # The Weight for F Filter is
    for filter in range(F): # for each Filter
      wPerFilterPerChannel = w[filter] # each filter contains C matrixes of HH * WW dimensions

      for i in range(outputSizeWidth):
        for j in range(outputSizeHeight):
          resultForFilter = 0
          for channel in range(C):
            dataToCompute = xPadded[sample_index,channel][j * stride: j * stride + HH, i * stride: i * stride + WW]
            resultForFilter += np.sum(dataToCompute  * wPerFilterPerChannel[channel])

          out[sample_index,filter][j , i] = resultForFilter + b[filter]


  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  - dout (N, K Filters, output height, output width)

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache

  K = dout.shape[1]
  N = dout.shape[0]
  C = x.shape[1]
  sample_rows = x.shape[2]
  sample_columns = x.shape[3]
  output_width = dout.shape[3]
  output_height = dout.shape[2]

  filter_width = w.shape[3]
  filter_height = w.shape[2]


  stride = conv_param['stride']
  pad = conv_param['pad']

  dxLocal = np.zeros((N,C,sample_rows,sample_columns))
  db = np.zeros_like(b)
  dw = np.zeros_like(w)


  for sample_index in range(N):
    for channel_index in range(C):
      for row in range(sample_rows):
        for column in range(sample_columns):
          for f in range (K):
            for filterActivationRow_index in range(output_height):
              for filterActivationColumn_index in range(output_width):
                wRow = row + pad - (filterActivationRow_index * stride)
                wColumn = column + pad - (filterActivationColumn_index * stride)
                if wRow>=0 and wRow <  filter_height and wColumn>=0 and wColumn<filter_width :
                  dxLocal[sample_index,channel_index,row,column] += dout[sample_index,f,filterActivationRow_index,filterActivationColumn_index] * w[f,channel_index,wRow,wColumn]

  for f in range(K):
    for sample_index in range(N):
      for filterRow_index in range(output_height):
        for filterColumn_index in range(output_width):
          db[f]+=dout[sample_index,f,filterRow_index,filterColumn_index]

  for sample_index in range(N):
    for f in range(K):
      for channel_index in range(C):
        for filterRow_index in range(filter_height):
          for filterColumn_index in range(filter_width):
            for filterActivationRow_index in range(output_height):
              for filterActivationColumn_index in range(output_width):

                rowXIndex = filterActivationRow_index * stride + filterRow_index -pad
                colXIndex = filterActivationColumn_index * stride + filterColumn_index -pad

                if rowXIndex >= 0 and rowXIndex < sample_rows and colXIndex >= 0 and colXIndex < sample_columns:
                  dw[f, channel_index, filterRow_index, filterColumn_index] += x[sample_index,channel_index,rowXIndex,colXIndex] * dout[sample_index, f,filterActivationRow_index,filterActivationColumn_index]



  dx = dxLocal
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None


  pool_width = pool_param['pool_width']
  pool_height = pool_param['pool_height']
  stride = pool_param['stride']

  sample_width = x.shape[3]
  sample_height = x.shape[2]
  N = x.shape[0]
  F = x.shape[1]

  outputSizeWidth = int((sample_width  - pool_width) / stride + 1)
  outputSizeHeight = int((sample_height - pool_height) / stride + 1)



  out = np.zeros((N, F, outputSizeHeight, outputSizeWidth))

  for sample_index in range(N):
    for activationFilter_index in range(F):
      for poolOutput_row in range(outputSizeHeight):
        for poolOutput_column in range(outputSizeWidth):
          dataToCompute = x[sample_index, activationFilter_index][poolOutput_row * stride: poolOutput_row * stride + pool_height, poolOutput_column * stride: poolOutput_column * stride + pool_width]
          out[sample_index,activationFilter_index][poolOutput_row,poolOutput_column] = np.max(dataToCompute)



  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None

  x, pool_param = cache

  pool_width = pool_param['pool_width']
  pool_height = pool_param['pool_height']
  stride = pool_param['stride']

  sample_width = x.shape[3]
  sample_height = x.shape[2]
  N = x.shape[0]
  F = x.shape[1]

  outputSizeWidth = int((sample_width - pool_width) / stride + 1)
  outputSizeHeight = int((sample_height - pool_height) / stride + 1)

  dx = np.zeros_like(x)

  # iterate to all items
  for sample_index in range(N):
    for activationFilter_index in range(F):
      for poolOutput_row in range(outputSizeHeight):
        for poolOutput_column in range(outputSizeWidth):
          dataToCompute = x[sample_index, activationFilter_index][
                          poolOutput_row * stride: poolOutput_row * stride + pool_height,
                          poolOutput_column * stride: poolOutput_column * stride + pool_width]

          arguments = np.unravel_index(np.argmax(dataToCompute), dataToCompute.shape)
          dx[sample_index, activationFilter_index][poolOutput_row * stride + arguments[0], poolOutput_column * stride +arguments[1]] += dout[sample_index,activationFilter_index,poolOutput_row,poolOutput_column]


  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """

  N, C, H, W = x.shape

  x = np.rollaxis(x, 1, 4)
  out, cache = batchnorm_forward(x.reshape(N * H * W, C), gamma, beta, bn_param)
  out = np.rollaxis(out.reshape(x.shape), 3, 1)

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape

  outroll = np.rollaxis(dout, 1, 4)
  # print(outroll.shape,N, C, H, W)
  outroll_newShape = outroll.reshape(N * H * W, C)
  dx, dgamma, dbeta = batchnorm_backward_alt(outroll_newShape, cache)
  tmp = dx.reshape(outroll.shape)
  dx = np.rollaxis(tmp, 3, 1)

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
