import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """

    self.params = {}
    self.reg = reg

    self.params = {}
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)



  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """

    w1 = self.params['W1']
    b1 = self.params['b1']
    w2 = self.params['W2']
    b2 = self.params['b2']

    num_train = X.shape[0]

    scores = None

    score1, cache1 = affine_forward(X,w1,b1)

    scoresRelu, cacheRelu = relu_forward(score1)

    scores2,cache2 = affine_forward(scoresRelu,w2,b2)

    scores = scores2
    if y is None:
      return scores
    
    loss, grads = 0, {}


    loss,dx = softmax_loss(scores,y)
    loss+= 0.5 * self.reg * (np.sum(w2*w2) + np.sum(w1*w1))

    dx, dw2, db2  = affine_backward(dx,cache2)

    dx = relu_backward(dx, cacheRelu)

    dx, dw1, db1 = affine_backward(dx, cache1)



    grads['W2'] = dw2 + w2 * self.reg
    grads['b2'] = db2
    grads['W1'] = dw1+ w1 * self.reg
    grads['b1'] = db1

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None, useXavierInitialization=False):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # print ('hiddenDims :',hidden_dims)


    for idx,hidden_dim in enumerate(hidden_dims):
      i_dim = input_dim if idx==0 else self.params['W' + str(idx)].shape[1]
      if useXavierInitialization:
        self.params['W' + str(idx+1)] = np.sqrt(i_dim/2)  * np.random.randn(i_dim, hidden_dim)
      else:
        self.params['W' + str(idx + 1)] = weight_scale * np.random.randn(i_dim, hidden_dim)

      self.params['b'+ str(idx+1)] = np.zeros(hidden_dim)

      if (use_batchnorm) :
        self.params['gamma' + str(idx+1)] = np.ones(hidden_dim)
        self.params['beta' + str(idx + 1)] =  np.zeros(hidden_dim)



    self.params['W'+ str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[self.num_layers-2], num_classes)
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)


    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)




  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    data_input = X
    cachedData = {}

    for idx in range(self.num_layers-1):
      itemKey = str(idx + 1)

      w = self.params['W' + itemKey]
      b = self.params['b' + itemKey]


      scores, cache = affine_forward(data_input, w, b)
      cachedData[itemKey] = cache

      if (self.use_batchnorm) :
        gamma = self.params['gamma' + itemKey]
        beta = self.params['beta' + itemKey]
        scores, cache = batchnorm_forward(scores, gamma, beta ,self.bn_params[idx])
        cachedData[itemKey + "_batchNormalization"] = cache

      scores, cache = relu_forward(scores)
      cachedData[itemKey + "_activation"] = cache

      if (self.use_dropout):
        scores, cache = dropout_forward(scores,self.dropout_param)
        cachedData[itemKey + "_dropOut"] = cache

      data_input = scores



    w = self.params['W' + str(self.num_layers)]
    b = self.params['b' + str(self.num_layers)]

    scores, cache = affine_forward(scores, w, b)
    cachedData[str(self.num_layers)] = cache


    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    loss, dx = softmax_loss(scores, y)
    regParamSum =0
    for idx in range(self.num_layers):
      w = self.params['W' + str(idx + 1)]
      regParamSum += np.sum(w*w)


    loss += 0.5 * self.reg * regParamSum


    cache = cachedData[str(self.num_layers)]
    w = self.params['W' + str(self.num_layers)]

    dx, dw, db = affine_backward(dx, cache)
    dw += w * self.reg
    grads["W"+str(self.num_layers)] = dw
    grads["b" + str(self.num_layers)] = db

    for i in range(self.num_layers-2,-1,-1):
      itemKey = str(i + 1)

      if self.use_dropout:
        cache = cachedData[itemKey + "_dropOut"]
        dx = dropout_backward(dx, cache)


      cache = cachedData[itemKey+"_activation"]
      dx = relu_backward(dx, cache)

      if self.use_batchnorm:
        cache = cachedData[itemKey + "_batchNormalization"]
        dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache)
        grads["gamma" + itemKey] = dgamma
        grads["beta" + itemKey] = dbeta

      cache = cachedData[itemKey]
      dx, dw, db = affine_backward(dx, cache)

      w = self.params['W' + itemKey]
      dw += w * self.reg
      grads["W" + itemKey] = dw
      grads["b" + itemKey] = db



    return loss, grads
