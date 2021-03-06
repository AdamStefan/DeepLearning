import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    # first layer pass
    scoresW1 = X.dot(W1) + b1

    # second Layer pass

    # 2. ReLU for the above scores
    X1_Relu = np.maximum(0, scoresW1)

    #compute the second layer score
    scoresW2 = X1_Relu.dot(W2) + b2
    scores = scoresW2 # NRows of C size

    if y is None:
      return scores


    # do the softmax

    scaleFactorW2 = np.max(scoresW2, axis=1, keepdims=True)
    scoresW2 = scoresW2 - scaleFactorW2
    sumsW2 = np.sum(np.exp(scoresW2), axis=1)[:, np.newaxis]

    #end second Layer pass


    # Compute the loss
    loss = None

    loss = - np.sum(scoresW2[range(N), y]) + np.sum(np.log(sumsW2))
    loss /= N
    loss += 0.5 * reg * (np.sum(W2 * W2) + np.sum(W1 * W1))

    scoresW2 = (np.exp(scoresW2) / sumsW2)

    # Backward pass: compute gradients
    grads = {}

    #compute the second layer

    correct_scores_2 = np.zeros_like(scores) # NRows wiht C size
    correct_scores_2[range(N), y] = 1
    dScores_Correct = scoresW2 - correct_scores_2 # NRows with C size

    dW2 = X1_Relu.T.dot(dScores_Correct) # dW2 = dL/dW

    dW2 /= N
    dW2 += W2 * reg
    grads['W2'] = dW2

    dW2Output = dScores_Correct.dot(W2.T)
    dB2 = np.sum(dScores_Correct, axis=0)/ N

    #end compute the second layer

    #compute the Relu backward layer

    dRelu = np.zeros_like(scoresW1) # N Rows D
    indexes = np.where(scoresW1>0)
    dRelu[indexes] = 1  # it behaves like a gradient rooter
    dRelu = dRelu * dW2Output

    #end compute the Relu backward layer

    # compute the fist backward layer

    dW1 = X.T.dot(dRelu)

    dW1 /= N
    dW1 += W1 * reg

    db1 = np.sum(dRelu, axis=0)
    db1 /= N

    #end compute the first backward layer


    #end compute the backward layer


    grads['W1'] = dW1
    grads['b2'] = dB2
    grads['b1'] = db1


    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    for it in range(num_iters):
      randomIndexes = np.random.choice(X.shape[0], batch_size)
      X_batch = X[randomIndexes]
      y_batch = y[randomIndexes]


      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      self.params['W2'] -= grads["W2"]*  learning_rate
      self.params['W1'] -= grads["W1"] * learning_rate
      self.params['b1'] -= grads["b1"] * learning_rate
      self.params['b2'] -= grads["b2"] * learning_rate


      if verbose and it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    scores = self.loss(X)
    y_pred = np.argmax(scores,axis=1)
    # print (scores.shape,y_pred.shape)



    return y_pred


