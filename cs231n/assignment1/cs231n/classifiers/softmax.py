import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_samples = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_samples):
    tmpScore = X[i].dot(W)
    scaleFactor = np.max(tmpScore)
    tmpScore = tmpScore - scaleFactor
    tmpSum = np.sum(np.exp(tmpScore))
    loss += - tmpScore[y[i]]  + np.log(np.sum(np.exp(tmpScore)))


    for j in range(num_classes):
      if (j == y[i]):
        dW[:,j]+= - X[i] + (np.exp(tmpScore[y[i]])*X[i] / tmpSum)
      else:
        dW[:,j] +=np.exp(tmpScore[j])*X[i]/tmpSum



  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_samples
  dW += W * reg


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  num_samples = X.shape[0]

  scores = X.dot(W)
  scaleFactor = np.max(scores,axis=1)
  scores = scores - scaleFactor[:,np.newaxis]
  sums = np.sum(np.exp(scores),axis=1)[:,np.newaxis]

  loss= - np.sum(scores[range(num_samples),y]) + np.sum(np.log(sums))

  correct_softmax_scores = np.zeros_like(scores)
  correct_softmax_scores[range(num_samples), y] = 1

  scores = (np.exp(scores) / sums ) - correct_softmax_scores
  dW = X.T.dot(scores)

  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_samples
  dW += W * reg

  return loss, dW

