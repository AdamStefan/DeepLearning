import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero


  # compute the loss and the gradient
  num_classes = W.shape[1]

  num_train = X.shape[0]
  loss = 0.0
  aa = 0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      if margin > 0:
        loss += margin
        # dW[:, j] += X[i]
        # dW[:, y[i]] -= X[i]

        for xi in range(dW.shape[0]):
          dW[xi, j] += X[i][xi]
          dW[xi, y[i]] -= X[i][xi]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /=num_train
  dW += dW * reg

  # Add regularization to the loss.


  return loss, dW



def svm_loss_vectorized(W, X, y, reg):

  dW = np.zeros(W.shape)  # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_score = scores[range(num_train),y]
  tScores = scores - correct_class_score[:,np.newaxis] + 1
  tScores[range(num_train),y] = 0
  tScores = np.maximum(0, tScores)
  loss = np.sum(tScores)

  for i in range(num_train):
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = tScores[i,j]
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += dW * reg

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
