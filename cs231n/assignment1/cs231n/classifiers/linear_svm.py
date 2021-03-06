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

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW /=num_train
  dW += W * reg

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



  for j in range(num_classes):
    indexesNonZeros = np.where(tScores[:,j]>0)
    xItems = X[indexesNonZeros]
    yItems = y[indexesNonZeros]
    dataSum = np.sum(xItems,axis=0)
    dW[:,j]+= dataSum

    # values,groups = sum_by_group(xItems,yItems)
    # dW[:,groups]-=values.T

    groups = np.unique(yItems)
    for idx,group in enumerate(groups):
      indexesToSum = np.where(yItems == group)
      dW[:,group] -= np.sum(xItems[indexesToSum], axis=0)


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += W * reg

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def sum_by_group(values, groups):
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values = np.cumsum(values,axis=0)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups