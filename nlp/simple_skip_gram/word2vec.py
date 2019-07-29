import options
import collections
import numpy as np
import math
settings = options.settings
from dataset import BatchGen, Dataset


class word2vec():
    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        self.dataset = Dataset(settings)
        
    def load_data(self, corpus):
        self.dataset.generate_training_data(corpus)


    def forward_pass(self, word_indices):
        # input   : 
        #           - word_indices (batch)
        # output  : 
        #           - probability of context words (batch, vocab_size) 
        #           - word_embeddings for word_indices (batch, n) 
        #           - unscaled score for context_words (batch, vocab_size)
        
        h = self.embeddings[word_indices]
        u = self.linear_dense(h)       
        y_c = self.softmax(u)

        return y_c, h, u

    def backprop(self, loss, h, y, y_pred, words_indices):
        # input  :
        #          - loss a scalar value representing the error
        #          - y (batch, context_words, self.v_count)
        #          - y_pred (batch, self.v_count)
        #          - h has shape (batch, N) 
        #          - words_indices (batch)

        #dy = self.cross_entropy_softmax_backward(y, y_pred)
        dy = self.logLikelihood_loss_backward(y, y_pred)

        dx,dw = self.linear_dense_backward(dy,h)

        self.embeddings[words_indices] = self.embeddings[words_indices] - (self.lr * dx)
        self.w = self.w - (self.lr * dw)
 


    def train(self):

        self.embeddings = np.random.uniform(-1, 1, (self.dataset.v_count, self.n))
        self.w          = np.random.uniform(-1, 1, (self.n, self.dataset.v_count))
  
        # Cycle through each epoch
        for i in range(self.epochs):
            # Intialise loss to 0
            self.loss = 0

            # Cycle through each training sample
            # w_t = vector for target word, w_c = vectors for context words

            batches = BatchGen(self.dataset.data)
            for w_t, w_c in batches:
                w_t_indices =  np.argmax(w_t, axis=1)

                y_pred, h, u = self.forward_pass(w_t_indices) #predict which words are neaby
                #loss = self.cross_entropy_loss(w_c, y_pred)
                loss = self.logLikelihood_loss(w_c, y_pred)
                self.loss += loss

                # Backpropagation 
                self.backprop(loss, h, w_c, y_pred, w_t_indices)
                
            print('Epoch:', i, "Loss:", self.loss)


    def logLikelihood_loss(self, y, y_pred):
        # inputs :
        #           - y - true values (batch, w_contexts, vocab_size)
        #           - y_pred - predicted values (batch, vocab_size)
        # outputs : 
        #           - logLikelihood_loss- scalar

        batch = y.shape[0]
        epsilon = 0.00000000001
        loss = 0
        for i in range(batch):
            row_loss = np.sum(np.log(y[i] * y_pred[i]+epsilon)[np.newaxis,:],axis = 1)
            loss += -np.sum(row_loss)
        return loss



    def logLikelihood_loss_backward(self,  y, y_pred):
        # inputs :
        #           - y - true values (batch, w_contexts, vocab_size)
        #           - y_pred - predicted values (batch, vocab_size)
        # outputs : 
        #           - dy the gradients with respect to unscaled input x (batch, vocab_size) 

        dy =  y_pred - y
        dy = np.sum(dy, axis=1)
        return dy



    #def logLikelihood_loss_backward2(self,  yc, yn, y_pred):
    #    # inputs :
    #    #           - yc - context one hot value (batch, w_contexts, vocab_size)
    #    #           - yn - negative one hot value (batch, w_negative, vocab_size)
    #    #           - y_pred - predicted values (batch, vocab_size)
    #    # outputs : 
    #    #           - dy the gradients with respect to unscaled input x (batch, vocab_size) 

    #    dy =  y_pred - y  #positive part
    #    dy = np.sum(dy, axis=1)

    #    dy_n = y_pred * yn.shape[1] - y_pred * (
    #    return dy


    def cross_entropy_loss(self, y, y_pred):
        # inputs :
        #           - y - true values (batch, w_contexts, vocab_size)
        #           - y_pred - predicted values (batch, vocab_size)
        # outputs : 
        #           - cross entropy loss - scalar

        batch = y.shape[0]
        epsilon = 0.00000000001
        loss = 0
        for i in range(batch):
            row_loss = np.sum(y[i]*np.log(y_pred[i]+epsilon)[np.newaxis,:],axis = 1)
            loss += -np.sum(row_loss)
        return loss/ (batch*y.shape[1])


    def cross_entropy_softmax_backward(self,  y, y_pred):
        # inputs :
        #           - y - true values (batch, w_contexts, vocab_size)
        #           - y_pred - predicted values (batch, vocab_size)
        # outputs : 
        #           - dy the gradients with respect to unscaled input x (batch, vocab_size) 

        dy =  y_pred - y
        dy = np.sum(dy, axis=1)
        return dy


    def linear_dense(self, x):
        # x represents the embeding value for the target word with shape (batch,N)
        # inputs :
        #           - x the word embeddings (batch,N)
        # ouptut :
        #           - linear combination between x and w (N, vocab_size) -> (batch, vocab_size) 

        ret = np.dot(x, self.w) # ret shape is (batch, self.v_count)
        return ret


    def linear_dense_backward(self, dy, x):
        # inputs :
        #           - dy (batch, v_size) - backward gradient
        # ouptut :
        #           - dx gradient with respect to input x (batch, N) 
        #           - dw gradient with respect to w parameters (N,vocab_size)

        w = self.w        
        batch = dy.shape[0]

        dx = np.dot(w, dy.T).T
        dw = np.dot(x.T, dy)

        return dx, dw



    def softmax(self, x):
        # inputs : 
        #         - x (batch, dim)
        # outputs:
        #         - softmax (batch, dim)

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1) 





