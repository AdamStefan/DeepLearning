import numpy as np
import collections
import math


class Dataset:
    def __init__(self, settings):
         self.v_count = 0
         self.words_list = []
         self.word_index = {}
         self.index_word = {}
         self.window_size =  settings['window_size']
         self.data = None

    def generate_training_data(self, corpus):
        # Find unique word counts using dictonary
        word_counts = collections.defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        ## How many unique words in vocab? 9
        self.v_count = len(word_counts.keys())
        # Generate Lookup Dictionaries (vocab)
        self.words_list = list(word_counts.keys())
        # Generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # Generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # Cycle through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])
                # Cycle through context window
                w_context = []
                # Note: window_size 2 will have range of 5 values
                for j in range(i - self.window_size, i + self.window_size+1):
                    # Criteria for context word 
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
                    if j != i and j <= sent_len-1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j]) 
                        # training_data contains a one-hot representation of the target word and context words
                training_data.append([w_target, np.asarray(w_context)])
        self.data = np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialise a blank vector
        word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
        # Get ID of word from word_index
        word_index = self.word_index[word]
        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1
        return word_vec




class BatchGen:
    def __init__(self, data, train=True, batch_size =1):
       
        self.data = data
        self.m = len(data)
        self.num_batches = math.ceil(self.m/batch_size)
        self.start_idx = 0
        if train:
            #shuffle data
            self.rows = np.random.permutation(self.m)
        else:
            self.rows = np.array([i for i in range(self.m)])
        self.batch_size = batch_size
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx >= len(self.rows):
            raise StopIteration
        #print('batch', self.start_idx/self.opt['trainer']['batch_size'], '/', self.num_batches)
        indices = self.rows[self.start_idx:self.start_idx + self.batch_size]
        self.start_idx += self.batch_size

        wt_batch = []
        wc_batch = []

        for i in indices:
            w_t, w_c = self.data[i]
            wt_batch.append(w_t)
            wc_batch.append(w_c)

        return np.asarray(wt_batch), np.asarray(wc_batch)