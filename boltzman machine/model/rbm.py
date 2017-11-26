import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class restricted_boltzman_machine:
    def __init__(self, n_visible, n_hidden):
        self.n_hidden = n_hidden # number of hidden units
        self.n_visible = n_visible # number of visible units
        self.init_placeHolders_and_variables()        

    def init_placeHolders_and_variables(self):
        self.v = tf.placeholder(tf.float32, shape=[None,self.n_visible], name="visible_layer")
        self.size = tf.cast(tf.shape(self.v)[0], tf.float32)
        # Initialize weights and biases a la Hinton
        with tf.name_scope('Weights'):
            random_normal = tf.random_normal([self.n_visible, self.n_hidden], mean=0., stddev=4 * np.sqrt(6. / (self.n_visible + self.n_hidden)))
            self.W = tf.Variable(random_normal, name="weights")
            tf.summary.histogram('weights',self.W)

        self.vb = tf.Variable(tf.zeros([1,self.n_visible]),tf.float32, name="visible_bias")
        self.hb = tf.Variable(tf.zeros([1,self.n_hidden]),tf.float32, name="hidden_bias")        

    def compute_rec_error_and_update(self, learning_rate):
        with tf.name_scope("Reconstruction_error"): 
            self.reconstruction_error = tf.Variable(0.0,name="reconstruction_error")
            tf.summary.scalar('reconstruction_error',self.reconstruction_error)

        #Positive divergence: $\mathbf{v_0^T \times p(h_0|v_0)}$
        #Sample hidden states from: $\mathbf{h_0 \sim p(h_0|v_0)}$.
        #Reconstruct visible units: $\mathbf{v_s \sim p(v_{s})=p(v_1|h_0)}$
        #Negative divergence: $\mathbf{p(v_{s})^T \times p(h_1|v_s)}$

        # K-step Contrastive Divergence using Gibbs sampling
        # Positive divergence  
        with tf.name_scope('Hidden_probabilities'):
            pos_hid_prob = get_probabilities(self.v,self.W,self.hb,True)
        with tf.name_scope('Positive_Divergence'):
            pos_divergence = tf.matmul(tf.transpose(self.v),pos_hid_prob)

        pos_hid_states = sample(pos_hid_prob)
        neg_vis_prob = get_probabilities(pos_hid_states, self.W, self.vb, False)

        # Negative divergence
        with tf.name_scope('Negative_hidden_probabilities'):
            neg_hid_prob = get_probabilities(neg_vis_prob, self.W, self.hb, True)
        with tf.name_scope('Negative_Divergence'):
            neg_divergence = tf.matmul(tf.transpose(neg_vis_prob),neg_hid_prob)

        a = tf.reduce_mean(tf.squared_difference(self.v,neg_vis_prob))
        rec_error = [self.reconstruction_error.assign(a)]

        # Update rules for weights and biases
        # Summaries of gradient for Tensorboard visualization
        #$w_{new} = w_{old} + \epsilon *$ (positive divergence - negative divergence)
        #$vb_{new} = vb_{old} + \epsilon * (v_0 - p(v_s))$
        #$vb_{new} = vb_{old} + \epsilon * (p(h_0) - p(h_1))$

        with tf.name_scope('Weight_gradient'):
            delta_w = tf.multiply(learning_rate/self.size, tf.subtract(pos_divergence,neg_divergence))
            weight_gradient_scalar = tf.summary.scalar('weight_increment',tf.reduce_mean(delta_w))
        with tf.name_scope('Visible_bias_gradient'):
            delta_vb = tf.multiply(learning_rate/self.size, tf.reduce_sum(tf.subtract(self.v,neg_vis_prob), 0, keep_dims=True))
        with tf.name_scope('Hidden_bias_gradient'):
            delta_hb= tf.multiply(learning_rate/self.size, tf.reduce_sum(tf.subtract(pos_hid_prob,neg_hid_prob), 0, keep_dims=True))

        update = [self.W.assign_add(delta_w), self.vb.assign_add(delta_vb), self.hb.assign_add(delta_hb)]
        return (rec_error, update)        
    
    def gibbs(self, steps):
       return gibbs(self.v,self.hb,self.vb,self.W,steps,get_probabilities,sample)
       
    def plot_weight_update(x=None, y=None):
        plt.xlabel("Epochs")
        plt.ylabel("CD difference")
        plt.title('Weight increment change throughout learning')
        plt.plot(x, y, 'r--')
        plt.show()

    def free_energy(v, weights, vbias, hbias):
        '''
        Compute the free energy for  a visible state
        :param v:
        :param weights:
        :param vbias:
        :param hbias:
        :return:
        '''
        vbias_term = tf.matmul(v, tf.transpose(vbias))
        x_b = tf.matmul(v, weights) + hbias
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(x_b)))
        return - hidden_term - vbias_term


def get_probabilities(units_value, weights, bias, is_hidden_layer = False):
    if is_hidden_layer:
        with tf.name_scope("Hidden_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(units_value,weights) + bias)
    else:
        with tf.name_scope("Visible_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(units_value,tf.transpose(weights)) + bias) 

def get_gaussian_probabilities(units_value, weights, bias, is_hidden_layer = False):
    if is_hidden_layer:
        with tf.name_scope("Hidden_Probabilities"):
            return tf.matmul(units_value,weights) + bias
    else:
        with tf.name_scope("Visible_Probabilities"):
            return tf.matmul(units_value,tf.transpose(weights)) + bias

def sample(probabilities):
    '''
    Sample a tensor based on the probabilities
    :param probabilities: A tensor of probabilities given by 'restricted_boltzman_machine.get_probabilities'
    :return: A sampled sampled tensor
    '''
    return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))

def sample_gaussian(probabilities, stddev=1):
    '''
    Sample a tensor based on the probabilities
    :param probabilities: A tensor of probabilities given by 'restricted_boltzman_machine.get_probabilities'
    :return: A sampled sampled tensor
    '''
    return tf.add(probabilities, tf.random_normal(tf.shape(probabilities), mean=0.0, stddev=stddev))

def gibbs(input_data, hidden_bias, visible_bias, weights, steps, get_prob_function, sample_function):
        with tf.name_scope("Gibbs_sampling"):
            for i in range(steps):
                hidden_p = get_prob_function(input_data, weights , hidden_bias, True)
                h = sample_function(hidden_p)

                visible_p = get_prob_function(h, weights, input_data, False)
                v = visible_p
                #v = sample(visible_p)
            return visible_p
 

if __name__=="__main__":
    print("start")




    
         