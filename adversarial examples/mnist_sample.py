import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class conv_model:
    def __init__(self, configuration):
        self.input_size = configuration['input_size']
        self.output_size = configuration['output_size']
        self.init_placeholders()        
        self.create_model(configuration)

    def init_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size],name="input_data")
        self.y = tf.placeholder(tf.float32, shape=[None, self.output_size])

    def create_model(self, configuration):
                    
        conv_Definitions = configuration['conv']
        fc_Definitions = configuration['fc']
        size = int(np.sqrt(self.input_size))        
        bottom = tf.reshape(self.x, [-1, size, size, 1])        

        for idx, conv_layer_def in enumerate(conv_Definitions):
            op = "conv"
            layer_name = op+ "_layer" + str(idx)
            out_channels = conv_layer_def['channels']
            filter_size = conv_layer_def['filter_size']
            bottom = tf.contrib.slim.conv2d(bottom, out_channels, [filter_size, filter_size],  scope=layer_name)            
            if 'pooling_size' in conv_layer_def:
                filter_pooling_size = conv_layer_def['pooling_size']
                bottom = tf.contrib.slim.max_pool2d(bottom,[filter_pooling_size,filter_pooling_size],scope = "pool_layer"+str(idx))

        bottom = tf.contrib.slim.flatten(bottom)
        for idx, layer_nodes in enumerate(fc_Definitions):         
            layer_name =  "fc_layer" + str(idx)
            bottom = tf.contrib.slim.fully_connected(bottom, layer_nodes, scope=layer_name)
        self.prediction = tf.contrib.slim.fully_connected(bottom, self.output_size, scope="out",activation_fn=None)
        return self.prediction


def train_mnist(model):
    #softmax = tf.nn.softmax_cross_entropy_with_logits( logits = self.prediction, labels = self.y)  
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=model.y, logits=model.prediction)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(model.prediction, 1), tf.argmax(model.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    keep_prob = tf.placeholder(tf.float32)
        

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    model.x: batch[0], model.y: batch[1]})
                ddd = grad_x.eval(feed_dict={model.x: batch[0]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={model.x: batch[0], model.y: batch[1]})            

        print('test accuracy %g' % accuracy.eval(feed_dict={
            model.x: mnist.test.images, model.y: mnist.test.labels}))



if __name__=="__main__":
    print("start")
    #ddd()
    configuration = {      'input_size':784,
                           'output_size':10,
                           'conv' :    
                                   [{'filter_size':5,'channels':32,'pooling_size':2},
                                    {'filter_size':5,'channels':64,'pooling_size':2},
                                    ],
                           'fc':[1024]
                         } 

    model = conv_model(configuration)
    train_mnist(model)
    tf.gradients(model.x,model.prediction)

            
                

