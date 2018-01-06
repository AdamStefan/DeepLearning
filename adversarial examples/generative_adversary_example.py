import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
from math import ceil
import matplotlib.pyplot as plt



def compute_conv_output_size(input_height, input_width, filter_size, stride, padding = 'SAME'):
    if padding == 'SAME':
        out_height = ceil(float(input_height) / float(stride))
        out_width  = ceil(float(input_width) / float(stride))
    else:
        out_height = ceil(float(input_height - filter_size + 1) / float(stride))
        out_width  = ceil(float(input_width - filter_size + 1) / float(stride))
    return out_height, out_width


def conv2d(x,W, conv_bias ,stride = 1, padd_val = 'SAME', name:str = None):
    ret = tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding = padd_val)
    ret = tf.add(ret,conv_bias, name)
    return ret


def conv2d_relu(x,W, conv_bias ,stride = 1, padd_val = 'SAME', name:str = None):
    ret = conv2d(x,W, conv_bias ,stride, padd_val)
    ret = tf.nn.relu(ret, name)
    return ret


def conv2d_batchnorm_relu(x, channels , kernel_size, stride = 1, padd_val = 'SAME', is_training = False, name = None):
    with tf.variable_scope(name):
        in_channels = x.get_shape()[3].value
        

        W = tf.get_variable('w',[kernel_size, kernel_size, in_channels, channels],initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b',[channels],initializer=tf.constant_initializer(0.0))

        ret = conv2d(x,W, b ,stride, padd_val)
        ret = tf.contrib.layers.batch_norm(ret, decay=0.9, is_training = is_training, scale = True, updates_collections=None, scope="bn")
        ret = tf.nn.relu(ret)
    return ret

def fc_net_batchnorm_relu(x, size, is_training = False, batchNorm = True, name = None):
    with tf.variable_scope(name):
        fc_inputSize = x.get_shape()[1].value
        fc_weights = tf.get_variable('w',[fc_inputSize, size],initializer=tf.truncated_normal_initializer(stddev=0.02))
        fc_biases = tf.get_variable('b',[size],initializer=tf.constant_initializer(0.0))
        ret = tf.add(tf.matmul(x, fc_weights), fc_biases)
        if batchNorm:
            ret = tf.contrib.layers.batch_norm(ret, decay=0.9, is_training = is_training, scale = True, updates_collections=None, scope="bn")
        ret = tf.nn.relu(ret)    
    return ret

def fc_net(x, size,  name = None):
    with tf.variable_scope(name):
        fc_inputSize = x.get_shape()[1].value
        fc_weights = tf.get_variable('w',[fc_inputSize, size],initializer=tf.truncated_normal_initializer(stddev=0.02))
        fc_biases = tf.get_variable('b',[size],initializer=tf.constant_initializer(0.0))
        ret = tf.add(tf.matmul(x, fc_weights), fc_biases)        
    return ret



def create_conv_model(configuration, input, output_size,name="conv", is_training= False, reuse_variables= False):
    with tf.variable_scope(name) as scope:
        if reuse_variables:
            scope.reuse_variables()

        conv_Definitions = configuration['conv']  
        fc_Definitions = configuration['fc']  
        bottom = input                          
        for idx, conv_layer_def in enumerate(conv_Definitions):
            op = "conv"
            layer_name = name +  op + "_layer" + str(idx)
            out_channels = conv_layer_def['channels']
            filter_size = conv_layer_def['filter_size']
            bottom = conv2d_batchnorm_relu(bottom, out_channels, filter_size, name=layer_name, is_training=is_training)            
            if 'pooling_size' in conv_layer_def:
                filter_pooling_size = conv_layer_def['pooling_size']
                bottom = tf.contrib.slim.max_pool2d(bottom,[filter_pooling_size,filter_pooling_size],scope = name + "pool_layer"+str(idx))

        bottom = tf.contrib.slim.flatten(bottom)   
        for idx, layer_nodes in enumerate(fc_Definitions):         
            layer_name =  name + "_" +  "fc_layer" + str(idx)
            bottom = fc_net_batchnorm_relu(bottom, layer_nodes, name=layer_name, is_training = is_training)     

        prediction = fc_net(bottom, output_size, name="out")        
    return prediction

def create_deconv_model(configuration,input, output_size, batch_size,name="deconv",is_training = False):   
    with tf.variable_scope(name) as scope:
        conv_output_sizes = [] 
        conv_input_size = output_size;
        conv_Definitions = configuration['conv']
        fc_Definitions = configuration['fc']      
        conv_output_sizes.append(output_size)

        for idx, conv_layer_def in enumerate(conv_Definitions):
            filter_size = conv_layer_def['filter_size']
            channels = conv_layer_def['channels']
            stride = 2
            if stride in conv_layer_def:
                stride = conv_layer_def['stride']

            conv_input_size = compute_conv_output_size(conv_input_size[0],conv_input_size[1],filter_size,stride)
            conv_output_sizes.insert(0,(conv_input_size[0],conv_input_size[1],channels, stride))
    
        first_de_conv_size =  conv_output_sizes[0]    
        top_fully_connected_size = first_de_conv_size[0] *  first_de_conv_size[1] * first_de_conv_size[2]

        bottom = input

        for idx, layer_nodes in enumerate(reversed(fc_Definitions)):         
            layer_name = name + "_" +   "fc_layer" + str(idx)
            bottom = fc_net_batchnorm_relu(bottom,layer_nodes,is_training=is_training, name=layer_name)        

        #add last_fully_connected_layer
        layer_name = name + "_" +  "fc_layer" + str(len(fc_Definitions))
        bottom = fc_net_batchnorm_relu(bottom,top_fully_connected_size,is_training=is_training, name=layer_name)    
        bottom = tf.reshape(bottom,[-1,first_de_conv_size[0],first_de_conv_size[1],first_de_conv_size[2]])

        for i in range(len(conv_output_sizes)-1):        
            input_size = conv_output_sizes[i]
            output_size = conv_output_sizes[i+1]   
            op = "de_conv"
            out_channels = output_size[2]
            in_channels = input_size[2]
            filter_size = conv_Definitions[len(conv_Definitions)-1-i]['filter_size']
            layer_name = name+"_" + op +"_" + str(i)

            with tf.variable_scope(layer_name):
                w = tf.get_variable('w',[filter_size,filter_size,out_channels,in_channels],initializer=tf.random_normal_initializer(stddev=0.02))
                bottom = tf.nn.conv2d_transpose(bottom,w,[batch_size,output_size[0],output_size[1],out_channels],strides=[1, input_size[3], input_size[3], 1])
                biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
                bottom = tf.nn.bias_add(bottom, biases)
                if (i<(len(conv_output_sizes)-2)):                 
                    bottom = tf.contrib.layers.batch_norm(bottom, decay=0.9, is_training = is_training, scale = True, updates_collections=None)
                    bottom = tf.nn.relu(bottom)
                else:
                    bottom = tf.tanh(bottom)
        return bottom


class generative_adversary_model:
    def __init__(self, configuration_discriminator, configuration_generator, input_x, input_z, batch_size, is_training):
        self.D = create_conv_model(configuration_discriminator,input_x,1,is_training=is_training, name='discriminator',reuse_variables=False)
        input_x_shape = input_x.get_shape()
        input_x_shape = (input_x_shape[1].value,input_x_shape[2].value,input_x_shape[3].value)
        self.G = create_deconv_model(configuration_generator,input_z,input_x_shape, batch_size,is_training=is_training, name='generator')
        self.Dg = create_conv_model(configuration_discriminator, self.G,1,is_training=is_training, name='discriminator',reuse_variables=True)


if __name__=="__main__":
    print("start")
    mnist = input_data.read_data_sets("MNIST_data/")
    configuration_disc = {      
                        'input_size':784,
                        'output_size':10,
                        'conv' :    
                                [{'filter_size':5,'channels':32,'pooling_size':2},
                                {'filter_size':5,'channels':64,'pooling_size':2},
                                ],
                        'fc':[1024,1024]
                    }   


    configuration_gen = {      
                        'input_size':784,
                        'output_size':10,
                        'conv' :    
                                [{'filter_size':5,'channels':32,'pooling_size':2},
                                {'filter_size':5,'channels':64,'pooling_size':2},
                                ],
                        'fc':[]
                    }

    batch_size = 10
    z_dimensions = 100
    x = tf.placeholder(tf.float32, shape=[None, 28,28, 1],name="input_data")
    z = tf.placeholder(tf.float32, shape=[None, z_dimensions],name="input_data")  
    is_training = tf.placeholder_with_default(False, shape=[],name="phase")   
    
    
    #create the session 
    sess  = tf.Session()        

    adversary_network = generative_adversary_model(configuration_disc,configuration_gen, x , z, batch_size, is_training) 

    #g(z)
    Gz = adversary_network.G

    #d(x)
    Dx = adversary_network.D

    #Dgz
    Dgz = adversary_network.Dg

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dgz, labels=tf.ones_like(Dgz))) #discriminator's weights are locked and generator's weights are trained

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dgz, labels=tf.zeros_like(Dgz))) #discriminator's weights are trained and generator's weights are locked
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels = tf.ones_like(Dx)))
    d_loss = d_loss_real + d_loss_fake

    total_var = tf.trainable_variables()

    d_vars = [var for var in total_var if 'discriminator/' in var.name]
    g_vars = [var for var in total_var if 'generator/' in var.name]

    print(tf.get_variable_scope().reuse)
    adam = tf.train.AdamOptimizer()
    trainerD = adam.minimize(d_loss, var_list=d_vars)
    trainerG = adam.minimize(g_loss, var_list=g_vars)
       

    sess.run(tf.global_variables_initializer())
    iterations = 50000
    for i in range(iterations):
        z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])        
        real_image_batch = mnist.train.next_batch(batch_size)
        real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
        _,dLoss = sess.run([trainerD, d_loss],feed_dict={x:real_image_batch,is_training:True,z:z_batch}) #Update the discriminator
        _,gLoss = sess.run([trainerG,g_loss],feed_dict={is_training:True,z:z_batch}) #Update the generator
        #if i%100 ==0:
        #    z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])        
        #    temp = (sess.run(Gz, feed_dict={is_training:False,z:z_batch}))
        #    my_i = temp[0].squeeze()
        #    plt.imshow(my_i, cmap='gray_r')
        #    plt.show()

    
    z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])        
    temp = (sess.run(Gz, feed_dict={is_training:False,z:z_batch}))
    my_i = temp[0].squeeze()
    plt.imshow(my_i, cmap='gray_r')
    plt.show()
   

    print("End")
      

