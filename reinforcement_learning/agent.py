import sys
import os
import numpy as np
import tensorflow as tf
import random
import gym
import cv2
import time


def rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """
    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b    

    return img_gray

def create_conv_model(x_input, configuration, output_size, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)):                    
    conv_Definitions = configuration['conv']
    fc_Definitions = configuration['fc']        
    bottom =  x_input       

    for idx, conv_layer_def in enumerate(conv_Definitions):
        op = "conv"
        layer_name = op+ "_layer" + str(idx)
        out_channels = conv_layer_def['channels']
        filter_size = conv_layer_def['filter_size']
        stride = conv_layer_def['stride']
        bottom = tf.contrib.slim.conv2d(bottom, out_channels, [filter_size, filter_size], stride = stride, weights_initializer=initializer,  scope=layer_name)            
        if 'pooling_size' in conv_layer_def:
            filter_pooling_size = conv_layer_def['pooling_size']
            bottom = tf.contrib.slim.max_pool2d(bottom,[filter_pooling_size,filter_pooling_size],scope = "pool_layer"+str(idx))
    bottom = tf.contrib.slim.flatten(bottom)
    for idx, layer_nodes in enumerate(fc_Definitions):         
        layer_name =  "fc_layer" + str(idx)
        bottom = tf.contrib.slim.fully_connected(bottom, layer_nodes, scope=layer_name, weights_initializer = initializer)
    out = tf.contrib.slim.fully_connected(bottom, output_size, scope="out",activation_fn=None, weights_initializer = initializer)
    return out

#################################################################################################

class Frame_Processor:
    def __init__(self, img_size, frames_count, gray_scale = True):
        self.frames = list()
        self.frames_count = frames_count                
        self.height = img_size[0]
        self.width = img_size[1]        
        self.gray_scale = gray_scale

    def _preprocess(self, image):
        img = cv2.resize(image,(self.width, self.height),interpolation=cv2.INTER_CUBIC)
        if self.gray_scale:
            img = rgb_to_grayscale(img)
        return img

    def reset(self):
        self.frames.clear()

    def get_state_size(self):
        if self.gray_scale:
            return [self.height,self.width,self.frames_count]
        return [self.height,self.width,self.frames_count*3]

    def process_frame(self, image):
        img = self._preprocess(image)
        current_frames = len(self.frames)
        for index in range(self.frames_count-current_frames+1):
            self.frames.append(img)
        if (len(self.frames)> self.frames_count):
            self.frames.pop(0)           

    def get_state(self): #the state of the systems is composed by the last frames_count frames.Usually frames_count is 4
        if len(self.frames)==0:
            raise Exception("Empty State")
        state = np.dstack(self.frames)
        return state

################################################################################################


#################################################################################################

class Linear_Decay():
    def __init__(self, start_value, end_value, num_iterations):
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.step = (self.end_value - self.start_value)/num_iterations

    def get_value(self, iteration):
        if iteration>=self.num_iterations:
            return self.end_value        
        return self.start_value + (self.step* iteration)

#################################################################################################

class Epsilon_Greedy_Policy:
    def __init__(self, start_epsilon, end_epsilon, num_iterations):
        self.epsilon_provider = Linear_Decay(start_epsilon,end_epsilon, num_iterations)
        
    def get_action(self, actions_reward, iteration, train):
        #choose a random action with epsilon probability otherwise choose the best action based on reward
        sample = np.random.random()
        if train == True:
            epsilon = self.epsilon_provider.get_value(iteration)
        else:
            epsilon = -1
        if sample<epsilon:
            action = random.randint(0,len(actions_reward)-1)
        else:
            action = np.argmax(actions_reward)
        return action
                

#################################################################################################
class Neural_Network():
    def __init__(self, config, num_actions, input_size, experiment_dir, learning_rate_minimum, learning_rate,learning_rate_decay,learning_rate_decay_step):
        """
        config - represents  layers configurations
        session - tensorflow session
        num_actions - the number of actions for which the reward is predicted
        input_size - input size tensor in format [height, width, channels]
        """    
        self.config = config                    
        self.num_actions = num_actions                
        self.input_size = input_size
        self.learning_rate_minimum = learning_rate_minimum
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_step = learning_rate_decay_step
        self.experiment_dir = experiment_dir        
        self._build_network()
       

                
    def _initPlaceholders(self):        
        self.states = tf.placeholder(name="states",shape=[None]+self.input_size,dtype='float') # represents differents from the game
        self.q_values_target = tf.placeholder(name="old_qvalues",shape=[None, self.num_actions],dtype='float') # the Qvalues predicted by NN at time T
        self.actions = tf.placeholder(name="actions",shape=[None,self.num_actions],dtype='float') # one hot encoding action applied
        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')

    def close(self):        
        self.sess.close()
    
    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """

        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.experiment_dir)

            # Try and load the data in the checkpoint.
            self.saver.restore(self.sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", self.experiment_dir)
            print("Initializing variables instead.")
            self.sess.run(tf.global_variables_initializer())

    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.sess, save_path=self.experiment_dir, global_step=current_iteration)
        print("Saved checkpoint.")

    def _build_network(self):
        self._initPlaceholders()

        self.count_states = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name='count_states') # this variables stores the number of states

        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name='count_episodes') # this variables stores the number of states

        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states, self.count_states + 1)

        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes, self.count_episodes + 1)

        self.q_values = create_conv_model(self.states, self.config, self.num_actions)
        error = tf.losses.mean_squared_error(self.q_values_target * self.actions,self.q_values * self.actions)
        self.loss = error

        self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))

        self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.load_checkpoint()

    def predict(self, states):
        """
        predicts the q values associates with the states for each action
        """
        values = self.sess.run(self.q_values,feed_dict={self.states:states})
        return values

    def train(self, states, target_values, actions, iteration):
        loss_val,_ = self.sess.run([self.loss,self.optim], feed_dict={self.states: states, self.q_values_target:target_values,self.actions:actions,self.learning_rate_step:iteration})            
        if iteration % 100 ==0:
            print("Iteration",iteration,"loss", loss_val)

    def get_count_states(self):
        """
        Returns the number of states executed. An episodes contains multiples states
        """
        return self.sess.run(self.count_states)

    def get_count_episodes(self):
        """
        Returns the number of episodes executed.
        """
        return self.sess.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.sess.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.sess.run(self.count_episodes_increase)


#################################################################################################


class Replay_Memory():
    def __init__(self, max_number_of_items, state_size, num_actions):                        
        self.max_number_of_items = max_number_of_items
        self.state_size = state_size
        self.num_actions = num_actions        
        self._datastore=list()
        

    def add(self, current_state, current_action, current_reward, end_of_life, end_of_episode):
        # in order to replay the actions done in the past we have to save it to our memory
        if self.is_full():
            self._datastore.pop(0)           
        self._datastore.append((current_state,current_action,current_reward,end_of_life,end_of_episode))

    
    def is_full(self):
        return len(self._datastore)==self.max_number_of_items        

    def sample_from_experience(self, batch_size):
        indexes = np.arange(len(self._datastore)-1)
        batch_indexes = np.random.choice(indexes,batch_size,replace=False)

        states = np.zeros([batch_size]+self.state_size,dtype=np.uint8)
        next_states = np.zeros([batch_size]+self.state_size,dtype=np.uint8)
        actions = np.zeros([batch_size, self.num_actions],dtype=np.uint8)
        rewards = np.zeros([batch_size],dtype=np.float)                
        end_of_life = np.zeros([batch_size],dtype=np.bool)                
        end_of_scenario = np.zeros([batch_size],dtype=np.bool)                        


        for index in range(batch_size):            
            current_item = batch_indexes[index]
            state, action, reward, is_end_of_life,is_end_of_episode = self._datastore[current_item]
            next_state, _, _, _,_ = self._datastore[current_item + 1]
            next_states[index] = next_state
            states[index]= state
            actions[index,action]=1
            rewards[index] = reward
            end_of_life[index] = is_end_of_life
            end_of_scenario[index] = is_end_of_episode
                                            
        
        return states, actions, rewards, end_of_life, end_of_scenario, next_states
    
        

#################################################################################################

class process_configuration:
    def __init__(self, config_dictionary):
        self.environment_name = 'Breakout-v0'
        self.policy_epsilon_start = 1.0
        self.policy_epsilon_end = 0.05
        self.policy_decay_iterations = 1e6                        
        self.img_size = [105, 80]
        self.frames_count = 4
        self.gray_scale = True
        self.discount_factor = 0.97
        self.alpha = 1
        self.replay_memory_size = 10000
        self.batch_size=128
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * 10000
        self.reward_interval = (-1,1)
        self.net_config = {'conv' :[{'filter_size':3,'channels':16,'pooling_size':2,'stride':2},
                                    {'filter_size':3,'channels':32,'pooling_size':2,'stride':2},
                                    {'filter_size':3,'channels':64,'pooling_size':2,'stride':1}],                                       
                           'fc':[1024,1024,1024,1024]
                           }
        self.experiment_dir = "C:\\Work\\RL_Experiments\\RL_experiment"
       

class Agent:
    def __init__(self, config):        
        if config is None:
            config = process_configuration()
                    
        self.config = config
        self.environment = gym.make(config.environment_name)        
        self.num_actions = self.environment.action_space.n                 
        self.frame_processor = Frame_Processor(config.img_size,config.frames_count,config.gray_scale)
        self.model = Neural_Network(config.net_config, self.num_actions, self.frame_processor.get_state_size(),self.config.experiment_dir,self.config.learning_rate_minimum,self.config.learning_rate,self.config.learning_rate_decay, self.config.learning_rate_decay_step)
        self.policy = Epsilon_Greedy_Policy(config.policy_epsilon_start,config.policy_epsilon_end, config.policy_decay_iterations)        
        self.action_names = self.environment.unwrapped.get_action_meanings()
        self.replay_memmory = Replay_Memory(config.replay_memory_size,self.frame_processor.get_state_size(),self.num_actions)        

        

    def get_lives(self):
        """Get the number of lives the agent has in the game-environment."""
        return self.environment.unwrapped.ale.lives()

    def run(self, train=True, render = False, number_of_episodes = float('inf')):
        count_episodes = self.model.get_count_episodes()
        count_states = self.model.get_count_states()
        end_episode = True

        while count_episodes < number_of_episodes:
            if end_episode:
                frame = self.environment.reset() # get first frame
                self.frame_processor.reset()
                self.frame_processor.process_frame(frame)
                reward_episode = 0.0
                count_episodes = self.model.increase_count_episodes()
                num_lives = self.get_lives()

            current_state = self.frame_processor.get_state()

            ## predict the q values for the current state
            q_values = self.model.predict([current_state])[0] 

            ##act using the epsilon greedy policy with respect with the predicted q_values
            action = self.policy.get_action(q_values, iteration=count_states, train= train)
            frame,reward,end_episode,info = self.environment.step([action])
            self.frame_processor.process_frame(frame)

            num_lives_new = self.get_lives()
            end_life = (num_lives_new < num_lives)
            num_lives = num_lives_new
            

            if train: 
                reward = np.clip(reward,self.config.reward_interval[0],self.config.reward_interval[1])
                self.replay_memmory.add(current_state,action,reward,end_life,end_episode)
                if self.replay_memmory.is_full():
                    # Increase the counter for the number of states that have been processed.
                    count_states = self.model.increase_count_states()
                    states_batch, actions_one_hot_batch, rewards_batch, end_of_life_batch, end_of_scenario_batch, next_states_batch = self.replay_memmory.sample_from_experience(self.config.batch_size)
                    actions_batch = np.argmax(actions_one_hot_batch,axis=1) 

                    next_states_q_values = self.model.predict(next_states_batch)
                    current_q_values = self.model.predict(states_batch)

                    target_q_values = np.zeros([self.config.batch_size,self.num_actions])
                    current_q_action_values = current_q_values[np.arange(self.config.batch_size),actions_batch]

                    target_q_action_values = current_q_action_values + self.config.alpha*(rewards_batch + self.config.discount_factor*np.max(next_states_q_values,axis=1)-current_q_action_values)
                    target_q_action_values = ((1-end_of_life_batch)*target_q_action_values) + (end_of_life_batch * rewards_batch)
                    target_q_values[np.arange(self.config.batch_size),actions_batch] = target_q_action_values
                    self.model.train(states_batch, target_q_values, actions_one_hot_batch, count_states)
                    if count_states % 10000 == 0:
                        self.model.save_checkpoint(count_states)
            if render:
                 self.environment.render()
                 if train == False:
                     time.sleep(0.01)

if __name__=="__main__":
    print("start")   
    config = process_configuration(None)     
    agent = Agent(config)
    agent.run(True,render=True)             
                                    






    
               



