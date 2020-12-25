import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, losses, optimizers

class DQNModel(Model):
    def __init__(self, obs_size, action_size):
        super(DQNModel, self).__init__()
        
        self.conv1 = Conv2D(32, kernel_size=(8,8), strides=4, activation='relu')
        self.conv2 = Conv2D(64, kernel_size=(4,4), strides=2, activation='relu')
        self.conv3 = Conv2D(64, kernel_size=(3,3), strides=4, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class DQNAgent():
    def __init__(self, env, savedir="dqn/"):
        self.env = env
        self.state = env.reset()
        self.obs_size = env.observation_space.shape[0]
        self.act_size = env.action_space.n
        
        self.model = DQNModel(self.obs_size, self.act_size)
        self.target_model = DQNModel(self.obs_size, self.act_size)
        self.lr = 0.000002
        self.model.compile(loss='mse', 
                           optimizer=optimizers.Adam(lr=self.lr))
        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02
        
        self.episode_time = 0
        self.total_time = 30000
        self.episode_reward = 0.0
        self.gamma = 0.9
        self.mean_rewards = []
              
        self.buffer_size = 10000
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 128
        self.replay_start_time = 10000
        
    def get_max_q_vals(self, states, target=False):
        if not target:
            return np.max(self.model.predict(states), axis=-1)
        else:
            return np.max(self.target_model.predict(states), axis=-1)
        
        
    def get_action(self, test=True):
        """
        return the optimal action in the given state.
        if "mode" is "train", use the 
        """
        action = np.argmax(self.model.predict(np.expand_dims(self.state, axis=0)), 
                           axis=-1)[0]
      
        if not test:
            if np.random.random() < max(self.epsilon, self.epsilon_min):
                action = self.env.action_space.sample()
                self.epsilon *= self.epsilon_decay 
        return action
        
    def reset(self):
        self.env.reset()
        self.episode_rewards = 0.0
    
    def step(self, action):
        """
        step in environment for data collection
        """
        state = self.state
        next_state, reward, is_done, _ = self.env.step(action)
        self.episode_reward += reward 
        self.buffer.append([state, action, reward, next_state])
        
        self.state = next_state
   
        if is_done:
            mean_reward = self.episode_reward / self.episode_time
            self.mean_rewards.append(mean_reward)
            print("mean reward of {} with {} steps".format(mean_reward, 
                                                           self.episode_time))
            self.episode_reward = 0.0
            self.episode_time = 0
            self.env.reset()
           
                               
    def sample(self, batch_size):
        """
        sample a minibatch from a buffer
        """
        return random.sample(self.buffer, self.batch_size)
            
    def train(self):
        """
        train the DQN model (network) while periodically updating the target model
        """
        
        for time in range(self.total_time):
            self.episode_time += 1
            
            # step in environment to collect the data
            action = self.get_action(test=False)
            self.step(action)
            
            
            # if ready, train the DQN with the sampled minibatch
            if time >= self.replay_start_time:
                
                # sample
                batch = self.sample(self.batch_size)
                
                states = np.array([sars[0] for sars in batch])
                actions = np.array([sars[1] for sars in batch])
                rewards = np.array([sars[2] for sars in batch])
                next_states = np.array([sars[3] for sars in batch])
                
                target_max_q_vals = self.get_max_q_vals(next_states, target=True)
                ys = rewards + self.gamma * target_max_q_vals
                
                # q_vals = self.get_q_vals(states)
                
                # train
                
                self.model.fit(states, ys, 
                               batch_size=self.batch_size,
                               epochs=1, 
                               verbose=1)

                # loss = losses.MSE(q_vals, ys)

                # global_step = tf.Variable(0, trainable=False, name='global_step')
                # training_op = self.optimizer.minimize(loss,
                #                                       global_step=global_step)

                self.target_model.set_weights(self.model.get_weights())
                
                
                
                                
                