#-*- coding:utf-8
import numpy as np
import tensorflow as tf
import random
import codecs
import os
import math

from GameEnv import Game2048
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

class DQN:
    def __init__(self,learning_rate=0.01,sigma = 1,grid_n = 4,batch_size=1000,file_data="train_data.csv",h1=128,net_type="full_cnn_valid",
                input_type="ont-hot",
                dqn_type="double_dqn",
                loss_type="huber",
                replace_update_fre=500,
                loss_show_step=100,
                greedy_min=0.000001,
                 data_rotate_expand = True,
                memory_max_size=10000,
                learning_rate_decay_rate=0.99): #初始化
        self.global_steps = tf.Variable(0, trainable=False)
#         self.learning_rate = learning_rate
        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=self.global_steps,
                                           decay_steps=1000,
                                           decay_rate=learning_rate_decay_rate,
                                           staircase=True)

        self.net_type = net_type
        self.input_type = input_type # raw/one-hot/emb
        self.dqn_type = dqn_type  # dqn_2015/double_dqn/dueling_dqn
        self.loss_type = loss_type # mse/abs/huber
        self.input_embedding_size = 64
        self.replace_update_fre = 100
        self.data_rotate_expand = data_rotate_expand #if rotate to expand the data
        self.data_statistic = False #get the q [r1,r2] [q1',q2']
        self.loss_no_bigger = False # a loss to avoid next bigger than this
        self.loss_show_step = loss_show_step
        self.greedy_min = greedy_min
        self.memory_max_size = memory_max_size
        self.greedy_per_step = 100
        self.opt_type = "rmsp"
        
        
        self.total_iters = 0
        self.episode = 0
        self.batch_size = batch_size
        self.sigma = sigma
        self.step = 0
        self.e = 0.9 
        self.explore_speed = 0.8
        self.h1 = h1

        self.grid_n = grid_n
        self.actions_index_dicts = {"a":0,"s":1,"w":2,"d":3}
        self.actions_index_dicts_reverse = {0:"a",1:"s",2:"w",3:"d"}
        self.actions_index_keys = ["a","s","w","d"]

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.update_target_q_network()
        self.memory = []
        self.memory_key_dict = {}
        self.memory_status_dict = {}
        self.file = codecs.open(file_data,"w",encoding='utf-8')
        self.transform_dict = {}

    
    def build_input(self):
        if self.input_type == "raw":
            self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.float32,name="matrixInput")
        elif self.input_type == "one-hot":
            self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n,self.grid_n ** 2],dtype=tf.float32,name="matrixInput")
        elif self.input_type == "emb":
            self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.int32,name="matrixInput")
            
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")
        self.yAllInput = tf.placeholder(shape=[None,4],dtype=tf.float32,name="yAllInput")
        self.qEvalInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="qEvalInput")
    
    def build_net(self):
        if self.input_type == "raw":
            input_x = tf.expand_dims(self.matrixInput, -1)
        elif self.input_type == "one-hot":
            input_x = self.matrixInput
        elif self.input_type == "emb":
            emb_weight = tf.Variable(tf.random_uniform([self.grid_n ** 2 + 1, self.input_embedding_size], -1.0, 1.0),name="num_emb")
            input_x   = tf.nn.embedding_lookup(emb_weight, self.matrixInput) # bs * grid_n * grid_n * emb
        
        if self.net_type == "full_cnn_valid":
            for channel_size in [128,128]:
                input_x = tf.layers.conv2d(input_x,channel_size,(2,2),padding="valid",activation=tf.nn.leaky_relu)

            conv_res = tf.reshape(input_x,shape=[-1,self.grid_n ** 2 * 16 * 2])
        elif self.net_type == "2conv":
            channel_size = 128
            conv1 = tf.layers.conv2d(input_x,channel_size,(1,2),padding="valid",activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(input_x,channel_size,(2,1),padding="valid",activation=tf.nn.leaky_relu)
            conv3 = tf.reshape(tf.layers.conv2d(conv1,channel_size,(1,2),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,8*channel_size])
            conv4 = tf.reshape(tf.layers.conv2d(conv1,channel_size,(2,1),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,9*channel_size])
            conv5 = tf.reshape(tf.layers.conv2d(conv2,channel_size,(1,2),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,9*channel_size])
            conv6 = tf.reshape(tf.layers.conv2d(conv2,channel_size,(2,1),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,8*channel_size])
            conv1_r = tf.reshape(tf.layers.conv2d(input_x,channel_size,(1,2),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,12*channel_size])
            conv2_r = tf.reshape(tf.layers.conv2d(input_x,channel_size,(2,1),padding="valid",activation=tf.nn.leaky_relu),shape=[-1,12*channel_size])
            conv_res = tf.concat([conv1_r,conv2_r,conv3,conv4,conv5,conv6],-1)
            
        elif self.net_type == "cnn_max_pooling":
            for channel_size in [128,128]:
                conv_i = tf.layers.conv2d(input_x,channel_size,(2,2),padding="valid",activation=tf.nn.leaky_relu)
                input_x = tf.layers.max_pooling2d(conv_i,(2,2),(1,1),padding="same")
            conv_res = tf.reshape(input_x,shape=[-1,self.grid_n ** 2 * 16])
        elif self.net_type == "full_cnn_same":
            for channel_size in [128,128,16]:
                input_x = tf.layers.conv2d(input_x,channel_size,(2,2),padding="same",activation=tf.nn.leaky_relu)

            conv_res = tf.reshape(input_x,shape=[-1,self.grid_n ** 2 * 16])
        layer1 = tf.layers.dense(conv_res, self.h1, activation=tf.nn.leaky_relu)
        predictions = tf.layers.dense(layer1,4)
        return predictions
    
    def build_model(self):
        self.build_input()
        if self.dqn_type == "dqn_2015" or self.dqn_type == "double_dqn" or self.dqn_type == "dqn_2013":
            with tf.variable_scope('current_net'):
                self.current_predictions = self.build_net()
            with tf.variable_scope('target_net'):
                self.target_predictions = self.build_net()
                  
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            
        self.predictionsMaxQValue = tf.reduce_max(self.current_predictions)
        self.predictionsMaxQAction = tf.arg_max(self.current_predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.current_predictions, self.actionInput), reduction_indices=1)
        self.action_target_predictions =  tf.reduce_sum(tf.multiply(self.target_predictions, self.actionInput), reduction_indices=1)
        
        # Calculate the loss
        if self.loss_type == "mse":
            self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        elif self.loss_type == "abs":
            self.losses = tf.abs(self.yInput - self.action_predictions)
        elif self.loss_type == "huber":
            self.losses = tf.losses.huber_loss(self.yInput,self.action_predictions)
        elif self.loss_type == "entroy_loss":
            #loss
            self.losses = tf.square(tf.subtract(self.yAllInput,self.current_predictions))
            self.losses = tf.reduce_sum(self.losses,axis=1,keep_dims=True)

        self.q_eval_loss = tf.reduce_mean(tf.square(tf.maximum(self.qEvalInput - self.action_predictions,0)))
        if self.loss_no_bigger:
            self.loss = tf.reduce_mean(self.losses) + self.q_eval_loss
        else:
            self.loss = tf.reduce_mean(self.losses) 
        # Optimizer Parameters from original paper
        if self.opt_type == "rmsp":
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_steps)
        elif self.opt_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)
        
       
    def update_target_q_network(self):
        # update target Q netowrk
        if self.episode % self.replace_update_fre == 0:
            self.sess.run(self.target_replace_op)

    
    def change_values(self,X):
        power_mat = np.zeros(shape=(4,4,16),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if(X[i][j]==0):
                    power_mat[i][j][0] = 1.0
                else:
                    power = int(math.log(X[i][j],2))
                    power_mat[i][j][power] = 1.0
        return power_mat.tolist()     

#     def one_hot(self,x):
#         res = [-1] * 17
#         if x == 0:
#             res = [0] * 17
#         else:
#             res[int(np.log2(x))] = 1
#         return res

    def transform_input(self,s):
        if str(s) in self.transform_dict:
            return self.transform_dict[str(s)]
        
        if self.input_type == "emb":
            res = np.floor(np.log2(np.array(s)+1)).tolist()
        elif self.input_type == "raw":
            res = np.log2(np.array(s)+1) / 10.0
        elif self.input_type == "one-hot":
            res = self.change_values(np.array(s))
        self.transform_dict[str(s)] = res
        return res
       

    def transform_reward(self,rewards):
        return rewards
#         return np.log2(np.array(rewards)+1) / 10.0

        
    def _greedy_e(self,seq,probabilities):
        if((self.episode>10000) or (self.episode>0.1 and self.total_iters%2500==0)):
            self.e = self.e/1.005
#         if self.episode > 10000:
#             self.e = self.e / 1.005
#         else:
#             self.e = max(self.explore_speed ** (self.step / self.greedy_per_step),self.greedy_min)
        res = seq[np.argmax(probabilities)]
        if random.random() < self.e:
            res = random.choice(seq)
        return res

#     def choose_action(self,status): #通过训练好的网络，根据状态获取动作
#         prob_all = self.sess.run(self.predictions, feed_dict={self.matrixInput:np.array(status)})[0]
#         return self._greedy_e(self.actions_index_keys,prob_all)

#     def choose_action_max(self,status):
#         max_action = self.sess.run(self.predictionsMaxQAction, feed_dict={self.matrixInput: np.array(status)})[0]
#         return self.actions_index_dicts_reverse[max_action]

    def get_actions(self,status,mode="train",key="action"):
        if mode == "train": #e-greedy
            prob_all = self.sess.run(self.current_predictions, feed_dict={self.matrixInput:np.array([self.transform_input(i) for i in status])})
            return [self._greedy_e(self.actions_index_keys,prob) for prob in prob_all]

    
    def get_action(self,status,mode="train",key="action"):
        if mode == "train": #e-greedy
            prob_all = self.sess.run(self.current_predictions, feed_dict={self.matrixInput:np.array([self.transform_input(status[0])])})[0]
            return self._greedy_e(self.actions_index_keys,prob_all)
        else: # get_max
            prob_all = self.sess.run(self.current_predictions, feed_dict={self.matrixInput:np.array([self.transform_input(status[0])])})[0]
        res_dict = {i:k for i,k in zip( ["a","s","w","d"],prob_all)}
        sort_res = sorted(res_dict.items(),key = lambda x:x[1],reverse = True) 
        if key == "action":
            return sort_res[0][0]
        return sort_res

    def get_q_eval_next(self,status):
        if self.dqn_type == "dqn_2015":
            prob_all = self.sess.run(self.target_predictions, feed_dict={self.matrixInput:np.array(status)})
            return np.max(prob_all,axis=1)
        elif self.dqn_type == "dqn_2013":
            prob_all = self.sess.run(self.current_predictions, feed_dict={self.matrixInput:np.array(status)})
            return np.max(prob_all,axis=1)
        elif self.dqn_type == "double_dqn":
            max_actions = self.sess.run(self.predictionsMaxQAction, feed_dict={self.matrixInput:np.array(status)})
            max_actions_one_hot = self._one_hot(max_actions.tolist())
            probs = self.sess.run(self.action_target_predictions, feed_dict={self.matrixInput:np.array(status),self.actionInput:max_actions_one_hot})
            return probs
    
        
    def _one_hot(self,x,size=4):
        res = np.zeros((len(x), size))
        res[[i for i in range(len(x))], x] = 1
        return res

    def train(self,train_data=None): #训练
        """
        memeory:[[ob_this,action,reward,done,ob_next],[ob_this...]]
        ob_this:[(seq,card,money),()]
        :return:
        """
        self.update_target_q_network()
        if len(self.memory) > self.memory_max_size:
            if train_data is None:
                train_data = self.experience_replay()
            status = np.array([self.transform_input(i) for i in train_data[:,0]])
            action = self._one_hot([self.actions_index_dicts[i] for i in train_data[:,1]])
            reward = self.transform_reward(train_data[:,2].tolist())
            done = train_data[:,3]
            next_status = np.array([self.transform_input(i) for i in train_data[:,4]])


            qEvalNext = self.get_q_eval_next(next_status)
            y = []
            for i in range(self.batch_size):
                if done[i] == True or status[i].tolist() == next_status[i].tolist():
                    y.append(reward[i])
                else:
    #                 y.append(reward[i] + self.sigma * qEvalNext[i])
                    y.append(reward[i] + (self.sigma - 0.1 ** (self.step / 1000.0)) * qEvalNext[i])
    #                 y.append(reward[i] + (1 - self.sigma ** (self.step / 1000.0)) * qEvalNext[i])
            feed_dict = {self.matrixInput: np.array(status), self.actionInput: np.array(action),self.yInput: np.array(y),self.qEvalInput:np.array(qEvalNext)}
            _, global_step,loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
            self.step = global_step
            if global_step % self.loss_show_step == 0:
                print("loss",global_step,loss)
#             self.memory = []

    def turn_left(self,status,action,next_status):
        action_dict = {"a":"s","s":"d","d":"w","w":"a"}
        action = action_dict[action]
        return np.flipud(np.transpose(status)).tolist(),action,np.flipud(np.transpose(status)).tolist()

    def turn_180(self,status,action,next_status):
        action_dict = {"a":"d","d":"a","s":"w","w":"s"}
        action = action_dict[action]
        return np.flip(status).tolist(),action,np.flip(next_status).tolist()

    def turn_right(self,status,action,next_status):
        action_dict = {"a":"w","w":"d","d":"s","s":"a"}
        return np.fliplr(np.transpose(status)).tolist(),action_dict[action], np.fliplr(np.transpose(next_status)).tolist()

    def experience_replay(self): #记忆回放
        res = []
        for i in range(self.batch_size):
            k = random.randint(0,len(self.memory)-1)
            res.append(self.memory.pop(k))
        return np.array(res)
#         return np.array(random.sample(self.memory, self.batch_size))

    def experience_store_expand(self,status,action,reward,done,next_status):
        if self.data_rotate_expand:
            status_,action_,next_status_ = self.turn_left(status,action,next_status)
            self.experience_store(status_,action_,reward,done,next_status_ )

            status_,action_,next_status_ = self.turn_right(status,action,next_status)
            self.experience_store(status_,action_,reward,done,next_status_ )

            status_,action_,next_status_ = self.turn_180(status,action,next_status)
            self.experience_store(status_,action_,reward,done,next_status_ )
        
        self.experience_store(status,action,reward,done,next_status)
        
    def experience_store(self,status,action,reward,done,next_status):
        self.memory_key_dict[str(status)] = 1
        if str(status) + str(next_status) not in self.memory_key_dict:
            self.memory.append([status,action,reward,done,next_status])
#             self.file.write(str(status) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(
#                next_status) + "\n")
            self.memory_key_dict[str(status) + str(next_status)] = 1