#-*- coding:utf-8
import numpy as np
import tensorflow as tf
import random
import codecs
import os

from GameEnv import Game2048
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

class DQN:
    def __init__(self,learning_rate=0.01,sigma = 1,grid_n = 4,batch_size=1000,file_data="train_data.csv",h1=128,h2=64,h3 = 32): #初始化
        self.global_steps = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
#         self.learning_rate = tf.train.exponential_decay(learning_rate,
#                                            global_step=self.global_steps,
#                                            decay_steps=1000,
#                                            decay_rate=0.8)
        self.batch_size = batch_size
        self.sigma = sigma
        self.step = 0
        self.explore_alpha = 0.9 ** (self.step / 1000)
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

        self.grid_n = grid_n
        self.actions_index_dicts = {"a":0,"s":1,"w":2,"d":3}
        self.actions_index_dicts_reverse = {0:"a",1:"s",2:"w",3:"d"}
        self.actions_index_keys = ["a","s","w","d"]


        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.bulid_embedding_cnn_network_2015()
        self.sess.run(tf.global_variables_initializer())
        self.memory = []
        self.memory_done = []
        self.file = codecs.open(file_data,"w",encoding='utf-8')
        self.transform_dict = {}


    def build_network(self): #构建网络模型，简单先搞个DNN

        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.float32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")

        matrixFlat = tf.reshape(self.matrixInput,shape=[-1,self.grid_n ** 2])
        layer1 = tf.layers.dense(matrixFlat, self.h1, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(layer1, self.h2 , activation=tf.nn.leaky_relu)
        layer3 = tf.layers.dense(layer1, self.h3 , activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(layer3,4)


        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
        self.losses = tf.abs(self.yInput - self.action_predictions) * 10
#         self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)

        
    def bulid_cnn_network(self,kernel_size=512):
        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.float32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")

        # --------------- tf.nn.conv1d  -------------------
        inputs_h=self.matrixInput
        inputs_w=tf.transpose(inputs_h,perm=[0,2,1])
        with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
            y1_h = tf.layers.dense(inputs_h,64,activation=tf.nn.leaky_relu,name='layer1')
            y1_w = tf.layers.dense(inputs_w,64,activation=tf.nn.leaky_relu,name='layer1')
            y2_h = tf.layers.dense(y1_h,32,activation=tf.nn.leaky_relu,name='layer2')
            y2_w = tf.layers.dense(y1_w,32,activation=tf.nn.leaky_relu,name='layer2')
        conv_out = tf.concat([tf.reshape(y2_w,shape=[-1,4*32]),tf.reshape(y2_h,shape=[-1,4*32])],1)

        layer1 = tf.layers.dense(conv_out, self.h1, activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(layer1,4)


        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
        self.losses = tf.abs(self.yInput - self.action_predictions)
#         self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)
        
    def bulid_embedding_cnn_network(self):
        embedding_size = 32
        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.int32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")
#         self.matrixInput = tf.floor(tf.log1p(self.matrixInput * np.e + 1))

        emb_weight = tf.Variable(tf.random_uniform([self.grid_n ** 2 + 1, embedding_size], -1.0, 1.0),name="num_emb")
        self.emb_pic   = tf.nn.embedding_lookup(emb_weight, self.matrixInput) # bs * grid_n * grid_n * emb

        conv1 = tf.layers.conv2d(self.emb_pic,32,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling1 = tf.layers.max_pooling2d(conv1,(2,2),(1,1),padding="same")

        conv2 = tf.layers.conv2d(max_pooling1,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling2 = tf.layers.max_pooling2d(conv2,(2,2),(1,1),padding="same")

        conv3 = tf.layers.conv2d(max_pooling2,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling3 = tf.layers.max_pooling2d(conv3,(2,2),(1,1),padding="same")
        
        conv_res = tf.reshape(max_pooling3,shape=[-1,self.grid_n ** 2 * 16])
        layer1 = tf.layers.dense(conv_res, self.h1, activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(layer1,4)


        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
#         self.losses = tf.abs(self.yInput - self.action_predictions)
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)
    
    def bulid_embedding_cnn_network_2015(self):
        embedding_size = 32
        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.int32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")
#         self.matrixInput = tf.floor(tf.log1p(self.matrixInput * np.e + 1))


        with tf.variable_scope('current_net'):
            emb_weight = tf.Variable(tf.random_uniform([self.grid_n ** 2 + 1, embedding_size], -1.0, 1.0))
            emb_pic   = tf.nn.embedding_lookup(emb_weight, self.matrixInput) # bs * grid_n * grid_n * emb

            conv1 = tf.layers.conv2d(emb_pic,32,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling1 = tf.layers.max_pooling2d(conv1,(2,2),(1,1),padding="same")

            conv2 = tf.layers.conv2d(max_pooling1,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling2 = tf.layers.max_pooling2d(conv2,(2,2),(1,1),padding="same")

            conv3 = tf.layers.conv2d(max_pooling2,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling3 = tf.layers.max_pooling2d(conv3,(2,2),(1,1),padding="same")

            conv_res = tf.reshape(max_pooling3,shape=[-1,self.grid_n ** 2 * 16])
            layer1 = tf.layers.dense(conv_res, self.h1, activation=tf.nn.leaky_relu)
            self.predictions = tf.layers.dense(layer1,4)
        
        with tf.variable_scope('target_net'):
            emb_weight = tf.Variable(tf.random_uniform([self.grid_n ** 2 + 1, embedding_size], -1.0, 1.0))
            emb_pic   = tf.nn.embedding_lookup(emb_weight, self.matrixInput) # bs * grid_n * grid_n * emb

            conv1 = tf.layers.conv2d(emb_pic,32,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling1 = tf.layers.max_pooling2d(conv1,(2,2),(1,1),padding="same")

            conv2 = tf.layers.conv2d(max_pooling1,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling2 = tf.layers.max_pooling2d(conv2,(2,2),(1,1),padding="same")

            conv3 = tf.layers.conv2d(max_pooling2,16,(3,3),padding="same",activation=tf.nn.leaky_relu)
            max_pooling3 = tf.layers.max_pooling2d(conv3,(2,2),(1,1),padding="same")

            conv_res = tf.reshape(max_pooling3,shape=[-1,self.grid_n ** 2 * 16])
            layer1 = tf.layers.dense(conv_res, self.h1, activation=tf.nn.leaky_relu)
            self.target_predictions = tf.layers.dense(layer1,4)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
#         self.losses = tf.abs(self.yInput - self.action_predictions)
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)
        
    def update_target_q_network(self):
        # update target Q netowrk
#         if episode % REPLACE_TARGET_FREQ == 0:
        self.sess.run(self.target_replace_op)
            #print('episode '+str(episode) +', target Q network params replaced!')

        
    def one_hot(self,x):
        res = [-1] * 17
        if x == 0:
            res = [0] * 17
        else:
            res[int(np.log2(x))] = 1
        return res

    def transform(self,s):
        if str(s) in self.transform_dict:
            return  self.transform_dict[str(s)]
        else:
            self.transform_dict[str(s)] = np.floor(np.log2(np.array(s)+1)).tolist()
        return self.transform_dict[str(s)]
    
    
#         if str(s) in self.transform_dict:
#             return self.transform_dict[str(s)]
#         res = np.zeros((4,4,17))
#         for i in range(4):
#             for j in range(4):
#     #             print one_hot(np.array(s)[i,j])
#                 res[i,j] = self.one_hot(np.array(s)[i,j])
#         self.transform_dict[str(s)] = res
#         return res

    def bulid_one_hot_cnn_network(self):
        embedding_size = 32
        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n,self.grid_n ** 2 + 1],dtype=tf.float32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")


        conv1 = tf.layers.conv2d(self.matrixInput,64,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling1 = tf.layers.max_pooling2d(conv1,(2,2),(1,1),padding="same")

        conv2 = tf.layers.conv2d(max_pooling1,32,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling2 = tf.layers.max_pooling2d(conv2,(2,2),(1,1),padding="same")

        conv3 = tf.layers.conv2d(max_pooling2,8,(3,3),padding="same",activation=tf.nn.leaky_relu)
        max_pooling3 = tf.layers.max_pooling2d(conv3,(2,2),(1,1),padding="same")
        
        conv_res = tf.reshape(max_pooling3,shape=[-1,self.grid_n ** 2 * 8])
        layer1 = tf.layers.dense(conv_res, self.h1, activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(layer1,4)


        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
#         self.losses = tf.abs(self.yInput - self.action_predictions)
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)
        
    def _greedy_e(self,seq,probabilities):
        e = max(0.95 ** (self.step / 1000),0.1)
        res = seq[np.argmax(probabilities)]
        if random.random() < e:
            res = random.choice(seq)
        return res

    def choose_action(self,status): #通过训练好的网络，根据状态获取动作
        prob_all = self.sess.run(self.predictions, feed_dict={self.matrixInput:np.array(status)})[0]
        return self._greedy_e(self.actions_index_keys,prob_all)

    def choose_action_max(self,status):
        max_action = self.sess.run(self.predictionsMaxQAction, feed_dict={self.matrixInput: np.array(status)})[0]
        return self.actions_index_dicts_reverse[max_action]


    def get_max_availble_action_value(self,status):
        prob_all = self.sess.run(self.target_predictions, feed_dict={self.matrixInput:np.array(status)})
        return np.max(prob_all,axis=1)

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
        if train_data is None:
            train_data = self.experience_replay()
        status = np.array([self.transform(i) for i in train_data[:,0]])
        action = self._one_hot([self.actions_index_dicts[i] for i in train_data[:,1]])
        reward = train_data[:,2]
        done = train_data[:,3]
        next_status = np.array([self.transform(i) for i in train_data[:,4]])


        # print (reward)
        # print (action)
        maxQNext = self.get_max_availble_action_value(next_status)
        y = []
        for i in range(self.batch_size):
            if done[i] == True:
                y.append(reward[i])
            elif status[i].tolist() == next_status[i].tolist():
                y.append(0)
            else:
#                 y.append(reward[i] + self.sigma * maxQNext[i])
                y.append(reward[i] + (self.sigma - 0.1 ** (self.step / 1000.0)) * maxQNext[i])
#                 y.append(reward[i] + (1 - self.sigma ** (self.step / 1000.0)) * maxQNext[i])
        feed_dict = {self.matrixInput: np.array(status), self.actionInput: np.array(action),self.yInput: np.array(y)}
        _, global_step,loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
        self.step = global_step
        if global_step % 100 == 0:
            print("loss",global_step,loss)

    def experience_replary_final(self):
        return np.array(random.sample(self.memory_done, self.batch_size))  
            
    def experience_replay(self): #记忆回放
        return np.array(random.sample(self.memory, self.batch_size))

    def experience_store(self,status,action,reward,done,next_status):
#         status = np.floor(np.log2(np.array(status)+1)).tolist()
#         next_status = np.floor(np.log2(np.array(next_status)+1)).tolist()
#         status = np.log10(np.array(status) + 1 ).tolist()
#         next_status = np.log10(np.array(next_status) + 1 ).tolist()
        if done:
           for action in ["a","s","w","d"]:
               self.memory.append([status,action,reward,done,next_status])
               self.memory_done.append([status,action,reward,done,next_status])
               self.file.write(str(status) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(
                   next_status) + "\n")
        self.memory.append([status,action,reward,done,next_status])
        self.file.write(str(status) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(
           next_status) + "\n")
