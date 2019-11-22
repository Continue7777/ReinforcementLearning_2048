#-*- coding:utf-8
import numpy as np
import tensorflow as tf
import random
import codecs

from GameEnv import Game2048


class DQN:
    def __init__(self,learning_rate=0.01,grid_n = 4,batch_size=1000): #初始化
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sigema = 0.05
        self.step = 0
        self.explore_alpha = 0.9 ** (self.step / 1000)

        self.grid_n = grid_n
        self.actions_index_dicts = {"a":0,"s":1,"w":2,"d":3}
        self.actions_index_dicts_reverse = {0:"a",1:"s",2:"w",3:"d"}
        self.actions_index_keys = ["a","s","w","d"]

        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.memory = []
        self.memory_open = {}
        self.file = codecs.open("train_data.csv","w",encoding='utf-8')


    def build_network(self): #构建网络模型，简单先搞个DNN

        self.global_steps = tf.Variable(0, trainable=False)
        self.matrixInput = tf.placeholder(shape=[None,self.grid_n,self.grid_n],dtype=tf.float32,name="matrixInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")

        matrixFlat = tf.reshape(self.matrixInput,shape=[-1,self.grid_n ** 2])
        layer1 = tf.layers.dense(matrixFlat, 128, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(layer1, 64, activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(layer2,4)


        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)


    def _greedy_e(self,seq,probabilities):
        e = 0.6 ** (self.step / 1000)
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
        prob_all = self.sess.run(self.predictions, feed_dict={self.matrixInput:np.array(status)})
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
        status = np.array([i for i in train_data[:,0]])
        action = self._one_hot([self.actions_index_dicts[i] for i in train_data[:,1]])
        reward = train_data[:,2]
        done = train_data[:,3]
        next_status = np.array([i for i in train_data[:,4]])


        # print (reward)
        # print (action)
        maxQNext = self.get_max_availble_action_value(next_status)
        y = []
        for i in range(self.batch_size):
            if done[i] == True:
                y.append(0)
            else:
                y.append(reward[i] + self.sigema * maxQNext[i])

        feed_dict = {self.matrixInput: np.array(status), self.actionInput: np.array(action),self.yInput: np.array(y)}
        _, global_step,loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
        self.step = global_step
        if global_step % 100 == 0:
            print("loss",global_step,loss)


    def experience_replay(self): #记忆回放
        return np.array(random.sample(self.memory, self.batch_size))

    def experience_store(self,status,action,reward,done,next_status):
        if status != next_status:
           self.memory.append([status,action,reward,done,next_status])
           self.file.write(str(status) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(
               next_status) + "\n")