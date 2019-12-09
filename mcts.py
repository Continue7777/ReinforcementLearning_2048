#-*-coding:utf-8-*-
from GameEnv import Game2048
import numpy as np
import copy
class Node:
    def __init__(self,matrix):
        self.matrix = matrix
        self.n = 0
        self.score = 0.0
        self.max_score = 0.0
        self.children_dict = {
            "a":[],
            "s":[],
            "w":[],
            "d":[]
        }

class MCTS:
    def __init__(self):
        self.node_dict = {}
        self.gameEnv = Game2048()
        self.gameEnv.grid_n = 4
        self.ucb_debug = True
        self.predict_debug = True
        self.search_debug = True
        self.search_n = 200
    
    def search(self,matrix=None):
        for i in range(self.search_n):
            if self.search_debug:
                if str(matrix) in self.node_dict:
                    print "matrix:",matrix
                    print "matrix_num:",self.node_dict[str(matrix)].n
                    print "children:",self.node_dict[str(matrix)].children_dict
            if matrix == None:
                self.gameEnv.reset()
            else:
                self.gameEnv.reset(copy.deepcopy(matrix))
            sum_score = 0
            node_list = []
            while True:
                tmp = copy.deepcopy(self.gameEnv.matrix)
                if str(self.gameEnv.matrix) in self.node_dict:
                    cur_node = self.node_dict[str(self.gameEnv.matrix)]
                else:
                    cur_node = Node(copy.deepcopy(self.gameEnv.matrix))
                    self.node_dict[str(self.gameEnv.matrix)] = cur_node
                observation_next,reward,action,done = self.select(cur_node)
                if self.search_debug:
                    print "action:",done,action,cur_node.matrix
                if observation_next == tmp and done is False:
                    cur_node.children_dict[action] = -1
                    continue
                else:
                    if str(observation_next) in cur_node.children_dict[action]: 
                        pass
                    elif len(cur_node.children_dict[action]) == 0:
                        cur_node.children_dict[action] = [str(observation_next)]
                    else:
                        cur_node.children_dict[action].append(str(observation_next))
                sum_score += reward
                node_list.append(cur_node)
                if done:
                    break
                    
            # 反向计算值
#             print "node_list:",len(node_list)
            for node in set(node_list):
#                 print sum_score,node.n
                node.score = (node.score * node.n + sum_score) / (node.n + 1)
                node.max_score = max(node.max_score,sum_score)
                node.n += 1
                
    
    def ucb(self,node):
        for action in ["a","s","w","d"]:
            if node.children_dict[action] == -1:continue
            if len(node.children_dict[action]) == 0:
                return action
#         print node.children_dict["a"]
#         print node.children_dict["s"]
#         print node.children_dict["w"]
#         print node.children_dict["d"]
        w_i = []
        n_i = []
        for a in ["a","s","w","d"]:
            if node.children_dict[a] != -1:
                w_i.append(np.sum([self.node_dict[i].score for i in node.children_dict[a]]))
                n_i.append(np.sum([self.node_dict[i].n for i in node.children_dict[a]]))
            else:
                w_i.append(-1)
                n_i.append(1)

        t = np.sum(n_i)
        
        c = 100
        score_i = []
        for i in range(4):
            if w_i[i] == -1:
                score_i.append(-1)
            else:
                score_i.append(w_i[i] / n_i[i] + c * np.sqrt(np.log(t) / n_i[i]))
        if self.ucb_debug:
            print w_i
            print n_i
            print score_i
            print node.matrix
            print "****************************"
        l = ["a","s","w","d"]
        index = np.argmax(score_i)
        return l[index]
    
    def select(self,node):
        # expand 
        action = self.ucb(node)
            
        observation_, reward, done = self.gameEnv.step(action)
        return observation_,reward,action,done
        
            
    def predict(self,matrix):
        if str(matrix) not in self.node_dict:
            self.search(copy.deepcopy(matrix))
        node = self.node_dict[str(matrix)]
        for action in ["a","s","w","d"]:
            if node.children_dict[action] == -1:
                continue
               
#         print "predict start"
        w_i = []
        n_i = []
        for a in ["a","s","w","d"]:
            if node.children_dict[a] != -1:
                w_i.append(np.sum([self.node_dict[i].score for i in node.children_dict[a]]))
#                 n_i.append(np.sum([self.node_dict[i].n for i in node.children_dict[a]]))
                n_i.append(len(node.children_dict[a]))
            else:
                w_i.append(0)
                n_i.append(1)

        score_i = []
        for i in range(4):
            score_i.append(w_i[i] / n_i[i])
        if self.predict_debug:
            print w_i
            print n_i
            print score_i
            
        l = ["a","s","w","d"]
        index = np.argmax(score_i)
        return l[index]