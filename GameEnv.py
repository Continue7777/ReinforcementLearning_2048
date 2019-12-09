#!/usr/bin/env python
# coding=utf-8
# ********************************************************
# > OS     : Linux 3.2.0-60-generic #91-Ubuntu
#	> Author : yaolong
#	> Mail   : dengyaolong@yeah.net
#	> Time   : 2014年06月01日 星期日 13:13:39
# ********************************************************
from __future__ import print_function
import random
import copy
import numpy as np


class Game2048:
    def __init__(self,grid_n=4,reward_type="score_empty_higher"):
        self.score = 0
        self.step_num = 0
        self.episode = 0
        self.max_score = 0
        self.max_matrix = [[]]

        self.reward_type = reward_type #score_update  score_update + empty_update + get_higher_max
        self.grid_n = grid_n
        self.reset()

    def reset(self,matrix=None):
        if matrix is None:
            mtr = [[0 for i in range(self.grid_n)] for j in range(self.grid_n)]  # 小小蛋疼..
            ran_pos = random.sample(range(self.grid_n**2), 2)
            mtr[int(ran_pos[0] / self.grid_n)][int(ran_pos[0] % self.grid_n)] = mtr[int(ran_pos[1] / self.grid_n)][int(ran_pos[1] % self.grid_n)] = 2
            self.matrix = mtr
        else:
            self.matrix = matrix


        self.step_num = 0
        self.score = 0
        self.episode += 1

    def step(self, action, show=False):
        """
          step:take the action and return status,reward,done,next_status
        """
        if self.is_over(self.matrix):
            if self.score > self.max_score:
                self.max_score = self.score
                self.max_matrix = self.matrix
            return copy.deepcopy(self.matrix),0, True

        if action == "a":
            dirct = 0
        elif action == "s":
            dirct = 1
        elif action == "w":
            dirct = 2
        elif action == "d":
            dirct = 3
        else:
            dirct = 0
            print("wrong")
        tmp = copy.deepcopy(self.matrix)
        cur_score = self.move(self.matrix, dirct)
        if tmp != self.matrix:
            self.score += cur_score
            self.update(self.matrix)  # 更新
            if show: self.display(self.matrix)
            self.step_num += 1  # 步数加1
            
            score_update = np.log2(cur_score+1) / 10.0
            empty_update = max(np.sum(np.array(self.matrix) == 0) - np.sum(np.array(tmp) == 0),0) 
            higher_update = np.log2(np.max(self.matrix)) / 10.0 if np.max(self.matrix) != np.max(tmp) else 0
            if self.reward_type == "score_empty_higher":
                score_final = score_update + empty_update + higher_update
            elif self.reward_type == "score":
                score_final = score_update
            elif self.reward_type == "higher":
                score_final = higher_update
            elif self.reward_type == "empty_higher":
                score_final = higher_update + empty_update
            return copy.deepcopy(self.matrix), score_final, False
#             return copy.deepcopy(self.matrix), np.sum(np.array(self.matrix) == 0), False
        else:
            return copy.deepcopy(self.matrix), 0, False
#         return copy.deepcopy(self.matrix), cur_score, False

    def display(self, mtr=None):
        def T(a):
            return str(a) + "\t" if a else '-' + "\t"

        if mtr is None:
            mtr = self.matrix
        for i in range(self.grid_n):
            for j in range(self.grid_n):
                print(T(mtr[i][j]), end=" ")
            print("\n")

    def is_over(self, mtr):
        for i in range(self.grid_n):
            if 0 in mtr[i]:
                return False
        for i in range(self.grid_n):
            for j in range(self.grid_n):
                if i < self.grid_n-1 and mtr[i][j] == mtr[i + 1][j]:
                    return False
                if j < self.grid_n-1 and mtr[i][j] == mtr[i][j + 1]:
                    return False
        return True

    def move_down(self,mtr):
        visit = []
        score = 0
        for j in range(self.grid_n):
            for i in range(self.grid_n-1, 0, -1):
                for k in range(0, i):
                    if mtr[k + 1][j] == 0:
                        mtr[k + 1][j] = mtr[k][j]
                        mtr[k][j] = 0
                    elif mtr[k + 1][j] == mtr[k][j] and (self.grid_n * (k + 1) + j) not in visit and (self.grid_n * k + j) not in visit:
                        mtr[k + 1][j] *= 2
                        mtr[k][j] = 0
                        score += mtr[k + 1][j]
                        visit.append(self.grid_n * (k) + j)
                        visit.append(self.grid_n * (k + 1) + j)
        return score
    
    def move_up(self,mtr):
        mtr = np.flipud(mtr).tolist()
        score = self.move_down(mtr)
        self.matrix = np.flipud(mtr).tolist()
        return score
    
    def move_left(self,mtr):
        mtr = np.flipud(np.transpose(mtr).tolist())
        score = self.move_down(mtr)
        self.matrix = np.transpose(np.flipud(mtr)).tolist()
        return score
    
    def move_right(self,mtr):
        mtr = np.transpose(np.flipud(mtr)).tolist()
        score = self.move_down(mtr)
        self.matrix = np.flipud(np.transpose(mtr)).tolist()
        return score
    
    def move(self, mtr, dirct):
        score = 0
        visit = []
        if dirct == 0:  # left
            score += self.move_left(mtr)
        elif dirct == 1:  # down
            score += self.move_down(mtr)
        elif dirct == 2:  # up
            score += self.move_up(mtr)
        elif dirct == 3:  # right
            score += self.move_right(mtr)

        return score

    def update(self, mtr):
        ran_pos = []
        ran_num = [2, 4]

        for i in range(self.grid_n):
            for j in range(self.grid_n):
                if mtr[i][j] == 0:
                    ran_pos.append(self.grid_n * i + j)
        if len(ran_pos) > 0:
            k = random.choice(ran_pos)
            n = random.choice(ran_num)
            mtr[int(k / self.grid_n)][int(k % self.grid_n)] = n

if __name__ == '__main__':
    gameEnv = Game2048()
    # gameEnv.display()
    for i in range(100):
        gameEnv.reset()
        while 1:
            action = random.choice(['a', 's', 'w', 'd'])
            s, r, d, s1 = gameEnv.step(action)
            if d:
                break
        if gameEnv.score > 2400:
            print(i, gameEnv.score, gameEnv.max_score)
    gameEnv.display(gameEnv.max_matrix)
    print("max_score:",gameEnv.max_score)