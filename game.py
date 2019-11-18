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



class Game2048:
    def __init__(self):
        self.score = 0
        self.step_num = 0
        self.episode = 0
        self.max_score = 0
        self.max_matrix = [[]]

        self.n = 4
        self.reset()

    def reset(self):
        mtr = [[0 for i in range(4)] for j in range(4)]  # 小小蛋疼..
        ran_pos = random.sample(range(16), 2)
        mtr[int(ran_pos[0] / 4)][int(ran_pos[0] % 4)] = mtr[int(ran_pos[1] / 4)][int(ran_pos[1] % 4)] = 2

        self.matrix = mtr
        self.step_num = 0
        self.score = 0
        self.episode += 1

    def step(self, action, show=False):
        """
        step:take the action and return status,reward,done,next_status
        """
        if action == "a":
            dirct = 0
        elif action == "s":
            dirct = 1
        elif action == "w":
            dirct = 2
        elif action == "d":
            dirct = 3
        else:
            print(illegal)
        tmp = copy.deepcopy(self.matrix)
        cur_score = self.move(self.matrix, dirct)
        if tmp != self.matrix:
            self.score += cur_score
            self.update(self.matrix)  # 更新
            if show: self.display(self.matrix)
            self.step_num += 1  # 步数加1
        if self.is_over(self.matrix):
            if self.score > self.max_score:
                self.max_score = self.score
                self.max_matrix = self.matrix
            return tmp, cur_score, True, self.matrix
        else:
            return tmp, -1000, False, self.matrix

    def display(self, mtr=None, n=4):
        def T(a):
            return str(a) + "\t" if a else '-' + "\t"

        if mtr is None:
            mtr = self.matrix
        for i in range(n):
            for j in range(n):
                print(T(mtr[i][j]), end=" ")
            print("\n")

    def is_over(self, mtr):
        for i in range(4):
            if 0 in mtr[i]:
                return False
        for i in range(4):
            for j in range(4):
                if i < 3 and mtr[i][j] == mtr[i + 1][j]:
                    return False
                if j < 3 and mtr[i][j] == mtr[i][j + 1]:
                    return False
        return True

    def move(self, mtr, dirct):
        score = 0
        visit = []
        if dirct == 0:  # left
            for i in range(4):
                for j in range(1, 4):
                    for k in range(j, 0, -1):
                        if mtr[i][k - 1] == 0:
                            mtr[i][k - 1] = mtr[i][k]
                            mtr[i][k] = 0
                        elif mtr[i][k - 1] == mtr[i][k] and 4 * i + k - 1 not in visit and 4 * i + k not in visit:
                            mtr[i][k - 1] *= 2
                            mtr[i][k] = 0
                            score += mtr[i][k - 1]
                            visit.append(4 * i + k)
                            visit.append(4 * i + k - 1)
                            # for i in range(4):
                            #    for j in range(3):

        elif dirct == 1:  # down
            for j in range(4):
                for i in range(3, 0, -1):
                    for k in range(0, i):
                        if mtr[k + 1][j] == 0:
                            mtr[k + 1][j] = mtr[k][j]
                            mtr[k][j] = 0
                        elif mtr[k + 1][j] == mtr[k][j] and (4 * (k + 1) + j) not in visit and (4 * k + j) not in visit:
                            mtr[k + 1][j] *= 2
                            mtr[k][j] = 0
                            score = mtr[k + 1][j]
                            visit.append(4 * (k) + j)
                            visit.append(4 * (k + 1) + j)


        elif dirct == 2:  # up
            for j in range(4):
                for i in range(1, 4):
                    for k in range(i, 0, -1):
                        if mtr[k - 1][j] == 0:
                            mtr[k - 1][j] = mtr[k][j]
                            mtr[k][j] = 0
                        elif mtr[k - 1][j] == mtr[k][j] and (4 * (k - 1) + j) not in visit and (4 * k + j) not in visit:
                            mtr[k - 1][j] *= 2
                            mtr[k][j] = 0
                            score += mtr[k - 1][j]
                            visit.append(4 * (k) + j)
                            visit.append(4 * (k - 1) + j)

        elif dirct == 3:  # right
            for i in range(4):
                for j in range(3, 0, -1):
                    for k in range(j):
                        if mtr[i][k + 1] == 0:
                            mtr[i][k + 1] = mtr[i][k]
                            mtr[i][k] = 0
                        elif mtr[i][k] == mtr[i][k + 1] and 4 * i + k + 1 not in visit and 4 * i + k not in visit:
                            mtr[i][k + 1] *= 2
                            mtr[i][k] = 0
                            score += mtr[i][k + 1]
                            visit.append(4 * i + k + 1)
                            visit.append(4 * i + k)

        return score

    def update(self, mtr):
        ran_pos = []
        ran_num = [2, 4]

        for i in range(4):
            for j in range(4):
                if mtr[i][j] == 0:
                    ran_pos.append(4 * i + j)
        if len(ran_pos) > 0:
            k = random.choice(ran_pos)
            n = random.choice(ran_num)
            mtr[int(k / 4)][int(k % 4)] = n

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