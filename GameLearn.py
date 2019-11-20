#-*-coding:utf-8-*-
from GameEnv import Game2048
from DQN import DQN
import random
gameEnv = Game2048()
RL = DQN()
observation = gameEnv.matrix
for episode in range(10000):
    # 初始化环境
    gameEnv.reset()

    while True:
        # DQN 根据观测值选择行为
        action = RL.choose_action([observation])

        # 环境根据行为给出下一个 state, reward, 是否终止
        observation_, reward, done = gameEnv.step(action)

        # DQN 存储记忆
        RL.experience_store(observation, action, reward,done,observation_)

        # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
        if (episode > 20) and (episode % 5 == 0):
            RL.train()

        # 将下一个 state_ 变为 下次循环的 state
        observation = observation_

        # 如果终止, 就跳出循环
        if done:
            break

    if episode % 100 == 0:
        gameEnv.reset()
        flag = False
        while True:
            if flag:
                action = random.choice(["a","w","d","s"])
            else:
                action = RL.choose_action_max([observation])
            observation_, reward, done = gameEnv.step(action)
            flag = True if observation_ == observation else False
            observation = observation_
            if done:
                print("this_score_____________________:",gameEnv.score)
                break
