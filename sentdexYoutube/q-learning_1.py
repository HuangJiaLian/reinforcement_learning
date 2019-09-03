'''
@Description: q-learning
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-02 14:08:45
@LastEditors: Jack Huang
@LastEditTime: 2019-09-02 15:03:46
'''
# https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
# 安装gym
# pip install gym

import gym
env = gym.make("MountainCar-v0")

# 输出总共可以有几个动作
print(env.action_space.n)
# 3 代表有三个动作
# 0:表示向左 1:表示不动 2: 表示向右


# 第0次尝试
# env.reset()
# done = False
# while not done:
#     action = 2 
#     env.step(action)
#     env.render()


# 第1次尝试
# env.reset()
# done = False
# while not done:
#     action = 2
#     new_state, reward, done, _ = env.step(action)
#     print(reward, new_state)
#     # -1.0 [-0.37714677  0.01280429] 
#     # state里面的第一个值代表x方向的位置，第二个代表速度

# env.reset()
# done = False
# action = 2
# while not done:
#     new_state, reward, done, _ = env.step(action)
#     env.render()
#     if new_state[-1] < 0:
#         action = 0
#     else:
#         action = 2
# # 通过自己定义的策略使得小车爬上了高地

# 第三次尝试 Q-Learning
# Q-learning 的最终目的是得到Q-Table
# 首先是要建立Q-Table; 将连续的Table离散化，只因为内存不够大

print(env.observation_space.high)
print(env.observation_space.low)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
print(DISCRETE_OS_SIZE)

# 这些都是同时处理两部分的内容
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)
# [0.09  0.007] 表示位移的步长是0.09, 速度的步长是0.007

# print(DISCRETE_OS_SIZE + [env.action_space.n])
# 上面的输出是[20,20,3], 原来可以这样用啊! 


# So these values are random, and the choice to be between -2 and 0 
# is also a variable. Each step is a -1 reward, and the flag is a 0 
# reward, so it seems to make sense to make the starting point of 
# random Q values all negative.
import numpy as np
q_table = np.random.uniform(low=-2, high=0, \
          size=(DISCRETE_OS_SIZE + [env.action_space.n]))

print(q_table)
# This table is our bible. We will consult with this table to 
# determine our moves. That final x3 is our 3 actions and 
# each of those 3 actions have the "Q value" associated with them


# When we're being "greedy" and trying to "exploit" our environment, 
# we will choose to go with the action that has the highest Q value 
# for this state. Sometimes, however, especially initially, 
# we may instead wish to "explore" and just choose a random action. 

