'''
@Description: q-learning2
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-02 14:08:45
@LastEditors: Jack Huang
@LastEditTime: 2019-09-02 21:13:12
'''
# https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
# 安装gym
# pip install gym
import numpy as np
import gym
env = gym.make("MountainCar-v0")
env.reset()

# Q-Learning 设置
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500
# 探索，尝试随机的动作
epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2 # 表示在前期做一些探索，后期就不做探索了，这是一个比较好的策略
# 每一步epsilon要减小的值，也就是是说随着训练的不断进行，探索的比例会不断减小。
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# 将得到的连续状态变成离散的
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# print(DISCRETE_OS_SIZE)

# 这些都是同时处理两部分的内容
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, \
          size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table)

discrete_state = get_discrete_state(env.reset())
# print(discrete_state)
# Index by tuple 又学到一招
# print(q_table[discrete_state]) # 输出Table对应格子上的每个Action的值
# print(np.argmax(q_table[discrete_state])) # 输出值最大的Action

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True 
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        # action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        # print(reward, new_state)
        if render:
            env.render() 
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >=  env.goal_position:
            # 奖励就是没有惩罚
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0  
        discrete_state = new_discrete_state
    
    # 这里可以连着一起写，长知识了
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
env.close()



