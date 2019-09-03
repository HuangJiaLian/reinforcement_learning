'''
@Description: q-learning3,主要关注地是怎么调整参数，如何更快地达到目的
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-02 14:08:45
@LastEditors: Jack Huang
@LastEditTime: 2019-09-03 11:05:22
'''

# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")
env.reset()

# Q-Learning 设置
LEARNING_RATE = 0.1
DISCOUNT = 0.95
SHOW_EVERY = 500

EPISODES = 20000
STATS_EVERY = 10

# 添加想要记录的量
ep_rewards = []
# 下面这种用法好高级哦
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

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
discrete_os_win_size = (env.observation_space.high - \
                       env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, \
          size=(DISCRETE_OS_SIZE + [env.action_space.n]))

discrete_state = get_discrete_state(env.reset())
for episode in range(EPISODES):
    # 
    episode_reward = 0
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
        # 记录奖励
        episode_reward += reward

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
    
    # 记录信息
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        # 这里的 [-STATS_EVERY:] 反向索引方式真的好赞
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    
    # 保存Q-table
    if episode % 100 == 0:
        # 这里的保存方法是值得学习的
        np.save(f"qtables/{episode}-qtable.npy", q_table)
env.close()


# 可视化
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True) 
plt.show()
