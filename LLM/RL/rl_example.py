import gym
from gym import spaces
import numpy as np
import random


class MultiAgentSocialMediaEnv(gym.Env):
    """
    多智能体社交媒体环境。
    """

    def __init__(self, num_agents=10, max_steps=100):
        super(MultiAgentSocialMediaEnv, self).__init__()

        self.rewards = None
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(2)  # 对于每个智能体，动作是关注或不关注
        self.observation_space = spaces.Discrete(10)  # 假设有10个不同的话题流行度等级

        self.max_steps = max_steps  # 最大步数
        self.current_step = 0  # 当前步数

    def step(self, actions):
        self.current_step += 1
        self.state = np.random.choice(self.observation_space.n, self.num_agents)

        # 示例奖励机制：每个智能体根据其动作和环境状态获得奖励
        for i, action in enumerate(actions):
            if action == 1:  # 如果智能体选择关注
                self.rewards[i] = np.random.rand()  # 随机奖励
            else:
                self.rewards[i] = 0

        # 如果达到最大步数，则结束这个episode
        done = self.current_step >= self.max_steps

        return self.state, self.rewards, done, {}

    def reset(self):
        self.current_step = 0
        self.state = np.random.choice(self.observation_space.n, self.num_agents)
        return self.state


class MultiAgentQLearning:
    def __init__(self, env, num_agents=10):
        self.env = env
        self.num_agents = num_agents
        self.q_tables = [np.zeros([env.observation_space.n, env.action_space.n]) for _ in range(num_agents)]

    def train(self, episodes=1000, learning_rate=0.1, gamma=0.6, epsilon=0.1):
        for i in range(episodes):
            states = self.env.reset()

            done = False
            while not done:
                actions = []

                for agent_idx, state in enumerate(states):
                    if np.random.uniform(0, 1) < epsilon:
                        actions.append(self.env.action_space.sample())  # 探索
                    else:
                        actions.append(np.argmax(self.q_tables[agent_idx][state]))  # 利用

                next_states, rewards, done, _ = self.env.step(actions)

                for agent_idx, (state, action, reward, next_state) in enumerate(
                        zip(states, actions, rewards, next_states)):
                    old_value = self.q_tables[agent_idx][state, action]
                    next_max = np.max(self.q_tables[agent_idx][next_state])

                    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)
                    self.q_tables[agent_idx][state, action] = new_value

                states = next_states


# 初始化环境和智能体
env = MultiAgentSocialMediaEnv(num_agents=10)
agents = MultiAgentQLearning(env, num_agents=10)

# 训练智能体
agents.train(episodes=1000)
