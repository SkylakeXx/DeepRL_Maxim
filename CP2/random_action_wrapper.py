import gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("RANDOM!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    env = gym.wrappers.Monitor(env, "/home/light/Projects/DEEP_DATA/CP2", force=True)
    obs = env.reset()
    total_reward = 0

    while True:
        obs, reward, done, _ = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            break
    print("GOT REWARD OF: {}".format(total_reward))




