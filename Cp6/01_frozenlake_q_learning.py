import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

# /////////////////// CP5  VALUE ITERATIION (OVER SAMPLE) //////////////
    # def play_n_random_steps(self, count):
    #     for _ in range(count):
    #         action = self.env.action_space.sample()
    #         new_state, reward, is_done, _ = self.env.step(action)
    #         self.rewards[(self.state, action, new_state)] = reward
    #         self.transits[(self.state, action)][new_state] += 1
    #         self.state = self.env.reset() if is_done else new_state

    def best_value_and_action(self, state):
        best_action, best_value = None,None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)] 
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

# /////////////////// CP5  VALUE ITERATIION (OVER SAMPLE) //////////////
    # def value_iteration(self):
    #     for state in range(self.env.observation_space.n):
    #         for action in range(self.env.action_space.n):
    #             action_value = 0.0
    #             target_counts = self.transits[(state, action)]
    #             total = sum(target_counts.values())
    #             for tgt_state, count in target_counts.items():
    #                 reward = self.rewards[(state, action, tgt_state)]
    #                 best_action = self.select_action(tgt_state)
    #                 val = reward + GAMMA * self.values[(tgt_state, best_action)]
    #                 action_value += (count/total) * val
    #             self.values[(state, action)] = action_value

if __name__ == "__main__":
    
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if iter_no != 0  and iter_no%1000==0:
            print("New iteration {} with reward: {} \n".format(iter_no, reward))
        if reward > best_reward:
            best_reward = reward
            print("New best reward, what a beast! ...... {} \n".format(best_reward))
        if reward > 0.8:
            print("We are done here my king, IS DONE!")
            break
        writer.close()
    