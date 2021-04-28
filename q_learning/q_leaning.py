import numpy as np


class QLearning(object):
    def __init__(self, num_state, num_action, **args):
        self.num_state = num_state
        self.num_action = num_action
        self.q_table = np.zeros((num_state, num_action))
        self.gamma = args.get("gamma", 0.95)

    def train(self, state_no, action_no, reward, next_state_no, done, lr=1e-2):
        self.q_table[state_no, action_no] += lr * (reward + self.gamma * max(self.q_table[next_state_no, :]) * float(1 - done)
                                                   - self.q_table[state_no, action_no])

    def choose_action(self, state_no, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_action-1)
        else:
            return np.argmax(self.q_table[state_no, :])

