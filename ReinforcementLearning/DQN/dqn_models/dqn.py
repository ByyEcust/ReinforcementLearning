from utils.utils_all import Buffer
from torch.optim.adam import Adam
import torch
import numpy as np


class DQN(object):
    def __init__(self, state_shape, action_shape, action_value_model, **args):
        super(DQN, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = args.get("Device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = args.get("Gamma", 0.95)
        self.batch_size = args.get("BatchSize", 64)
        self.action_value_model = action_value_model.to(self.device)
        self.optimizer = Adam(self.action_value_model.parameters(), lr=args.get("LearningRate", 1e-3))
        self.rpm = Buffer(args.get("ReplayMemorySize", 10000))

    def update_model(self):
        # sample a batch update the model
        self.action_value_model.train()
        sample_batch = self.rpm.sample_batch(self.batch_size)
        states, actions, next_states, rewards, done = (self.to_tensor(value) for value in decode_batch(sample_batch))
        target_values = self.action_value_model(self.norm(next_states)).max(1)[0].detach() * self.gamma * (1 - done) + rewards
        current_values = (self.action_value_model(self.norm(states)) * actions).sum(1)
        td_errors = target_values - current_values

        loss = (td_errors ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def choose_action(self, state, epsilon):
        # epsilon-greedy policy
        states = state[np.newaxis, :]
        random_actions = self.random_action(states)
        if epsilon == 1:
            return random_actions[0]
        greedy_actions = self.greedy_action(states)
        actions = [ra if np.random.rand() < epsilon
                   else ga for ra, ga in zip(random_actions, greedy_actions)]
        actions = np.array(actions)
        return actions[0]

    def random_action(self, state):
        out = np.random.rand(state.shape[0], self.action_shape)
        out[:, 1] *= 0.2
        actions = decode_action(out)
        return actions

    def greedy_action(self, state):
        state_tensor = self.to_tensor(state)
        out = self.to_array(self.action_value_model(self.norm(state_tensor)))
        actions = decode_action(out)
        return actions

    def observe_state(self, trans_tuples):
        # trans_tuples: <state, action, next_state, reward, done>
        self.rpm.append(trans_tuples)

    def norm(self, state):
        # roughly normalize the state variable
        normed = (state.permute([0, 2, 3, 1]) - torch.FloatTensor([40.27] * 4).to(self.device)) / \
                 torch.FloatTensor([40.27] * 4).to(self.device)
        return normed.permute([0, 3, 1, 2])

    def train(self):
        # switch to train mode
        self.action_value_model.train()

    def test(self):
        # switch to test mode
        self.action_value_model.eval()

    def save_state_dict(self, path):
        torch.save(self.action_value_model.state_dict(), path)
        return True

    def load_state_dict(self, path):
        self.action_value_model.load_state_dict(torch.load(path))
        return True

    def to_tensor(self, array):
        return torch.tensor(array).float().to(self.device)

    def to_array(self, tensor):
        if self.device == "cpu":
            return tensor.data.numpy()
        else:
            return tensor.to("cpu").data.numpy()


def decode_action(arr: np.array):
    # decode q-values to action vector
    # array: batch * num_action
    res = np.zeros(arr.shape)
    for r, row in enumerate(arr):
        res[r, row.argmax(0)] = 1
    return res


def decode_batch(sample_batch):
    # decode data from tuples
    assert len(sample_batch) > 0, "The data batch to decode is empty."
    num_tuple = len(sample_batch[0])
    res = (np.array([sample[idx] for sample in sample_batch]) for idx in range(num_tuple))
    return res
