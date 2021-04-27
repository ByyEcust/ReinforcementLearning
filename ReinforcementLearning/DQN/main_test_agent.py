from DQN.dqn_models.dqn import DQN
from game.flappy_bird.wrapped_flappy_bird import GameState
from DQN.dqn_models.action_value_models import ActionValueModelFlappyBird
import torch
from DQN.dqn_models.dqn import decode_action


def test_dqn(agent: DQN, env: GameState):
    state, done = env.get_state(), False
    total_reward_episode = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.frame_step(action)
        state = next_state
        total_reward_episode += reward

    print("Total reward is: {}".format(total_reward_episode))


class Agent(object):
    def __init__(self, state_shape, action_shape, device="cpu"):
        self.state_shape = (4, 80, 80)
        self.action_shape = 2
        self.action_value_model = ActionValueModelFlappyBird(state_shape, action_shape)
        self.device = device

    def load(self, param_file):
        self.action_value_model.load_state_dict(torch.load(param_file, map_location=self.device))

    def choose_action(self, state):
        self.action_value_model.eval()
        state_tensor = self.to_tensor([state])
        out = self.to_array(self.action_value_model(self.norm(state_tensor)))
        actions = decode_action(out)
        return actions[0]

    def to_tensor(self, array):
        return torch.tensor(array).float().to(self.device)

    def to_array(self, tensor):
        if self.device == "cpu":
            return tensor.data.numpy()
        else:
            return tensor.to("cpu").data.numpy()

    def norm(self, state):
        normed = (state.permute([0, 2, 3, 1]) - torch.FloatTensor([40.27] * 4).to(self.device)) / \
                 torch.FloatTensor([40.27] * 4).to(self.device)
        return normed.permute([0, 3, 1, 2])


state_shape = (4, 80, 80)
action_shape = 2
agent = Agent(state_shape, action_shape)
agent.load("results/model_param_10000.pt")
env_flappy_bird = GameState()
test_dqn(agent, env_flappy_bird)
