from DQN.dqn_models.dqn import DQN
from game.flappy_bird.wrapped_flappy_bird import GameState
from utils.utils_all import RollAverage
import time
from DQN.dqn_models.action_value_models import ActionValueModelFlappyBird


def train_dqn(agent: DQN, env: GameState, **args):
    t0 = time.time()
    agent.train()
    reward_average = RollAverage(args.get("NumAvg", 10))
    epsilon = args.get("InitEpsilon", 0.5)
    epsilon_final = args.get("FinalEpsilon", 0.01)
    epsilon_decay = args.get("EpsilonDecay", 0.999)
    # train loops
    for iteration in range(args.get("MaxIterations", 10000)):
        state, done = env.get_state(), False
        total_reward_episode = 0
        while not done:
            if iteration < args.get("WarmUp", 30):
                # choose random action for warm up
                action = agent.choose_action(state, epsilon=1)
            else:
                # epsilon-greedy policy
                action = agent.choose_action(state, epsilon=epsilon)
            next_state, reward, done = env.frame_step(action)
            agent.observe_state((state, action, next_state, reward, done))
            state = next_state
            total_reward_episode += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_final)
        reward_average.update(total_reward_episode)        # Roll average for further figure plot

        # save model parameters every 1000 episodes
        if iteration % 1000 == 0:
            agent_dqn.save_state_dict("results/model_param_{}.pt".format(iteration))

        # update model parameters 5 times for each episode (This could be move into the while loop at 20 line)
        loss_train = 0
        if iteration >= args.get("WarmUp", 30):
            for _ in range(5):
                loss_train += agent_dqn.update_model() / 5

        # print log info
        print("Iteration: {} | Time Consume: {} | Total Reward: {} | Loss Train: {}".format(
            iteration, time.time() - t0, total_reward_episode, loss_train))
        t0 = time.time()


if __name__ == "__main__":
    state_shape = (4, 80, 80)
    action_shape = 2
    agent_args = {"Device": "cpu",
                  "Gamma": 0.99,
                  "BatchSize": 64,
                  "LearningRate": 1e-4,
                  "ReplayMemorySize": 50000}
    train_args = {"MaxIterations": 10000,
                  "WarmUp": 30,
                  "InitEpsilon": 0.3,
                  "FinalEpsilon": 0.01,
                  "EpsilonDecay": 0.999}
    action_value_model_flappy_bird = ActionValueModelFlappyBird(state_shape, action_shape)
    agent_dqn = DQN(state_shape=state_shape,
                    action_shape=action_shape,
                    action_value_model=action_value_model_flappy_bird, **agent_args)
    env_flappy_bird = GameState()

    train_dqn(agent_dqn, env_flappy_bird, **train_args)
