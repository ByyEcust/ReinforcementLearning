from q_learning.q_leaning import QLearning
from game.climb_mountains.climb_mountains import ClimbMountain
import time
import matplotlib.pyplot as plt
import seaborn as sns


def train_q_learning_climb_mountain(env, agent, **train_args):
    init_epsilon = train_args.get("init_epsilon", 0.3)
    final_epsilon = train_args.get("final_epsilon", 0.001)
    epsilon_decay = train_args.get("epsilon_decay", 0.99)
    epsilon = init_epsilon
    lr = train_args.get("lr", 0.01)
    max_iterations = train_args.get("max_iterations", 10000)
    warmup = train_args.get("warmup", 50)

    for epoch in range(max_iterations):
        t0 = time.time()
        total_reward, done = 0, False
        state = env.reset()
        state_no = env.state2num(state)
        while not done:
            if epoch < warmup:
                action = agent.choose_action(state_no=state_no, epsilon=1)
            else:
                action = agent.choose_action(state_no=state_no, epsilon=epsilon)
            next_state, r, done = env.step(action + 1)
            next_state_no = env.state2num(next_state)

            agent.train(state_no, action, r, next_state_no, done, lr)
            total_reward += r

            state_no = next_state_no

        epsilon = max(epsilon * epsilon_decay, final_epsilon)
        print("Epoch: {}, Time Consume: {} | Reward: {} | Reach: {}".format(epoch, time.time() - t0, total_reward, next_state))

        q_table = agent.q_table
        plt.clf()
        sns.heatmap(q_table)
        plt.pause(0.01)


env = ClimbMountain("mount_cfg.json")
agent = QLearning(num_state=50, num_action=4, gamma=1)
train_q_learning_climb_mountain(env, agent, warmup=100, lr=0.3)

