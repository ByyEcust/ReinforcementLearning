from sarsa_.sarsa import SARSA
from game.climb_mountains.climb_mountains import ClimbMountain
import time
import matplotlib.pyplot as plt
import seaborn as sns


def train_sarsa_climb_mountain(env, agent, **train_args):
    init_epsilon = train_args.get("init_epsilon", 0.3)
    final_epsilon = train_args.get("final_epsilon", 0.001)
    epsilon_decay = train_args.get("epsilon_decay", 0.99)
    epsilon = init_epsilon
    lr = train_args.get("lr", 0.01)
    max_iterations = train_args.get("max_iterations", 3000)
    warmup = train_args.get("warmup", 50)

    for epoch in range(max_iterations):
        t0 = time.time()
        total_reward, done = 0, False
        state = env.reset()
        state_no = env.state2num(state)
        if epoch < warmup:
            action = agent.choose_action(state_no=state_no, epsilon=1)
        else:
            action = agent.choose_action(state_no=state_no, epsilon=epsilon)
        while not done:
            next_state, r, done = env.step(action + 1)
            next_state_no = env.state2num(next_state)

            if epoch < warmup:
                next_action = agent.choose_action(state_no=next_state_no, epsilon=1)
            else:
                next_action = agent.choose_action(state_no=next_state_no, epsilon=epsilon)

            agent.train(state_no, action, r, next_state_no, next_action, done, lr)
            total_reward += r

            state_no = next_state_no
            action = next_action

        epsilon = max(epsilon * epsilon_decay, final_epsilon)
        print("Epoch: {}, Time Consume: {} | Reward: {} | Reach: {}".format(epoch, time.time() - t0, total_reward, next_state))

        q_table = agent.q_table
        plt.clf()
        sns.heatmap(q_table)
        plt.pause(0.01)


env = ClimbMountain("mount_cfg.json")
agent = SARSA(num_state=50, num_action=4, gamma=1)
train_sarsa_climb_mountain(env, agent, warmup=100, lr=0.3)

import pandas as pd
pd.DataFrame(agent.q_table).to_csv("../../Q-Learning/sarsa_q_table.csv")