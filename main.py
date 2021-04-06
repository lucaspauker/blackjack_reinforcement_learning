import numpy as np
import os
import json
import matplotlib.pyplot as plt

from blackjack import Blackjack
from scheduler import Scheduler
from policy import Policy

ACTION_DIM = 2
STATE_DIM = 18 * 2 * 10 + 1
TEST_NUM_TRIALS = 1000000
NUM_DECKS = 6
GRAPH_NUM_TIMESTEPS = 250

def test_policy(p, env, n=TEST_NUM_TRIALS):
    wins = 0
    losses = 0
    rounds = 0
    for t in range(1, n):
        env.shuffle_deck()
        while True:
            total, soft, hole = env.get_state()
            state = Blackjack.get_state_number(total, soft, hole)
            action = p.get_action(state)
            env.take_action(action)

            if env.player_turn_done():
                break
        reward = env.get_reward()
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        rounds += 1
    return wins / rounds, losses / rounds

def get_timesteps_vs_win_rate_data():
    data_file = "output/tmp_policy.txt"
    env = Blackjack(NUM_DECKS)
    win_loss_rates = []
    s = Scheduler(ACTION_DIM, STATE_DIM, 0)
    for i in range(GRAPH_NUM_TIMESTEPS):
        print("Iteration number", i + 1)
        s.step_through()
        s.save_policy(filename=data_file)
        p = Policy(ACTION_DIM, STATE_DIM, data_file)
        win_rate, loss_rate = test_policy(p, env, n=TEST_NUM_TRIALS)
        win_loss_rates.append((win_rate, loss_rate))
        s.n += 1
    os.remove(data_file)
    return win_loss_rates

def graph_timesteps_vs_win_rate(win_loss_rates):
    optimal_win_rate, _ = test_optimal_policy()
    stand_win_rate, _ = test_stand_policy()
    win_rates = []
    for tup in win_loss_rates:
        win_rates.append(tup[0])
    plt.plot(range(0, GRAPH_NUM_TIMESTEPS), win_rates, color="green", label="Q learning")
    plt.title("Win rate vs. number of time steps")
    plt.xlabel("Number of time steps")
    plt.ylabel("Win rate")
    plt.hlines(optimal_win_rate, 0, GRAPH_NUM_TIMESTEPS - 1, color="red", label="Basic strategy")
    plt.hlines(stand_win_rate, 0, GRAPH_NUM_TIMESTEPS - 1, color="orange", label="Stand only strategy")
    plt.legend()
    plt.savefig("output/win_loss_plot.png")

def test_random_policy():
    return (0.282, 0.676)  # Hardcoded
    env = Blackjack(6)
    p = Policy(ACTION_DIM, STATE_DIM, random=True)
    return test_policy(p, env, n=TEST_NUM_TRIALS)

def test_stand_policy():
    return (0.382, 0.566)  # Hardcoded
    env = Blackjack(6)
    p = Policy(ACTION_DIM, STATE_DIM, "output/Stand_policy.txt")
    return test_policy(p, env, n=TEST_NUM_TRIALS)

def test_optimal_policy():
    return (0.428, 0.478)  # Hardcoded
    env = Blackjack(6)
    p = Policy(ACTION_DIM, STATE_DIM, "output/Optimal_policy.txt")
    return test_policy(p, env, n=TEST_NUM_TRIALS)

if __name__ == "__main__":
    np.random.seed(100)
    if False:
        win_loss_rates = get_timesteps_vs_win_rate_data()
        #with open("tmp/win_loss_rates.txt", "r") as f:
        #    win_loss_rates = json.loads(f.read())
        with open("output/win_loss_rates.txt", "w") as f:
            json.dump(win_loss_rates, f)
        graph_timesteps_vs_win_rate(win_loss_rates)

    if True:
        n = 250
        s = Scheduler(ACTION_DIM, STATE_DIM, n)
        s.step_through()
        s.save_policy(filename="output/250_iterations_policy.txt")
        s.print_policy()

    if False:
        # Test hardcoded policy
        p = Policy(ACTION_DIM, STATE_DIM, "Policy.txt")
        env = Blackjack(6)
        win_rate, loss_rate = test_policy(p, env, n=1000)
        print("Win rate:", win_rate, "| Loss rate:", loss_rate)

