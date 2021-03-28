import numpy as np
from blackjack import Blackjack

class QLearning:
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.Q = np.zeros((self.state_dim, self.action_dim))
        self.N = np.zeros((self.state_dim, self.action_dim))

        self.alpha = 0.01  # Learning rate
        self.epsilon = 1  # For e-greedy policy
        self.gamma = 1

    def update_Q(self, state, action, reward, next_state):
        if next_state == 0:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
        else:
            max_Q = self.Q[next_state][0]
            for a in range(self.action_dim):
                if self.Q[next_state][a] > max_Q:
                    max_Q = self.Q[next_state][a]
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max_Q - self.Q[state][action])
        self.N[state][action] += 1

    def get_Q(self):
        return self.Q

    def save_Q(self, filename="Q_values.txt"):
        np.savetxt(filename, self.Q)

    def load_Q(self, filename="Q_values.txt"):
        self.Q = np.load(filename)

    def get_N(self):
        return self.N

    def update_alpha(self, t, n):
        self.alpha = 1 / t
        #pass

    def update_epsilon(self, t, n):
        #if t < n / 2:
        #    self.epsilon = 1
        #else:
        #    self.epsilon = (n - t) / n
        self.epsilon = 1 / t
        #self.epsilon = 0.1
        #epsilon = 1

    def get_optimal_action(self, state_num):
        max_Q = self.Q[state_num][0]
        arg_max = 0
        for a in range(self.action_dim):
            if self.Q[state_num][a] > max_Q:
                max_Q = self.Q[state_num][a]
                arg_max = a
        return arg_max

    def policy(self, state_num):
        if np.random.uniform() < 1 - self.epsilon:
            return self.get_optimal_action(state_num)
        return int(np.random.uniform() * self.action_dim)

class Policy:
    def __init__(self, action_dim, state_dim, policy_file=None):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.policy_file = policy_file
        self.policy = np.zeros(state_dim)
        if self.policy_file:
            self.load_blackjack_policy()

    def load_blackjack_policy(self):
        with open(self.policy_file, "r") as f:
            soft = False
            total = 0
            for line in f:
                if "hard" in line.lower():
                    soft = False
                    continue
                elif "soft" in line.lower():
                    soft = True
                    continue
                total = int(line.split(":")[0].strip())
                arr = line.split("[")[1].split("]")[0].strip()
                actions = arr.split(",")
                actions = [int(a.strip()) for a in actions]
                states = []
                for i in range(len(actions)):
                    action = actions[i]
                    state = Blackjack.get_state_number(total, soft, i + 1)
                    self.policy[state] = action

    def get_action(self, state_num):
        return self.policy[state_num]

class Scheduler:
    def __init__(self, action_dim, state_dim, n):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.iterations_in_step = 100000

        self.n = n  # Number of iterations
        self.t = 1  # Timestep

        num_decks = 6
        self.env = Blackjack(num_decks)
        self.learner = QLearning(action_dim, state_dim)

    def get_random_state(self):
        return int(np.random.uniform() * (self.state_dim - 1)) + 1

    def step(self):
        state_action_reward_tuples = []
        for i in range(self.iterations_in_step):
            s = self.get_random_state()
            if s == 0: continue
            self.env.set_initial_state(s)
            self.env.shuffle_deck()

            # Play hand
            states = []
            actions = []
            while True:
                total, soft, hole = self.env.get_state()
                state = Blackjack.get_state_number(total, soft, hole)
                states.append(state)
                action = self.learner.policy(state)
                actions.append(action)
                self.env.take_action(action)

                if self.env.player_turn_done():
                    states.append(0)
                    actions.append(-1)
                    break
            reward = self.env.get_reward()

            for state, action in zip(states, actions):
                if state in states[-2:]:
                    state_action_reward_tuples.append((state, action, reward))
                else:
                    state_action_reward_tuples.append((state, action, 0))

        for i in range(len(state_action_reward_tuples)):
            tup = state_action_reward_tuples[i]
            if tup[0] == 0: continue
            next_state = 0
            if i != len(state_action_reward_tuples) - 1:
                next_state = state_action_reward_tuples[i + 1][0]
            self.learner.update_Q(tup[0], tup[1], tup[2], next_state)  # s, a, r, s'

    def step_through(self):
        while self.t <= self.n:
            self.learner.update_epsilon(self.t, self.n)
            self.learner.update_alpha(self.t, self.n)
            self.step()
            self.t += 1
            if self.t % 100 == 0 or self.t == 2:
                self.print_policy()
                print("t =", self.t)

    def get_Q(self):
        return self.learner.get_Q()

    def save(self):
        self.learner.save_Q()

    def save_policy(self, filename="Policy.txt"):
        hard_optimal_actions = np.zeros((18, 10))
        soft_optimal_actions = np.zeros((9, 10))
        N = self.learner.get_N()
        Q = self.learner.get_Q()
        for hole_card in range(1, 11):
            for total in range(4, 22):
                # Hard total
                state_num = Blackjack.get_state_number(total, False, hole_card)
                optimal_action = self.learner.get_optimal_action(state_num)
                hard_optimal_actions[total - 4][hole_card - 1] = optimal_action

                # Soft total
                if total >= 13:
                    state_num = Blackjack.get_state_number(total, True, hole_card)
                    optimal_action = self.learner.get_optimal_action(state_num)
                    soft_optimal_actions[total - 13][hole_card - 1] = optimal_action

        with open(filename, "w") as f:
            f.write("Hard totals\n")
            for total in range(4, 22):
                f.write(str(total) + ": " + str(list(hard_optimal_actions[total - 4].astype(int))) + "\n")
            f.write("Soft totals\n")
            for total in range(13, 22):
                f.write(str(total) + ": " + str(list(soft_optimal_actions[total - 13].astype(int))) + "\n")

    def print_policy(self):
        hard_optimal_actions = np.zeros((18, 10))
        soft_optimal_actions = np.zeros((9, 10))
        N = self.learner.get_N()
        Q = self.learner.get_Q()
        for hole_card in range(1, 11):
            for total in range(4, 22):
                # Hard total
                state_num = Blackjack.get_state_number(total, False, hole_card)
                optimal_action = self.learner.get_optimal_action(state_num)
                hard_optimal_actions[total - 4][hole_card - 1] = optimal_action
                print(total, hole_card)
                for i in range(self.action_dim):
                    print(N[state_num][i])
                print(Q[state_num])
                print()

                # Soft total
                if total >= 13:
                    state_num = Blackjack.get_state_number(total, True, hole_card)
                    optimal_action = self.learner.get_optimal_action(state_num)
                    soft_optimal_actions[total - 13][hole_card - 1] = optimal_action

        print("-" * 50)
        print("HARD TOTALS")
        print("-" * 50)
        print("Dealer upcard")
        print(" ", list(range(1, 11)))
        for total in range(4, 22):
            print(total, hard_optimal_actions[total - 4])

        print()
        print("-" * 50)
        print("SOFT TOTALS")
        print("-" * 50)
        print("Dealer upcard")
        print(" ", list(range(1, 11)))
        for total in range(13, 22):
            print(total, soft_optimal_actions[total - 13])

if __name__ == "__main__":
    np.random.seed(100)
    action_dim = 2
    state_dim = 18 * 2 * 10 + 1

    if True:
        n = 400
        s = Scheduler(action_dim, state_dim, n)
        s.step_through()
        s.save_policy()
        s.print_policy()

    def test_policy(p, env, n=10000):
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
        print("Win rate:", wins / rounds, "| Loss rate:", losses / rounds)

    if True:
        # Test hardcoded policy
        p = Policy(action_dim, state_dim, "Policy.txt")
        env = Blackjack(6)
        test_policy(p, env, n=10000)

