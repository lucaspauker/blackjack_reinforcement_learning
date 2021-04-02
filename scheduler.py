import numpy as np
from blackjack import Blackjack
from q_learning import QLearning

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

            # if self.t % 100 == 0 or self.t == 2:
            #     self.print_policy()
            #     print("t =", self.t)

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

                # print(total, hole_card)
                # for i in range(self.action_dim):
                #     print(N[state_num][i])
                # print(Q[state_num])
                # print()

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
