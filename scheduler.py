import numpy as np
from blackjack import Blackjack
from q_learning import QLearning

ITERATIONS_IN_TIMESTEP = 100000
NUM_DECKS = 6

class Scheduler:
    """This class runs Q-learning on blackjack.
    """
    def __init__(self, action_dim, state_dim, n):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.iterations_in_step = ITERATIONS_IN_TIMESTEP

        self.n = n  # Number of iterations
        self.t = 1  # Timestep

        self.env = Blackjack(NUM_DECKS)
        self.learner = QLearning(action_dim, state_dim)

    def get_random_state(self):
        return int(np.random.uniform() * (self.state_dim - 1)) + 1

    def step(self):
        """Performs one timestep of playing the game and updating the learner. A timestep
        consists of self.iterations_in_step iterations.
        """
        state_action_reward_tuples = []
        for i in range(self.iterations_in_step):  # Play self.iterations_in_step hands
            # Initialize a random state (player hand and dealer card)
            s = self.get_random_state()
            if s == 0: continue  # State 0 is reserved for the termination state
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
                    states.append(0)  # Add terminating state to states
                    actions.append(0)
                    break
            reward = self.env.get_reward()

            for state, action in zip(states, actions):
                if state in states[-2:-1]:  # The last state that led to the terminating state
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
        """Run the step function until self.t equals self.n.
        """
        while self.t <= self.n:
            print(self.t)
            self.learner.update_epsilon(self.t, self.n)
            self.learner.update_alpha(self.t, self.n)
            self.step()
            self.t += 1

    def get_Q(self):
        return self.learner.get_Q()

    def save(self):
        self.learner.save_Q()

    def get_optimal_actions(self):
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
        return hard_optimal_actions, soft_optimal_actions

    def save_policy(self, filename="Policy.txt"):
        """Saves the policy used by the Q-learner.
        """
        hard_optimal_actions, soft_optimal_actions = self.get_optimal_actions()

        with open(filename, "w") as f:
            f.write("Hard totals\n")
            for total in range(4, 22):
                f.write(str(total) + ": " +\
                        str(list(hard_optimal_actions[total - 4].astype(int))) + "\n")
            f.write("Soft totals\n")
            for total in range(13, 22):
                f.write(str(total) + ": " +\
                        str(list(soft_optimal_actions[total - 13].astype(int))) + "\n")

    def print_policy(self):
        """Prints the policy used by the Q-learner.
        """
        hard_optimal_actions, soft_optimal_actions = self.get_optimal_actions()

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
