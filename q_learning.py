import numpy as np
from blackjack import Blackjack

class QLearning:
    """Class that tracks and updates Q values in a table.
    """
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim

        # N is a matrix of counts of each (state, action) pair
        self.N = np.zeros((self.state_dim, self.action_dim))
        self.Q = np.zeros((self.state_dim, self.action_dim))

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

    def update_epsilon(self, t, n):
        self.epsilon = 1 / t

    def get_optimal_action(self, state_num):
        max_Q = self.Q[state_num][0]
        arg_max = 0
        for a in range(self.action_dim):
            if self.Q[state_num][a] > max_Q:
                max_Q = self.Q[state_num][a]
                arg_max = a
        return arg_max

    def policy(self, state_num):
        """Epsilon-greedy policy. Chooses a random action with probability epsilon
        and the highest Q-value action with probability 1 - epsilon
        """
        if np.random.uniform() < 1 - self.epsilon:
            return self.get_optimal_action(state_num)
        return int(np.random.uniform() * self.action_dim)

