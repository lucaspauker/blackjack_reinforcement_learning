import numpy as np
from blackjack import Blackjack

class Policy:
    """This class is useful for implementing hardcoded policies. Note that the
    Blackjack class is used to translate a file into the blackjack policy. A random
    policy can also be created.
    """
    def __init__(self, action_dim, state_dim, policy_file=None, random=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.policy_file = policy_file
        self.random = random
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
        if self.random:
            return int(np.random.random() * self.action_dim)
        return self.policy[state_num]

