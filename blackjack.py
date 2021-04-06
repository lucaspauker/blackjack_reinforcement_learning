import numpy as np

deck_base = list(range(1, 10)) + [10, 10, 10, 10]
k52_DECK = []
for _ in range(4): k52_DECK += deck_base

class Blackjack:
    """Implementation of blackjack where the dealer hits soft 17.
    """
    def __init__(self, num_decks, player_hand=None, hole_card=-1):
        self.num_decks = num_decks
        self.deck = []
        for _ in range(self.num_decks): self.deck += k52_DECK
        self.deck = sorted(self.deck)
        self.deck_pointer = 0

        if player_hand is None:
            player_hand = []
        self.player_hand = player_hand
        self.hole_card = hole_card
        self.dealer_hand = []

        self.player_turn_over = False

    def get_random_hand(self, value):
        """Gets a random two card hand. Picks a random two cards given the inputted
        card sum value.
        """
        hand = []
        first_cards = list(range(max(2, value - 11), min(11, value - 2) + 1))
        hand.append(np.random.choice(first_cards))
        hand.append(value - hand[0])
        assert(sum(hand) == value)
        for i in range(len(hand)):
            if hand[i] == 11: hand[i] = 1
            assert(hand[i] >= 1 and hand[i] <= 10)
        return hand

    def set_initial_state(self, state):
        """Sets the state (player hand and dealer hole card) given a numerical state
        input.
        """
        if state == 0: return  # Special case

        hole_card = (state - 1) % 10 + 1
        soft = ((state - hole_card) % 20) / 10
        value = int((state - hole_card - soft * 10) / (2 * 10) + 4)
        assert(Blackjack.get_state_number(value, soft, hole_card) == state)

        hand = self.get_random_hand(value)
        # Soft hands are under-represented
        self.player_hand = hand
        self.hole_card = hole_card

    def hand_value(self, hand):
        """Returns the value of the blackjack hand
        """
        value = sum(hand)
        soft = False
        num_as = 0
        for card in hand:
            if card == 1:
                num_as += 1
                value += 10
        for _ in range(num_as):
            if value > 21:
                value -= 10
                soft = False
            else: soft = True
        return value, soft

    @staticmethod
    def get_state_number(value, soft, hole_card):
        """State number 0 is the termination state. Note that:
           - value is in range [4, 21] --> 18 possibilities
           - soft is True or False --> 2 possibilites
           - hole card is in range [1, 10] --> 10 possibilities
        """
        return (value - 4) * (2 * 10) + (int(soft)) * 10 + (hole_card - 1) + 1

    def get_state(self):
        """Gets a representation of the state of value, soft boolean, and hole card.
        """
        if len(self.player_hand) == 0 or len(self.dealer_hand) == 0:
            return -1
        value, soft = self.hand_value(self.player_hand)
        return value, soft, self.dealer_hand[0]

    def shuffle_deck(self):
        np.random.shuffle(self.deck)
        self.player_turn_over = False
        self.deck_pointer = 0
        if self.player_hand != [] and self.hole_card != -1:
            for card in self.player_hand:
                self.deck.remove(card)
            self.deck.remove(self.hole_card)
            self.deck = self.player_hand + [self.hole_card] + self.deck
        else:
            self.player_hand = self.deck[0:2]
        self.dealer_hand = self.deck[2:4]
        self.deck_pointer += 4

    def player_turn_done(self):
        return self.player_turn_over

    def take_action(self, a):
        """There are two actions supported: staying and hitting (0, 1) respectively.
        """
        if a == 0:
            self.player_turn_over = True
        elif a == 1:
            self.player_hand.append(self.deck[self.deck_pointer])
            self.deck_pointer += 1
            if self.hand_value(self.player_hand)[0] > 21:
                self.player_turn_over = True
        else:
            pass

        if self.player_turn_over: self.dealer_turn()

    def dealer_turn(self):
        while True:
            value, soft = self.hand_value(self.dealer_hand)
            if soft and value == 17:  # Dealer hits on soft 17s
                self.dealer_hand.append(self.deck[self.deck_pointer])
                self.deck_pointer += 1
            elif value <= 16:
                self.dealer_hand.append(self.deck[self.deck_pointer])
                self.deck_pointer += 1
            else:
                break

    def get_reward(self):
        """Get the reward once the game is over, indicating who won the game.
        """
        assert(self.player_turn_over)
        if self.hand_value(self.player_hand)[0] > 21:
            return -10
        elif self.hand_value(self.player_hand)[0] == self.hand_value(self.dealer_hand)[0]:
            return 0
        elif self.hand_value(self.dealer_hand)[0] > 21\
                or self.hand_value(self.player_hand)[0] > self.hand_value(self.dealer_hand)[0]:
            return 10
        else:
            return -10

def play_blackjack():
    """Play blackjack with user input.
    """
    num_decks = 6
    env = Blackjack(num_decks)
    env.shuffle_deck()
    state = env.get_state()
    print("Hole card:", env.dealer_hand[0])
    print("Your hand: {0}; hand total = {1}".format(env.player_hand, env.hand_value(env.player_hand)[0]))
    while True:
        action = input("0 to stay, 1 to hit\n > ").strip()
        if action != "0" and action != "1":
            print("Invalid action")
            continue
        env.take_action(int(action))
        print("Your hand: {0}; hand total = {1}".format(env.player_hand, env.hand_value(env.player_hand)[0]))
        if env.player_turn_done():
            break
    if env.hand_value(env.player_hand)[0] > 21: print("You busted")
    print("Dealer hand: {0}; hand total = {1}".format(env.dealer_hand, env.hand_value(env.dealer_hand)[0]))
    if env.hand_value(env.dealer_hand)[0] > 21: print("Dealer busted")
    reward = env.get_reward()
    if reward > 0:
        print("You win!")
    elif reward < 0:
        print("You lose :(")
    else:
        print("Tie game.")

if __name__ == "__main__":
    if True:
        play_blackjack()

    if False:
        num_decks = 6
        env = Blackjack(num_decks, [1, 1], 1)
        env.shuffle_deck()
        state = env.get_state()
        while True:
            env.take_action(0)
            state = env.get_state()
            state_num = Blackjack.get_state_number(state[0], state[1], state[2])
            if env.player_turn_done():
                break
        reward = env.get_reward()

