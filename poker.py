import random
import numpy as np
import matplotlib.pyplot as plt

# Suit doesnt matter for blackjack
NUM_DECKS = 2
deck_base = list(range(1, 14))
k52_DECK = []
for _ in range(4): k52_DECK += deck_base

BASE_DECK = []
for _ in range(NUM_DECKS):
    BASE_DECK += k52_DECK
BASE_DECK = sorted(BASE_DECK)

def shuffle(deck):
    new_deck = []
    deck_copy = deck.copy()
    while len(deck_copy) > 0:
        i = int(random.random() * len(deck_copy))
        new_deck.append(deck_copy[i])
        deck_copy = deck_copy[:i] + deck_copy[i + 1:]
    return new_deck

def convert_card_to_value(card):
    if card == 1:
        return 11
    elif card > 10:
        return 10
    else:
        return card

n = 10000
num_busts = 0
nonbust_hand_sums = []
for _ in range(n):
    # Shuffle the deck
    deck = shuffle(BASE_DECK)
    #for i in range(len(BASE_DECK)):
    #    print(deck[i], BASE_DECK[i])
    assert(sorted(deck) == BASE_DECK)

    # Draw two cards
    deck_pointer = 0
    card_hand = deck[:2]
    deck_pointer += 2
    # Replace cards with values
    dealer_hand = []
    for card in card_hand:
        dealer_hand.append(convert_card_to_value(card))
    SEPARATOR = " || "
    print("Dealer hand:", ", ".join([str(d) for d in dealer_hand]), SEPARATOR,
          "Sum: {}".format(sum(dealer_hand)))
    shown_card = dealer_hand[0]
    hand_value = sum(dealer_hand)

    # Dealer rules
    while True:
        if hand_value <= 16:
            # Hit
            deck_pointer += 1
            card_val = convert_card_to_value(deck[deck_pointer])
            dealer_hand.append(card_val)
            print("Hit", deck[deck_pointer], SEPARATOR,
                "Value =", card_val, SEPARATOR,
                "Sum: {}".format(sum(dealer_hand)))
            hand_value += card_val
        elif hand_value == 17:
            # Hit soft 17s
            if 11 in dealer_hand:
                print("SOFT 17")
                deck_pointer += 1
                card_val = convert_card_to_value(deck[deck_pointer])
                dealer_hand.append(card_val)
                hand_value += card_val

                if hand_value > 21:
                    dealer_hand.remove(11)
                    dealer_hand.append(1)
                    hand_value -= 10
                    assert(hand_value == sum(dealer_hand))

                print("Hit", deck[deck_pointer], SEPARATOR,
                    "Value =", card_val, SEPARATOR,
                    "Sum: {}".format(sum(dealer_hand)))
            else:
                nonbust_hand_sums.append(hand_value)
                break
        elif hand_value > 21:
            num_busts += 1
            print("BUST")
            break
        else:
            nonbust_hand_sums.append(hand_value)
            break
    print("------------------------------------------")

data = np.asarray(nonbust_hand_sums)
print(np.mean(data))
print("Runs: {}".format(n))
print("Busts: {}".format(num_busts))
print("Mean: {}".format(round(np.mean(data), 2)))
print("Median: {}".format(np.median(data)))
print("Std: {}".format(round(np.std(data), 2)))

freqs = {}
values = list(range(17, 22))
for elem in data:
    if elem in freqs:
        freqs[elem] += 1
    else:
        freqs[elem] = 1
freqs_list = np.array([freqs[v] for v in values])
freqs_list = freqs_list / sum(freqs_list)
plt.scatter(values, freqs_list, color="black", s=100)
plt.plot(values, freqs_list, color="navy")
#plt.hist(data, bins=5, align="mid")
plt.title("Distribution of Non Bust Dealer Hands")
plt.xticks(range(17, 22))
plt.savefig("nonbust_graph.jpg")
print("Image exported")

