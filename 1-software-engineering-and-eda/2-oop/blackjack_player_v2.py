from blackjack_utils import value, cards_to_str

class Player(object):
    def __init__(self, name):
        self.name = name
        self.cards = []

    def take_card(self, card):
        self.cards.append(card)

    def give_cards(self):
        cards = self.cards
        self.cards = []
        return cards

    def display(self):
        print "%s: %s" % (self.name, cards_to_str(self.cards))


class Dealer(Player):
    def __init__(self):
        self.shown = False
        return super(Dealer, self).__init__("dealer")

    def show(self):
        self.shown = True

    def hit(self):
        return value(self.cards) <= 16

    def shown_card(self):
        if self.cards:
            return self.cards[0]

    def display(self):
        if self.shown:
            print "%s: %s" % (self.name, cards_to_str(self.cards))
        else:
            print "%s: X %s" % (self.name, str(self.shown_card()))


class HumanPlayer(Player):
    def __init__(self, name, money):
        self.money = money
        self.bet = 0
        return super(HumanPlayer, self).__init__(name)

    def make_bet(self, amount):
        self.bet = amount

    def win(self):
        self.money += self.bet
        self.bet = 0

    def lose(self):
        self.money -= self.bet
        self.bet = 0

    def blackjack(self):
        self.money += 3 * self.bet / 2
        self.bet = 0

    def push(self):
        self.bet = 0

    def hit(self, dealer_card):
        choice = None
        while choice not in ("hit", "stay", "h", "H", "s", "S"):
            choice = raw_input("%s: hit or stay? " % self.name)
        return choice in ("hit", "h", "H")

    def display(self):
        print "%s (%d): %s" % (self.name, self.bet, cards_to_str(self.cards))



