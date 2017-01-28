from blackjack_utils import value, cards_to_str

class Dealer(object):
    def hit(self, cards):
        return value(cards) <= 16


class Player(object):
    def __init__(self, money):
        self.name = raw_input("Enter your name: ")
        print "Welcome, %s!" % self.name
        self.money = money

    def bet(self):
        amount = 0
        while not (amount > 0 and amount <= self.money):
            amount = input("You have %d money. How much would you like to bet? " % self.money)
        return amount

    def win(self, amount):
        print "You win %d" % amount
        self.money += amount

    def lose(self, amount):
        print "You lose %d" % amount
        self.money -= amount

    def push(self):
        print "Push!"

    def hit(self, cards, dealer_card):
        print "You have %s" % cards_to_str(cards)
        print "The dealer has %s" % str(dealer_card)
        choice = None
        while choice not in ("hit", "stay", "h", "H", "s", "S"):
            choice = raw_input("What would you like to do? hit or stay? ")
        return choice in ("hit", "h", "H")



