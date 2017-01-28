from deck import Deck
from blackjack_player_v2 import Dealer, HumanPlayer
from blackjack_utils import value, is_bust, is_blackjack, cards_to_str

class Blackjack(object):
    def __init__(self, amount, num_players):
        self.dealer = Dealer()
        self.players = self.generate_players(amount, num_players)
        self.deck = Deck()
        self.dealer_card = None

    def display(self):
        print "--------------------"
        for player in [self.dealer] + self.players:
            player.display()
        print "--------------------"

    def generate_players(self, amount, num_players):
        players = []
        for i in xrange(1, num_players + 1):
            name = raw_input("Enter player %d's name: " % i)
            players.append(HumanPlayer(name, amount))
        return players

    def get_bet(self, player):
        if player.money == 0:
            print "You are broke! Goodbye."
            return False
        amount = 0
        while amount <= 0 or amount > player.money:
            question = "%s: You have %d money. How much would you like to bet? " \
                       % (player.name, player.money) + \
                       "Enter Q to exit the game. "
            amount_str = raw_input(question)
            if amount_str.isdigit():
                amount = int(amount_str)
            elif amount_str == 'Q':
                print "Goodbye, %s. You leave with %s." % (player.name, player.money)
                return False
        player.make_bet(amount)
        return True

    def get_all_bets(self):
        self.players = [player for player in self.players if self.get_bet(player)]
        return self.players != []

    def deal(self):
        all_players = self.players + [self.dealer]
        for player in all_players:
            self.deck.add_cards(player.give_cards())
        self.deck.shuffle()
        for i in xrange(2):
            for player in all_players:
                player.take_card(self.deck.draw_card())

    def play(self):
        while True:
            if not self.get_all_bets():
                print "Goodbye!"
                return

            self.deal()

            self.display()

            for player in self.players:
                self.player_play(player)

            self.dealer_play()
            self.score_all()

    def score_all(self):
        dealer_value = value(self.dealer.cards)
        for player in self.players:
            self.score(player, dealer_value)

    def bust(self, player):
        print "Sorry, %s, you bust!" % player.name
        player.lose()

    def blackjack(self, player):
        print "Congrats, %s! You got blackjack!" % player.name
        player.blackjack()

    def lose(self, player):
        print "You lose, %s." % player.name
        player.lose()

    def win(self, player):
        print "You win, %s!" % player.name
        player.win()

    def push(self, player):
        print "Push, %s" % player.name
        player.push()

    def player_play(self, player):
        if is_blackjack(player.cards):
            self.blackjack(player)
            return

        while value(player.cards) <= 21 and player.hit(self.dealer.shown_card()):
            player.take_card(self.deck.draw_card())
            self.display()

        if value(player.cards) > 21:
            self.bust(player)

    def dealer_play(self):
        self.dealer.show()
        self.display()
        while value(self.dealer.cards) < 21 and self.dealer.hit():
            self.dealer.take_card(self.deck.draw_card())
            self.display()

    def score(self, player, dealer_value):
        player_value = value(player.cards)
        if player.bet:
            if player_value > dealer_value:
                self.win(player)
            elif player_value < dealer_value:
                self.lose(player)
            else:
                self.push(player)


if __name__ == '__main__':
    num_players = 0
    while num_players <= 0 or num_players > 10:
        string = raw_input("How many players? (1-10) ")
        if string.isdigit():
            num_players = int(string)
    blackjack = Blackjack(100, num_players)
    blackjack.play()

