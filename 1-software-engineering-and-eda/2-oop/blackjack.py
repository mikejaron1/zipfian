from deck import Deck
from blackjack_player import Dealer, Player
from blackjack_utils import value, is_bust, is_blackjack

class Blackjack(object):
    def __init__(self, amount):
        self.dealer = Dealer()
        self.player = Player(amount)
        self.deck = Deck()
        self.play()

    def play(self):
        while self.player.money > 0:
            self.play_once()

    def play_once(self):
        amount = self.player.bet()
        self.deck.shuffle()
        dealer_cards = []
        player_cards = []
        for i in xrange(2):
            dealer_cards.append(self.deck.draw_card())
            player_cards.append(self.deck.draw_card())
        if is_blackjack(player_cards):
            self.player.win(1.5 * amount)
            return

        while True:
            if value(player_cards) >= 21:
                break
            if not self.player.hit(player_cards, dealer_cards[1]):
                break
            player_cards.append(self.deck.draw_card())

        player_value = value(player_cards)
        if player_value >= 22:
            self.player.lose(amount)
            return

        while True:
            if value(dealer_cards) >= 21:
                break
            if not self.dealer.hit(dealer_cards):
                break
            dealer_cards.append(self.deck.draw_card())

        dealer_value = value(dealer_cards)
        if player_value < dealer_value:
            self.player.lose(amount)
        elif player_value > dealer_value:
            self.player.win(amount)
        else:
            self.player.push()


if __name__ == '__main__':
    blackjack = Blackjack(100)
    blackjack.play()

