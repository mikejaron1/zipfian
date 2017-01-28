import nose.tools as n
from deck import Card, Deck
from blackjack_utils import value, cards_to_str
from blackjack_player_v2 import Dealer, HumanPlayer
import blackjack_v2 as blackjack


def test_value():
    cards = [Card('5', 'c'), Card('K', 'd')]
    n.assert_equal(value(cards), 15)

def test_ace_value1():
    cards = [Card('6', 'c'), Card('A', 'd')]
    n.assert_equal(value(cards), 17)

def test_ace_value1():
    cards = [Card('7', 'c'), Card('J', 's'), Card('A', 'd')]
    n.assert_equal(value(cards), 18)

def test_player_take_card():
    player = Player('player')
    player.take_card((Card('J', 's')))
    n.assert_equal(len(player.cards), 1)
    n.assert_equal(player.cards[0].number == 'J' and player.cards[0].suit == 's')
