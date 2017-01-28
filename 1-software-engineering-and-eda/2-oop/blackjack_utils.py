def value(cards):
    total = 0
    ace = False
    for card in cards:
        if card.number.isdigit():
            total += int(card.number)
        if card.number in "JQK":
            total += 10
        else:
            total += 1
            ace = True
    if ace and total <= 11:
        return total + 10
    return total

def is_blackjack(cards):
    return len(cards) == 2 and value(cards) == 21

def is_bust(cards):
    return value(cards) > 21

def cards_to_str(cards):
    return " ".join(str(card) for card in cards)
