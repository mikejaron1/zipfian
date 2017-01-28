from itertools import izip


def stock_trade(prices):
    min_price = None
    max_gain = 0
    for price in prices:
        if not min_price or price < min_price:
            min_price = price
        elif price - min_price > max_gain:
            max_gain = price - min_price
    return max_gain


def stock_trade_inefficient(prices):
    max_gain = 0
    for i, p1 in enumerate(prices):
        for p2 in prices[i + 1:]:
            gain = p2 - p1
            if gain > max_gain:
                max_gain = gain
    return max_gain


def stock_trade_new_rules(prices):
    max_price = 0
    should_sell_list = [False] * len(prices)
    for i in xrange(len(prices) - 1, -1, -1):
        price = prices[i]
        if price >= max_price:
            max_price = price
            should_sell_list[i] = True
    profit = 0
    stock = 0
    for should_sell, price in izip(should_sell_list, prices):
        if should_sell:
            profit += stock * price
            stock = 0
        else:
            profit -= price
            stock += 1
    return profit


def even_odd_split(lst):
    odd_index = 0
    even_index = len(lst) - 1
    while odd_index < even_index:
        if lst[odd_index] % 2 == 1:
            odd_index += 1
        elif lst[even_index] % 2 == 0:
            even_index -= 1
        else:
            lst[even_index], lst[odd_index] = lst[odd_index], lst[even_index]


def even_odd_split_space_hog(lst):
    evens = []
    odds = []
    for item in lst:
        if item % 2 == 0:
            evens.append(item)
        else:
            odds.append(item)
    return odds + evens


def word_break_two_words(string, dictionary):
    for i in xrange(1, len(string)):
        first = string[:i]
        second = string[i:]
        if first in dictionary and second in dictionary:
            return first, second


def word_break(string, dictionary):
    if string in dictionary:
        return [string]
    for i in xrange(1, len(string)):
        first = string[:i]
        second = string[i:]
        if first in dictionary:
            rest = word_break(second, dictionary)
            if rest:
                return [first] + rest
    return False

