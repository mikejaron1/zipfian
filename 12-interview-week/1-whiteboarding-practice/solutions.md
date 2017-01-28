### 1. Stock Trading

*You have an array that contains the stock prices for each day over the past year. You would like to know the maximum money you could have made with one purchase and one sale. We are assuming that the price of stock is constant over each day.*

Here's the optimal solution, which is O(n) (linear time).

```python
def stock_trade(prices):
    min_price = None
    max_gain = 0
    for price in prices:
        if not min_price or price < min_price:
            min_price = price
        elif price - min_price > max_gain:
            max_gain = price - min_price
    return max_gain
```

Here's the solution you may have come up with, which is O(n^2) time (quadratic time). Much slower! This tries every possible pair of days for buying and selling.

```python
def stock_trade_inefficient(prices):
    max_gain = 0
    for i, p1 in enumerate(prices):
        for p2 in prices[i + 1:]:
            gain = p2 - p1
            if gain > max_gain:
                max_gain = gain
    return max_gain
```

*Now assume you have these rules: Each day you can buy at most 1 stock. You can sell any amount of your stock at any time. What is the maximum amount of money you could have made over the year?*

Basically, on each day, you should either buy a stock or sell all your stock. It's easier to go from the end and find the maximum price to the end.

```python
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
```

### 2. SQL
*a. Write a query to get the employee with the second highest salary.*

There are definitely other ways to do this.

```sql
SELECT employee_id
FROM employees
WHERE salary = (SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees));
```


*b. Write a query to get the average salary for each department.*

```sql
SELECT department, AVG(salary)
FROM employees
GROUP BY department;
```


*c. Find the user who has the highest salary increase since their start date.*

```sql
SELECT employee_id, name
FROM employees
WHERE
    salary = (SELECT MAX(employee.salary - hired.salary)
              FROM employees JOIN hired
              ON employees.employee_id=hired.employee_id);
```

*d. What percent of current employees switched departments in under a year?*

```sql
SELECT COUNT(1)/(SELECT COUNT(1) FROM employees)
FROM employees e
JOIN hired h
ON e.employee_id=h.employee_id AND e.department<>h.department;
```


### 3. Even-Odd Split

*Given a list of integers, move all the odd numbers to the left and the even numbers to the right.*

Here is the most optimal solution, which is linear time and uses constant extra space.

```python
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
```

This solution is also linear time, but uses linear extra space.

```python
def even_odd_split_space_hog(lst):
    evens = []
    odds = []
    for item in lst:
        if item % 2 == 0:
            evens.append(item)
        else:
            odds.append(item)
    return odds + evens
```


### 4. Word Break

*Given an input string and a dictionary of words, segment the input string into two dictionary words if possible. For example, if the input string is "applepie" and dictionary contains a standard set of English words, then we would return the string "apple pie" as output.*

The runtime here is linear, since we try every possible word break and there are n-1 possible spots to break the string into two strings.

```python
def word_break_two_words(string, dictionary):
    for i in xrange(1, len(string)):
        first = string[:i]
        second = string[i:]
        if first in dictionary and second in dictionary:
            return first, second
```

*Can you make an algorithm that would work for any number words (rather than just 2)?*

This algorithm is not going to be super efficient. We just try every possible break. For recursive problems, rather going through step by step to get the runtime, it's easier to just determine all the things you try.

So we need to determine how many breaks there are. There are n-1 places to break the word and we could either choose each one or not (2 possiblities), so that's 2^(n-1), so we can just say 2^n (exponential).

```python
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
```
