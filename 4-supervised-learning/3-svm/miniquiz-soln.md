## Miniquiz Solution

1. Given an infinite number of US coins (`coins = [1, 5, 10, 25]`) and an amount `value` in cents, what are minimum number of coins needed to make change for `value`? Write a function `find_change` that takes as input the coin denominations `coins`, and `value` as the amount in cents. Your function should return the minimum amount of coins necessary to make change for the specified value as an integer. 

### Example

```python
coins = [1, 5, 10, 25]

In [23]: find_change(coins, 100)

4 coins: [25, 25, 25, 25]

In [24]: find_change(coins, 74)

8 coins: [25, 25, 10, 10, 1, 1, 1, 1]
```

## Solution

```python
def find_change(coins, value):
   minCoins = [[0 for j in range(value + 1)]
               for i in range(len(coins))]
   minCoins[0] = range(value + 1)

   for i in range(1,len(coins)):
      for j in range(0, value + 1):
         if j < coins[i]:
            minCoins[i][j] = minCoins[i-1][j]
         else:
            minCoins[i][j] = min(minCoins[i-1][j],
             1 + minCoins[i][j-coins[i]])

   return minCoins[-1][-1]
```

### Discussion

How do we answer the question, “what is the smallest number of coins I can use to make exact change?” The method of picking the largest coins first (and only taking smaller coins when you need to) happens to give the optimal solution in many cases (U.S. coins are one example). However, with an unusual set of coins, this is not always true. For instance, the best way to make change for thirty cents if you have quarters, dimes, and pennies, would be to use three dimes, as opposed to a quarter and five pennies.

This example shows us that the so-called “greedy” solution to this problem doesn’t work. So let’s try a dynamic algorithm. As with Fibonacci numbers, we need to compute some sub-problems and build off of them to get an optimal solution for the whole problem.

In particular, denote our coin values `v_n` and the amount of change we need to make `n`. Let us further force `v_1 = 1`, so that there is guaranteed to be a solution. Our sub-problems are to find the minimal number of coins needed to make change if we only use the first `i` coin values, and we make change for `j` cents. We will eventually store these values in a two-dimensional array, and call it`minCoins[i,j]`. In other words, this is a two-dimensional dynamic program, because we have sub-problems that depend on two parameters.

The base cases are easy: how many coins do I need to make change for zero cents? Zero! And how many pennies do I need to make `j` cents? Exactly `j`. Now, suppose we know `minCoins[i-x,j]` and `minCoins[i,j-y]` for all relevant `x`, `y`. To determine `minCoins[i,j]`, we have two choices. First, we can use a coin of value `v_i`, and add 1 to the array entry `minCoins[i, j - v_i]`, which contains the best solution if we have `j - v_i` cents. Second, we can ignore the fact that we have coins of value `v_i` available, and we can use the solution `minCoins[i-1,j]`, which gives us the same amount of money either.

In words, the algorithm says we can suppose our last coin is the newly introduced coin, and if that doesn’t improve on our best solution yet (above, `minCoins[i,j-v_i]`) then we don’t use the new coin, and stick with our best solution whose last coin is not the new one `minCoins[i-1,j]`).

The first two lines of the function initialize the array with zeros, and fill the first row with the relevant penny values. The nested for loop runs down each row, updating the coin values as described above. The only difference is that we check to see if the coin value is too large to allow us to choose it (`j < coinValues[i]`, or equivalently `j – coinValues < 0`). In that case, we default to not using that coin, and there is no other way to go. Finally, at the end we return the last entry in the last row of the matrix.

