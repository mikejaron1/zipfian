1. Consider *n* people who are attending a party. We assume that every person has
   an equal probability of being born on any day during the year, independent of
   everyone else, and ignore leap years (i.e. year has 365 days). What is the
   probability that each person has a distinct birthday? How many people are
   necessary to obtain a 99% probability of at least one birthday collision (i.e. 2 people
   with the same birthday).
   
   The probability of having distinct birthdays with *n* people is
   
     `p = (1) * (1 - 1 / 365) * (1 - 2 / 365) * ... * (1 - (n-1) / 365)`
   
   For 99% probability that at least two people have the same birthday, set `p = 0.01` 
   and solve for ceiling of *n*. We get `n = 57.`
   
     ```python
     # Calculate the complementary probability, all distinct birthdays
     n = 1.
     prob = 1.
     def find_prob(n, prob):
         if prob < 0.01 or n > 365:
             return n
         else:
             return find_prob(n + 1, prob * (1 - n / 365))
         
     print find_prob(n, prob)
     # 57
     ```

1. A hunter has two hunting dogs. One day, on the trail of some animal, the
   hunter comes to a place where the road diverges into two paths. He knows that
   each dog, independent of the other, will choose the correct path with
   probability `p`. The hunter decides to let each dog choose a path, and if they
   agree, he takes that one, and if they disagree, he randomly picks a path. Is his
   strategy better than just letting one of the two dogs decide on a path?
   Explain why or why not.

   Denote by `A, B` the events dog 1 and dog 2 choose the right path and `X`
   be the event the hunter chooses the right path. That is,
   `P(A) = P(B) = p` and we want to find `P(X)`. Compute all four probabilities
   
   `P(A and B) = p ^ 2`
   
   `P(A and B') = P(A' and B) = p * (1 - p)`
   
   `P(A' and B') = (1 - p) ^ 2`
     
   where `'` is the complement. Computing the hunter's conditional probabilities
   
   `P(X | A and B) = 1`
   
   `P(X | A and B') =  0.5`
   
   `P(X | A' and B) =  0.5`
   
   `P(X | A' and B') = 0`
   
   `P(X) = 1 * p ^ 2 + 0.5 * p * (1 - p) + 0.5 * p * (1 - p) + 0 * (1 - p) ^ 2`
   
   `     = p ^ 2 + p - p ^ 2 = p`
   
   Therefore, the strategy is identical to letting one dog decide.
