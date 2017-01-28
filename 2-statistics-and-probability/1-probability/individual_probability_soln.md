##Solutions to Probability Exercises

##Probability Exercises

1. Suppose that two cards are drawn from a standard 52 card deck.
  * What's the probability that the first was a queen and the second was a king?
  ```
  The probability that the first card was a queen is 4/52. 
  Conditional on this, the probability that the second card is a king is 4/51.
  So the probability is (4/52)x(4/51) ~ 0.0060331825 ~ 0.06%.
  ```
  * What's the probability that both cards were queens?
  ```
  The probability that the second card was queen conditional on the first being a queen is (3/51).
  So the probability is (4/52)x(3/51) ~  0.00452488688 ~ 0.45%.
  ```
  * Suppose that before the second card was drawn, the first was inserted back into the deck and the deck reshuffled. What's the probability that both cards are queens?
  ```
  Now the probabilities of the first and second cards being queens are each 4/52.
  So the probability is (4/52)x(4/52) ~ 0.00591715976 ~ 0.59%.
  ```

2. A Store Manager wants to understand how his customers use different payment methods, and suspects that the size of the purchace is a major deciding factor. He organizes the table below.

   |           | Cash | Debit | Credit |
   |-----------|:----:|------:|--------|
   | Under $20 |  400 |   150 | 150    |
   | $20 - $50 |  200 |  1200 | 800    |
   | Over $50  |  100 |   600 | 1400   |

   * Given that a customer spent over $50, what's the probability that the customer used a credit card?
   ```
   	There were 100 + 600 + 1400 = 2100 purchases over $50. 
	A credit card was used for 1400 of them.
	So the probability that a customer used a credit card for a purchase over $50 was: 
	1400/2100 = 2/3 ~ 0.666 = 66.6%.
   ```
   * Given that a customer paid in cash, what's the probability that the customer spent less than $20?
   ```
      There were 400 + 200 + 100 = 700 purchases made in cash. 
      400 of them were purchases of less than $20. 
      So the probability that a cash purchase was < $20 is:
      400/700 ~ 0.571 ~ 57.1%.
   ```
   * What's the probability that a customer spent under $20 using cash?
   ```
   The total number of purchases is the sum of all entries in the table = 5000.
   400 of them were cash purchases under $20. 
   So the probability is 400/5000 = 0.8%.
   ```
3. A gSchool grad is looking for her first job! Given that she is freaked out, her chances of not getting an offer are 70%. Given that she isn't freaked out, her chances of not getting an offer are 30%. Suppose that the probability that she's freaked out is 80%. What's the probability that she gets an offer?
	```
	The probability that she has no offer is 
	
	P(freaked_out) x P(no offer| freaked out) + 
	
	P(not freaked out))x(no offer| not freaked out) = 
	
	(0.8)x(0.7) + (1 - 0.8)x(0.3) = 0.62
	
	So the probability of no offer is 1 - 0.62 = 0.38 = 38%.
```

4. Google decides to do random drug tests for heroin on their employees.
   They know that 3% of their population uses heroin. The drug test has the
   following accuracy: The test correctly identifies 95% of the
   heroin users (sensitivity) and 90% of the non-users (specificity).

   |                | Uses heroin | Doesn't use heroin |
   | -------------- | ----------: | -----------------: |
   | Tests positive |        0.95 |               0.10 |
   | Tests negative |        0.05 |               0.90 |

   Alice gets tested and the test comes back positive. What is the probability
   that she uses heroin?
      ```
   Use Bayes' Theorem:

    P(uses|tests positive) = P(uses) * P(tests positive|uses) / P(tests positive)

   Calculate each part:

    P(uses) = 0.03

    P(tests positive|uses) = 0.95

    P(tests positive)
    = P(uses) * P(tests positive|uses) + P(doesn't use) * P(tests positive|doesn't use)
    = 0.03 * 0.95 + 0.97 * 0.10 = 0.1255


    P(uses|tests positive) = 0.03 * 0.95 / 0.1255 = 0.227

    23% of people who test positive are users.
    ```

   

5. The Birthday Problem. Suppose there are 23 people in a data science class, lined up in a single file line. 
Let A_i be the probability that the i'th person doesn't have the same birthday as the j'th person for any j < i.
Use the chain rule from probability to calculate the probability that none of the 23 people have the same birthday.
    ```
    P(A_1) = 365/365
    P(A_2|A_1) = (365 - 1)/365
    P(A_3|A_1,A_2) = (365 - 2)/365
    ...

    By the chain rule: 
        P(none have same birthday) =
        P(A_1)*P(A_2|A_1)*P(A_3|A_1,A_2)*...*P(A_23|A_1, ..., A_22) = 
	    (365/365)*(364/365)*(363/365)(362/365)...(343/365) = 49.3%
	
	P(at least two people have same birthday) = 1 - P(none have same birthday) = 50.7%
        

