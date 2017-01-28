### Probability 
1.
The bias of a coin is 0.6. What is the probability of flipping 
8 or more heads in 10 flips?

``` python 

import scipy.stats as scs

rv = scs.binom(10, 0.6)

rv.pmf(8) + rv.pmf(9) + rv.pmf(10)

0.1672897536 
```
2.
You've found a secret admirer note on your desk, and don't know who it might've 
come from but you know it must've been one of your  three office mates:  Jack, John, or Jimmy.  
As of yesterday, you thought it was twice as likely that Jimmy had a crush on you than John,
and that John and Jack were equally likely to have a crush on you.  
However even if Jimmy liked you, you think there'd only be 5% he'd leave you a note.
On the other hand, if Jack liked you there'd be a whopping 50% chance he'd leave you a note.
and if John liked you, there'd be a 20% chance he'd leave you a note. 

What's the probability that the note came from John?

`J1 = {Jack likes you}, J2 = {John likes you}, J3 = {Jimmy likes you}`

Calculate their probabilities.

`P(J1) = P(J2), P(J3) = 2 * P(J1), P(J1) + P(J2) + P(J3) = 1`

`implies P(J1) = P(J2) = 0.25, P(J3) = 0.50`

`N = {Note is left}`

Find probability of ever getting a note

`P(N|J1) = 0.50, P(N|J2) = 0.20, P(N|J3) = 0.05.`

`P(N) = P(N|J1) * P(J1) + P(N|J2) * P(J2) + P(N|J3) * P(J3)`

     = 0.05 * 0.25 + 0.20 * 0.25 + 0.05 * 0.50
     
     = 0.20

Use Bayes' theorem

`P(J2|N) = P(N|J2) * P(J2) / P(N) `
        `= 0.125 / 0.20 = 0.25`



### Statistics 

Below are the total number of log-ins for 20 different randomly selected users from 2014:

     logins = [10, 25, 12, 35, 14, 18, 16, 15, 22, 10, 9, 11, 49, 20, 15, 9, 18, 19, 20, 20]
3.
What is the sample mean?
4.
What is the sample variance?

```python 
def calc_mean_var(v):
     return np.mean(v), np.var(v) * len(v) / (len(v) - 1) 
```

`mean = 18.35`
`variance = 91.502631578947359`

5.
If we randomly select another user from 2014, what is the probability that he/she 
has more than 15 log-ins?
```
mu = 18.35; std = sqrt(91.5) = 9.57
assumption that logins are normal
```
```
P(X > 15) = P(Z > (15-18.35)/9.57) 
          = P(Z > -0.35) 
          = 1 - P(Z < -0.35) 
          = 1 - 0.3631 
          = 0.6369
          
1-norm.cdf(15, loc=18.35, scale=9.57) = 0.6369
```
6.
Sales targets weren't met last year.  The sales department asserts that on average, there 
were only 10 log-ins per user, however the web team thinks there were more.  Set up a 
frequentist hypothesis test and compute a p-value based on your data.  

Use One-sided t-test comparison of means

Null Hypothesis: average log-ins per user = 10

Alternative Hypothesis: average log-ins per user > 10

Assume a normal distribution to the log-ins. Let mu = 18.35, 
sigma = np.sqrt(var) ~ 9.5657. Since len(logins) = 20, 

`t = (mu - 10) / (sigma / np.sqrt(20)) ~ - 3.90377412477`

`p_value = P(T < t) ~ 0.000477`

`With 95% confidence, and p_value < 0.05, we reject the Null Hypothesis.`

7.
A major charity organization is interested to see if changing the mission statement 
on the website increases donations. As a pilot study, they randomly show 1 of 10 newly 
drawn up mission statements to each user that comes to the site.  As such, you set up 
10 separate hypothesis tests, each testing whether or not there was an increase in donations.
What adjustments would you make to account for the 10 tests? 

```
You would divide the overall alpha by 10 and compare the individual test p-values
to the corrected level of significance.
```

###  Modeling 

8.
Generally, when we increase the flexiblity or complexity of the model, what happens to bias?  
What about variance? What do we expect to happen to the training error?  What about the test error?

```
As flexibility and complexity go up, variance increases and bias decreases.
Furthermore, training error decreases and test error will initially decrease,
but eventually will increase.
```

9.
You have two models:
Model 1:   Salary ~ Occupation + Height + Gender
Model 2:   Salary ~ Occupation + Region + Height + Gender

Name 2 appropriate ways to compare these two models.
Name 1 inappropriate way to compare these two models.  

```
Appropriate comparisons:
Cross-validation, K-fold cross-validation, AIC, BIC, Mallow's Cp, 
Adjusted-R^2, F-test

Few inappropriate ways to compare:
RSS, MSE, R^2

or any such comparison that doesn't penalize for complexity.
```
