#Part 1: Experimental Design
(**Estimated Time: 30 mins**)

Include your answers in ``afternoon_answers.md``.

1. You are a data scientist at Etsy, a peer-to-peer e-commerce platform. The marketing department has been
  organizing meetups among its sellers in various markets, and you task is to determine if attending these meetups
  causes sellers to generate more revenue.



  - Your boss suggests comparing the past sales from attendees to sales from sellers who did not attend a meetup. Will this work?

  ```
  - More competent sellers (higher sales) might just be more inclined to go
    to local meetups, so the higher sales cannot be attributed to going to
    local meetups alone
  ```
  - What's the statistical term for the problem with your boss's suggestion?

  ```
  - Selection bias and confounding
  (See the difference between selection bias and confounding here: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2982715/)
  ```

  - Outline the steps to design an experiment that would allow us to
determine if local meetups _**cause**_ a seller to sell more?

  ```
  - Randomly assign half of the sellers to go to local meetups (perhaps through 
    small $ incentives) and half of the sellers not to for a period of time, 
    say 3 months.
  - Record the sales of the sellers who attend local meetup groups vs. those who don't.
  - Compare the means of the sales between the 2 groups using a two-sample
    t-test for independent samples
  - Determine if the group that goes to local meetups has greater sales
    at a predefined statistical significance level
  ```

2. Suppose you are tasked with designing a book recommendation system for Amazon. How would you design an experiment to measure the effectiveness of your recommender system?

  ```
  Run an A/B test with users randomly split into two groups, one of which receives your new recommender system
  (treatment group) and another which receives either your previous recommendation system or possible a randomly
  chosen book. The choice of your control depends on the specific objective, but often you will compare your old
  systems with the new.
  ```

#Part 2: A / B Testing

Include your answers in ``afternoon_answers.md``.

Include your code in ``ab_test.py``.

Designers at Etsy have created a **new landing page** in an attempt to
improve sign-up rate for local meetups.

The historic sign-up rate for the **old landing page** is 10%.
An improvement to only 10.1% would provide a lift of 1%.
If statistically significant, this would be considered a success.
The product manager will not consider implementing the new page if
the lift is not more or equal to 1%.

Your task is to determine if the new_landing page can provide a 1% or more
lift to sign-up rate.

<br>

```python
# imports into script
import pandas as pd
from z_test import z_test
from power_functions import ProportionPower
```

1. Design an experiment **to collect data** in order to decide if the new page
   has a 0.1% or greater sign-up rate than the old page? (**_3 bullet points_**)

   (**Estimated Time: 10 mins**)

    ```
    - Randomly divert 50% of incoming users to the new page. The rest 50%
      will be directed to the old page.
    - Record the page the user landed on and whether he/she signed up
    - Only allow each user to be recorded once in the whole experiment to
      to ensure the observations are independent
    ```

2. State your null hypothesis and alternative hypothesis?

   (**Estimated Time: 5 mins**)

    ```
    - H0: new_conversion - old_conversion = 0.001
    - H1: new_conversion - old_conversion > 0.001
    ```

3. You ran a pilot experiment according to ``Question 1`` for ~1 day. The
   collected data is in ``experiment.csv``. Import the data into a pandas
   dataframe.

   **Hint: Write a function to check consistency between ab and
   landing_page columns and duplicate rows.**

   (**Estimated Time: 20 mins**)

   ```python
    # Read in the csv into a pandas dataframe
    data = pd.read_csv('data/experiment.csv')

    # Discover the old/new landing page label does not always match
    # the control/treatment label
    print 'ab column counts:'
    print data['ab'].value_counts()
    print 'landing_page column counts:'
    print data['landing_page'].value_counts()

    # Find out the rows where the landing page / control-treatment
    # labels are mismatched
    def find_mismatch(ab_cell, landing_page_cell):
        if ab_cell == 'treatment' and landing_page_cell == 'new_page':
            return 0
        elif ab_cell == 'control' and landing_page_cell == 'old_page':
            return 0
        else:
            return 1

    # Function that will be applied to the 2 columns
    func = lambda row: find_mismatch(row['ab'], row['landing_page'])

    # Define a mismatch column where 0 is ok, and 1 is mismatched
    print 'Dropping treatment / control and landing page mismatch...'
    # axis=1 means iterate the rows in the dataframe
    data['mismatch'] = data.apply(func, axis=1)

    # Get the percent mismatched
    mismatched = data[data['mismatch'] == 1]
    percent = (len(mismatched) / (len(data['mismatch']) * 1.) * 100)
    print 'Percentage mismatched:', percent

    # Dropping rows that are mismatched
    data = data[data['mismatch'] == 0]
   ```

4. State a rationale for using a one-tailed z-test or a two-tailed z-test.
   Calculate a p-value for a 0.1% lift from using the new page compare to the
   old page. Use ``z_test()``  from ``z_test.py`` to execute the z-test.
   ``z_test.py`` is already in the repo.

   Based on the p-value alone, explain your decision to adopt the
   new page or not.

   ```python
    # Get the parameters needed for z_test()
    old = data[data['landing_page'] == 'old_page']
    new = data[data['landing_page'] == 'new_page']
    old_nrow = old.shape[0] * 1.
    new_nrow = new.shape[0] * 1.
    old_convert = old[old['converted'] == 1].shape[0]
    new_convert = new[new['converted'] == 1].shape[0]
    old_conversion = old_convert / old_nrow
    new_conversion = new_convert / new_nrow

    # These are the arguments that z_test() takes:
    # old_conversion, new_conversion, old_nrow, new_nrow,
    # effect_size=0., two_tailed=True, alpha=.05
    # The running z_test() will print out the results
    z_test(old_conversion, new_conversion,
           old_nrow, new_nrow,
           effect_size=0.001, two_tailed=False, alpha=.05)
  ```

  ```
   - One tailed z-test is used because we will only adopt the new page if the
     lift is larger than 0.1% with statistical significance. Otherwise, we keep the old page.
   - p-value: 0.244060255972. Based on the p-value alone, the lift is not
     significantly larger than 0.1%. The null is not rejected, and the new
     page is not adopted
  ```

5. Based on the pilot, calculate the minimum sample size required to achieve
   80% power at 0.05 significance level, assuming equal sample size in both
   old page and new page groups. Also calculate the power the pilot has.

   Import ``ProportionPower()`` from ``power_functions.py``. Create an
   instance of ``ProportionPower()`` using the appropriate parameters (read
   the doc in ``power_functions.py``).Then you can run the following functions
   in ``ProportionPower()``. ``power_functions.py`` is already in the
   repo.

   Use ``calc_min_sample()`` function to calculate the minimum sample required
   to get 80% power at 0.05 significance level. Subsequently, calculate the
   approximate time needed to run the experiment based on the number of users
   in the pilot.

   Use ``calc_power()`` to calculate the power of the pilot. Interpret what the
   power means. Is the power enough to draw any conclusions?

   (**Estimated Time: 20 mins**)

   ```python
    # Parameters for ProportionPower()
    # p_samc, p_samt, effect_size, alpha=.05, power=None,
    # total=None, two_tailed=True
    nrow = old_nrow + new_nrow
    p_power = ProportionPower(old_conversion, new_conversion,
                              effect_size=.001, alpha=.05, power=.8,
                              total=nrow, two_tailed=False)

    # Calculate minimum sample size
    print 'Minimum Sample: ', p_power.calc_min_sample()

    # Calculate power of the pilot
    print 'Power of the pilot:', p_power.calc_power()
   ```

   ```
   For 80% power at 0.05 significance level:
    - Minimum requred sample size: 1201311
    - Approx. required time: 8.2 days
    - Current power: 0.252200577093
    - Interpretation: There is only a 25.2% chance of correctly rejecting 
      the null (that the two landing pages have the same conversion rate), 
      when it is false.  
    - The pilot sample is underpowered.  That the p-value is high  
      doesn't necessarily mean we can't reject the null.  It could be that we
      just need to increase the sample size.  

   ```
   
   probability that it correctly rejects the null hypothesis (H0) when it is false

6. State why running a pilot experiment allows us to more accurately
   determine the minimum sample size required for a certain power.

   (**Estimated Time: 5 mins**)

  ```
   - Running a pilot provides an estimate of the sign-up rate for the new
     page, as well as the variability around that estimate.  
     Consequently, we are also more informed about the sample size
     needed to show statistical significance of the difference between the new and the old pages.
  ```
