##Part 0: Introduction to Power

Suppose you are interested in testing if on average a bottle of coke weighs 20.4 ounces. You have collected
simple random samples of 130 bottles of coke and weighed them.

1. Load the data in with `numpy.loadtxt('data/coke_weights.txt')`.

2. State your null and alternative hypothesis.

   ```
   H0: mu = 20.4
   H1: mu != 20.4
   ```

3. Compute the mean and standard error of the sample. State why you are able to apply the Central
   Limit Theorem here to approximate the sample distribution to be normal.
   
   ```python
   def sample_sd(arr):
    return np.sqrt(np.sum((arr - np.mean(arr)) ** 2) / (len(arr) - 1))

   def standard_error(arr):
       return sample_sd(arr) / np.sqrt(len(arr))
    
   m = coke_weights.mean()
   se = standard_error(coke_weights)
   print 'The sample mean is', m
   print 'The standard error is', se
  
   # The sample mean is 20.519861441
   # The standard error is 0.084319217426
   ```
   
   The CLT applies here because the sample size is large enough, where **the mean** of a large enough sample
   (> 30) would approximate to a normal distribution.

4. Use `scipy` to make a random variable of the sampling distribution. Plot the PDF over `+/- 4
   standard error`. Make another random variable of the sampling distribution formulated by the
   null hypothesis (null distribution). Use the standard error from the sample as an estimator
   for the standard error of the null distribution. Plot the PDF of the second random variable on the
   same plot.
   
   ```python
   def plot_dist(m ,se, c, label, plot=True):
    rv = sc.norm(m, se)
    x_range = np.linspace( m - (4 * se), m + (4 * se), 1000)
    y = rv.pdf(x_range)
    if plot:
        plt.plot(x_range, y, c=c, label=label)
        # Vertical line of the mean
        plt.axvline(x=m, c=c, linestyle='--')
    return rv
   ```
   
5. Plot vertical lines to indicate the bounds for rejecting the null hypothesis assuming a
   significance level of 0.05. Based on the bounds for rejecting the null, what conclusion
   can we draw based on the sample of 130 bottles of coke?
   
   ```python
   def plot_ci(m, se, c, plot=True, ci=0.95):
    z = sc.norm.ppf(ci + (1 - ci) / 2)
    lower_ci = m - z * se 
    upper_ci = m + z * se
    if plot:
        plt.axvline(x=lower_ci, c=c, alpha=.3, linestyle=':')
        plt.axvline(x=upper_ci, c=c, alpha=.3, linestyle=':')
    return lower_ci, upper_ci
   ```
 
   - Based on the 95% CI we conclude that the weight of a bottle of coke on average is not different 
     from 20.4, i.e. we cannot reject the null that the weight is 20.4

   ![image](images/power_plot.png)

6. Compute the probability of a false negative (Type II Error) by using the upper/lower confidence
   interval (**Use `cdf` on the sampling distribution rv). Explain what a false negative is in the
   context of this problem.

7. Compute the power of the test. Explain what power means in the context of this problem.

   
   ```python
   def calc_power(data, null_mean, ci=0.95):
    m = data.mean()
    se = standard_error(data)
    sample_rv = plot_dist(m, se, 'r', 'sample')
    population_rv = plot_dist(null_mean, se, 'b', 'null')

    # Plot vertical line of the 95% CI 
    lower_ci, upper_ci = plot_ci(null_mean, se, 'b', ci=ci)
    print lower_ci, upper_ci
    print 'mean', m
    type_2_error = sample_rv.cdf(upper_ci) * 100
    power = 100 - type_2_error
    print 'Probability of Type 2 Error is %.2f%%' % type_2_error
    print 'Power is %.2f%%' % power
    return power
    
   power = calc_power(coke_weights, 20.4)
   print power # 29.51
   ```

   - Power, in this context, means the ability to detect if the mean weight of a bottle of coke is different
     from 20.4 given the that the weight of a bottle of coke is indeed different from 20.4.

<br>

##Part 1: Factors that Influence Power of a Test


1. Write a function `explore_power` that includes all the steps in `Part 0`. The input will be the mean value under the
   null hypothesis (i.e. `20.4` ounce as we have specified above) and the output is the power.

   Assume now the null hypothesis is that a bottle of coke weights `20.2` ounces. Run  `explore_power()` with the new null
   hypothesis. Did the power increase or decrease? Explain your observation.

2. Make a plot of **effect size (x)** against **power (y)**. The effect size is the absolute difference between the value under
   the null hypothesis and the sample statistic (i.e. `effect size = 20.519 - 20.4` if the null is `20.4`).

   ![image](images/effect_size.png)

3. Without writing any code, explain why the standard error decreases as the sample size increases. Furthermore, extrapolate
   and explain the relationship between **sample size** and **power**. Verify your result by computing power on a larger
   dataset with 1000 data points (`numpy.loadtxt('data/coke_weights_1000.txt')`). Is the power higher or lower with a
   larger sample size given the effect size and significance level held constant?

4. How does the power change if the significance level is increased from `0.05` to `0.1`. Explain your observation in terms
   of the increase/decrease probability of false positive/false negative. Plot **significance level (x)**
   (over a range of `0.01 - 0.3`) against **power (y)**.

<br>

##Part 2: Power Calculations for A/B testing  

One common problem in A/B testing is to decide when to stop the experiment. Power calculations are very useful in determining what
required minimum sample size is necessary to reach a certain power (usually 80%) given an effect size and a significance level.
A powerful test would ensure we are able to detect differences in conversion the majority of the time given the difference
in fact exists. To gain insights about the effect size, a small-scale pilot experiment is usually launched. The minimum
sample size is computed. Subsequently, a full-scale experiment is run until the minimum sample size is reached.

<br>

Continuing from yesterday's [Etsy case study](https://github.com/zipfian/ab-testing/blob/master/pair.md), get the
conversion data for the new and old landing pages with the code below.

```python
data = pd.read_csv('data/experiment.csv')
old_data = data[data['landing_page'] == 'old_page']['converted']
new_data = data[data['landing_page'] == 'new_page']['converted']
```

<br>

Historically, the old page has a conversion of 10% and we wish to test if the new page provides a 0.1% increase
(1% lift) in conversion. Recall the null and alternative hypotheses below:

```
# Set X as a random variable which is the (new conversion - old conversion)
X ~ p_new - p_old

H0: X = 0.001
H1: X > 0.001
```

<br>

###Part 2.1: Computing Power for Pilot Sample

In this part, we are going to compute statistical power for the pilot experiment given the null hypothesis.

1. By CLT, we can approximate the sampling distribution of proportions (`p_new, p_old`) to be normal (since proportion is
   effectively a measure of mean). We can further assume the sampling distribution of `p_new - p_old` to be normal.

   Compute `p_new - p_old` and the standard error from the sample and define a normal distribution random variable. Plot
   the PDF of the random variable as you have done previously

   **Hint: Standard Error for difference of proportions**

   ![image](images/sd_prop.gif)


   - `p` is a weighted average of the `p1` and `p2`
   - `n1` is the number of subjects sampled from the first population
   - `n2` is the number of subjects sampled from the second population


2. Define another random variable for the null distribution and plot the PDF of the random variable. Add
   a vertical line on the plot to indicate the bound for rejecting the null hypothesis given a significance
   level of 5% (not shown in plot below).

   ![image](images/ab.png)

3. Compute the power of the test given the null hypothesis. If the result seems strange to you, move onto `5.`.

4. What problem do you spot here with the plot from `2.`? Is increasing sample size going to increase power?
   If the effect size in the pilot is indeed representative of the ground truth, will the test ever be statisitcally
   significant? Explain your answer and suggest what the next steps should be.

5. Assume after reviewing the data, Etsy decided the pilot is a plausible enough representation of the company's daily
   traffic. As a result, Esty decided on a two-tailed test instead, which is as follows:

   ```
   X ~ p_new - p_old

   H0: X = 0.001
   H1: X != 0.001
   ```

   **Recompute the power of the test**


###Part 2.2: Computing Minimum Sample Size

Assume Etsy is staying with the two-tailed test described in `Part 2.1: 5`. A decision then would have to be
made about how long the test is running.

The minimum sample size is calculated by following the exact same process with calculating power, except power is a given (80%)
and sample size is omitted

1. Write a function `calc_min_sample_size` that would take
   - 2 lists/arrays of data (i.e. new page converts and old page converts)
   - Significance Level (Default: 0.05)
   - One-tailed to two-tailed test
   - Effect Size
   - Power (Default: 0.8)

   And return the minimum sample size required (rounded up to the nearest whole number).
