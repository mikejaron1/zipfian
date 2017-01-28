from scipy.stats import norm, zscore
import matplotlib.pyplot as plt
import numpy as np
import math


# finding a Z-score using scipy
# Using an inverse survival function:
z_1 = abs(norm.isf([sig])) #one-tailed t test
z_2 = abs(norm.isf([sig/2])) #two-tailed t test

z_1 = stats.norm.ppf(sig) #one-tailed t test
z_2 = stats.norm.ppf(sig/2) #two-tailed t test


# Answering problem 1 (Xin and Melanie):
def sample_power_probtest(c, t, power, sig):
    z = norm.isf([sig/2]) #two-sided t test
    zp = -1 * norm.isf([power]) 
    d = (t-c)
    s = z * math.pow(c * (1-c), .5) + zp * math.pow(t * (1-t), .5) 
    n = (s / d)**2
    return int(round(n))

# Using statsmodels
import statsmodels.stats.api as sms
es = sms.proportion_effectsize(0.12, 0.1)
sms.NormalIndPower().solve_power(es, power=0.8, alpha=0.05, ratio=1)
sms.NormalIndPower().plot_power(dep_var="nobs", nobs=np.linspace(1,10000),  effect_size=np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1]))

# Answer problem 2:
# 1. using a linear approximation(Xin and Melanie):
days_to_complete = (test_days * 1. / views * sample_size) - test_days

# 2. Using a poisson distibution: 
'''
A Poisson distribution has a variance = mean (usually noted lambda). The standard deviation is np.sqrt(variance).
'''


def estimate_days(lambda, required, current, num_trials=100, current_days):
	# set up 100 experiments
	trials = np.zeros(num_trials)

	for i in num_trials: # count the number of days that each experiment will take to reach the required_number
		aggregating = current # 
		days = current_days
		while aggragating < required:
			# randomly sample views from a lambda until a required is reached: 
			days+=1
			sample = np.random.poisson(lambda)
			aggragating = aggragating + sample

		trials[cnt] = days

	return np.trials.mean(), np.trials.std()
