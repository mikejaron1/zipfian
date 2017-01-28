'''
1. What is hypothesis testing?
  - Hypothesis testing uses statistical tests to determine if a hypothesis is
    true based on a sample drawn from the population

2. Name the hypothesis test you would use in the following situations:
  a. Testing a population mean give 1 sample
     - One sample t-test

  b. Testing if 2 populations have the same mean given 2 samples
     - Two sample independent t-test

  c. Testing if 2 populations have the same mean given 2 samples with a
     one-to-one mapping between each of the values in the samples
     - Two sample paired t-test
'''


from __future__ import division
import scipy.stats

class PMF(object):
    def __init__(self, pmf_d):
        self.pdict = pmf_d

    def __repr__(self):
        return str(self.pdict.items())

    def prob(self, key):
        return self.pdict[key]

    def set(self, key_val):
        self.pdict[key_val[0]] = key_val[1]
        norm_const = sum(self.pdict.values())
        self.pdict = { k : v/norm_const for k,v in self.pdict.iteritems() }


class RV(object):
    def __init__(self, PMF):
        self.pmf = PMF
        self.outcome_list = self.pmf.pdict.keys()
        self.rv = scipy.stats.rv_discrete(values=(self.pmf.pdict.keys(), self.pmf.pdict.values()) )

    def all_outcomes(self):
        return self.outcome_list

    def pmf(self):
        return self.pmf

    def sample(self):
        return self.rv.rvs()


die = PMF({"1": 1/6, "2": 1/6, "3": 1/6, "4": 1/6, "5": 1/6, "6": 1/6 })
die_rv = RV(die)
print die_rv.sample()
print die_rv.sample()
print die_rv.sample()
print die_rv.sample()
print die_rv.all_outcomes()
