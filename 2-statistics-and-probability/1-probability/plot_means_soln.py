import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt
import sys


def make_draws(distribution, parameters, size=200):
    """
    - distribution(STR) [specify which distribution to draw from]
    - parameters(DICT) [dictionary with different params depending]
    - size(INT) [the number of values we draw]
    """
    if distribution == 'uniform':
        a, b = parameters['a'], parameters['b']
        values = scs.uniform(a, b).rvs(size)

    elif distribution == 'poisson':
        lam = parameters['lam']
        values = scs.poisson(lam).rvs(size)

    elif distribution == 'binomial':
        n, p = parameters['n'], parameters['p']
        values = scs.binom(n, p).rvs(size)

    elif distribution == 'exponential':
        lam = parameters['lam']
        values = scs.expon(lam).rvs(size)

    elif distribution == 'geometric':
        p = parameters['p']
        values = scs.geom(p).rvs(size)

    return values


def plot_means(distribution, parameters, size=200, repeats=5000):
    """
    - distribution(STR) [specify which distribution to draw from]
    - parameters(DICT) [dictionary with different params depending]
    - size(INT) [the number of values we draw]
    - repeat(INT) [the times we draw values]
    """
    mean_vals = []
    for _ in range(repeats):
        values = make_draws(distribution, parameters, size=200)
        mean_vals.append(np.mean(values))

    d_xlabel = {'uniform': 'Mean of randomly drawn values from a uniform',
                'poisson': 'Mean events happening in an interval',
                'binomial': 'Mean number of success',
                'exponential': 'Mean of waiting time before an event happens',
                'geometric': 'Mean rounds of failures before a success'
                }
    xlabel = d_xlabel[distribution]
    plt.hist(mean_vals, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title('Mean of %s samples with size %s drawn from %s distribution' %
              (repeats, size, distribution.capitalize()), fontsize=14)
    plt.show()

if __name__ == '__main__':
    plot_means('uniform', {'a': 10, 'b': 20})
    plot_means('poisson', {'lam': 2})
    plot_means('binomial', {'n': 100, 'p': 0.1})
    plot_means('exponential', {'lam': 2})
    plot_means('geometric', {'p': 0.1})
