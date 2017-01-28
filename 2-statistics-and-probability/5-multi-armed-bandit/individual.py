import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_with_fill(x, y, label):
    lines = plt.plot(x, y, label=label, lw=2)
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())


# 1. Getting the data from the files
with open('data/siteA.txt') as f:
    siteA = [int(line.strip()) for line in f]

with open('data/siteB.txt') as f:
    siteB = [int(line.strip()) for line in f]

# 2. Plotting the uniform prior distribution as a beta distribution
x = np.arange(0, 1.001, 0.001)
y = stats.uniform.pdf(x)
plot_with_fill(x, y, "Prior")
plt.ylim(0, 2)
plt.legend()
plt.show()

# 3. Plotting the uniform prior distribution as a beta distribution
alpha = 1
beta = 1
y = stats.beta.pdf(x, alpha, beta)
plot_with_fill(x, y, "Prior")

# 4-6. Plotting the posterior distributions 
for start, end in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800)]:
    alpha += siteA[start:end].count(1)
    beta += siteA[start:end].count(0)
    y = stats.beta.pdf(x, alpha, beta)
    plot_with_fill(x, y, label="Posterior after %d views" % end)

plt.legend()
plt.show()

# 7. Make a graph of both site A's and site B's distributions
alphaA = 1 + siteA.count(1)
betaA = 1 + siteA.count(0)
alphaB = 1 + siteB.count(1)
betaB = 1 + siteB.count(0)

plot_with_fill(x, stats.beta.pdf(x, alphaA, betaA), "Site A")
plot_with_fill(x, stats.beta.pdf(x, alphaB, betaB), "Site B")
plt.xlim(0, 0.2)
plt.legend()
plt.show()

print 'mean of A is', alphaA / float(alphaA + betaA)

# 8. Determine what probability that site B is better than site A.
sample_size = 10000
A = np.random.beta(alphaA, betaA, sample_size)
B = np.random.beta(alphaB, betaB, sample_size)
print "%f%% chance site B is better than site A" \
    % (np.sum(B > A) / float(sample_size))

# 9. 95% HDI
print 'min', stats.beta.ppf(.025, alphaA, betaA)
print 'max', stats.beta.ppf(.975, alphaA, betaA)

# 10. What is the probability that site B is 2 percentage points better than
# site A?
print "%f%% chance site B is 2 percentage points better than site A" \
    % (np.sum(B > A + 0.02) / float(sample_size))

# 11. Frequentist approach
t, p = stats.ttest_ind(siteA, siteB)
if p < 0.05:
    print "We can reject null hypothesis"
else:
    print "We cannot reject null hypothesis"

# 12. Should the company switch to site B? 
# In order to decide whether to switch, it's necessary to understand the opportunity cost--i.e.
# the cost of what is foregoing other initiatives by investing in switching to site B.
# Possibly the company could make even more by investing the same resources in some other project.
# Furthermore, presumably neither site A nore B is going to be used forever, so the marginal benefit is
# only realized for as long as site A would have been up if site B were not used.

