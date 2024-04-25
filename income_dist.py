import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


data = {
    0: 7.6,
    10000: 11.9,
    15000: 13.3,
    25000: 14.6,
    35000: 12.3,
    50000: 14.3,
    75000: 9.3,
    100000: 11.1,
    150000: 2.6,
    200000: 3
}

bootstrap = np.array(random.choices(list(data.keys()), weights=list(data.values()), k=10000))
mean_b = np.mean(bootstrap)
var_b = np.var(bootstrap, ddof=1)
print(f"mean = {mean_b}")
print(f"var = {var_b}")

a = mean_b**2/var_b
b = var_b/mean_b

print(f"random gamma value: {random.gammavariate(a, b)}")

fig, ax = plt.subplots(1, 1)
x = np.linspace(stats.gamma.ppf(0.01, a, scale=1/b),
                stats.gamma.ppf(0.99, a, scale=1/b), 100)

ax.plot(x, stats.gamma.pdf(x, a, scale=1/b),
        'r-', lw=5, alpha=0.6, label='gamma pdf')

rv = stats.gamma(a, scale=1/b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

r = stats.gamma.rvs(a, scale=1/b, size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])

ax.legend(loc='best', frameon=False)

plt.show()
