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


def bootstrap(n, data, b):
    '''
    Method to bootstrap data

    parameters:
    -----------
    n (int):
        number of samples to pick each set
    b (int):
        number of sets to draw
    '''

    output = np.empty((b, n))
    for i in range(b):
        bootstrap = np.array(
            random.choices(list(data.keys()), weights=list(data.values()), k=n)
        )
        output[i, :] = bootstrap

    return output


b_arr = bootstrap(10, data, 1000)
# bootstrap = np.array(random.choices(list(data.keys()), weights=list(data.values()), k=10000))
mean_b = np.mean(b_arr, axis=1)
med_b = np.median(b_arr, axis=1)
up_b = np.percentile(b_arr, 90, axis=1)
var_b = np.var(b_arr, axis=1, ddof=1)
print(f"mean = {np.mean(mean_b)}")
print(f"var = {np.mean(var_b)}")
print(f"90th percentile: {np.percentile(up_b, 90)}")
print(f"50th percentile: {np.percentile(med_b, 50)}")

a = mean_b**2/var_b
b = var_b/mean_b

print(f"random gamma value: {random.gammavariate(a, b)}")

fig, ax = plt.subplots(1, 1)
x = np.linspace(stats.gamma.ppf(0.001, a, scale=1/b),
                stats.gamma.ppf(0.999, a, scale=1/b), 100)

ax.plot(x, stats.gamma.pdf(x, a, scale=1/b),
        'r-', lw=5, alpha=0.6, label='gamma pdf')

rv = stats.gamma(a, scale=1/b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

r = stats.gamma.rvs(a, scale=1/b, size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])

ax.legend(loc='best', frameon=False)

plt.show()
