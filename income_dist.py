import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


data = {
    0: 76,
    10000: 119,
    15000: 133,
    25000: 146,
    35000: 123,
    50000: 143,
    75000: 93,
    100000: 111,
    150000: 26,
    200000: 30
}


def bootstrap(data, b):
    '''
    Method to bootstrap data

    parameters:
    -----------
    n (int):
        number of samples to pick each set
    b (int):
        number of sets to draw
    '''
    rng = np.random.default_rng(seed=218)
    n = sum(list(data.values()))
    print(n)
    output = np.empty((b, n))
    for i in range(b):
        bootstrap = np.empty(n)
        index = 0
        for j, key in enumerate(data):
            '''
            Get a set of samples for the given income range that is 100
            times longer than the percentage given by the data.
            e.g. for $0 - $10,000, we want 76 samples uniformly distributed
            between 0 and 10000.
            '''
            if j != (len(data) - 1):
                bootstrap[index:(index + data[key])] = rng.uniform(
                    list(data.keys())[j], list(data.keys())[j+1],
                    size=data[key]
                )
            else:
                # at the end we need to arbitrarily set an upper bound
                bootstrap[index:(index + data[key])] = rng.uniform(
                    list(data.keys())[j], list(data.keys())[j]*3,
                    size=data[key]
                )
            index += data[key]

        output[i, :] = bootstrap

    return output


b_arr = bootstrap(data, 10000)
# bootstrap = np.array(random.choices(list(data.keys()), weights=list(data.values()), k=10000))
mean_b = np.mean(b_arr)
med_b = np.median(b_arr, axis=1)
up_b = np.percentile(b_arr, 90, axis=1)
var_b = np.var(b_arr, ddof=1)
print(f"mean = {(mean_b)}")
print(f"var = {(var_b)}")
print(f"90th percentile: {np.percentile(up_b, 90)}")
print(f"50th percentile: {np.percentile(med_b, 50)}")

a = mean_b**2/var_b
b = var_b/mean_b

print(f"median from gamma: {np.median(stats.gamma.rvs(a, scale=b, size=1000))}")

fig, ax = plt.subplots(1, 1)
x = np.linspace(stats.gamma.ppf(0.001, a, scale=b),
                stats.gamma.ppf(0.999, a, scale=b), 100)

ax.plot(x, stats.gamma.pdf(x, a, scale=b),
        'r-', lw=5, alpha=0.6, label='gamma pdf')

rv = stats.gamma(a, scale=b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

r = stats.gamma.rvs(a, scale=b, size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])

ax.legend(loc='best', frameon=False)

plt.show()
