import numpy as np
import scipy.stats as stats
# import matplotlib.pyplot as plt


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
    a = list()
    scale = list()
    loc = list()
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

        # curr_mean = np.mean(bootstrap)
        # curr_var = np.var(bootstrap, ddof=1)

        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(bootstrap)
        a.append(fit_alpha)
        scale.append(fit_beta)
        loc.append(fit_loc)

        output[i, :] = bootstrap

    return output, a, scale, loc


if __name__ in "__main__":
    b_arr, a, scale, loc = bootstrap(data, 10000)
    # bootstrap = np.array(random.choices(list(data.keys()), weights=list(data.values()), k=10000))
    mean_b = np.mean(b_arr)
    med_b = np.median(b_arr, axis=1)
    up_b = np.percentile(b_arr, 90, axis=1)
    low_b = np.percentile(b_arr, 20, axis=1)
    var_b = np.var(b_arr, ddof=1)
    print(f"mean = {(mean_b)}")
    print(f"var = {(var_b)}")
    print(f"90th percentile: {np.percentile(up_b, 50)}")
    print(f"50th percentile: {np.percentile(med_b, 50)}")
    print(f"20th percentile: {np.percentile(low_b, 50)}")

    # a = mean_b**2/var_b
    # b = var_b/mean_b

    mean_a = np.mean(np.array(a))
    mean_b = np.mean(np.array(scale))
    mean_loc = np.mean(np.array(loc))
    # mean_a = a[0]
    # mean_b = scale[0]
    # mean_loc = loc[0]

    # fit_alpha, fit_loc, fit_beta = stats.gamma.fit(b_arr[0, :])
    # print(fit_alpha)
    # print(fit_loc)
    # print(fit_beta)

    # gamma_data = stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=10000)
    gamma_data = stats.gamma.rvs(mean_a, loc=mean_loc, scale=mean_b, size=10000)
    print(f"median from gamma: {np.median(gamma_data)}")
    print(f"20th percentile from gamma: {np.percentile(gamma_data, 20)}")

    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(stats.gamma.ppf(0.001, a, scale=b),
    #                 stats.gamma.ppf(0.999, a, scale=b), 100)

    # ax.plot(x, stats.gamma.pdf(x, a, scale=b),
    #         'r-', lw=5, alpha=0.6, label='gamma pdf')

    # rv = stats.gamma(a, scale=b)
    # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    # r = stats.gamma.rvs(a, scale=b, size=1000)
    # ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    # ax.set_xlim([x[0], x[-1]])

    # ax.legend(loc='best', frameon=False)

    # plt.show()
