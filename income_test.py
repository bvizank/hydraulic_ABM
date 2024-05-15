import os
import sys
if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Hydraulic_abm_SEIR import ConsumerModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


plt.rcParams['figure.dpi'] = 500
plt.rcParams['figure.figsize'] = [3.5, 3.5]

high_income = list()
med_income = list()
low_income = list()


def box_plots(ind_dist, model, i):
    p20 = (model.income_comb['income']).to_numpy()[(ind_dist <= 0.2)]
    p40 = (model.income_comb['income']).to_numpy()[
        (ind_dist > 0.2) & (ind_dist <= 0.4)
    ]
    p60 = (model.income_comb['income']).to_numpy()[
        (ind_dist > 0.4) & (ind_dist <= 0.6)
    ]
    p80 = (model.income_comb['income']).to_numpy()[
        (ind_dist > 0.6) & (ind_dist <= 0.8)
    ]
    p100 = (model.income_comb['income']).to_numpy()[ind_dist > 0.80]

    plt.boxplot([p20, p40, p60, p80, p100])
    plt.savefig('income_boxplot_' + str(i) + '.png',
                format='png', bbox_inches='tight')
    plt.close()


def scatter_plot(ind_dist, model, i):
    x = ind_dist
    x_norm = (
        (x - np.min(x)) / (np.max(x) - np.min(x))
    )
    x_with_intercept = np.empty(shape=(len(x_norm), 2), dtype=np.float64)
    x_with_intercept[:, 0] = 1
    x_with_intercept[:, 1] = x_norm

    y = model.income_comb.loc[:, 'income']

    ols_model = sm.OLS(y, x_with_intercept).fit()
    sse = np.sum((ols_model.fittedvalues - y)**2)
    ssr = np.sum((ols_model.fittedvalues - y.mean())**2)
    sst = ssr + sse
    print(f'R2 = {ssr/sst}')
    print(len(x_norm))
    print(np.sqrt(sse/(len(x_norm) - 2)))
    print(ols_model.summary())

    lr_model = LinearRegression()
    x = x_norm[:, np.newaxis]
    lr_model.fit(x, y)
    print(lr_model.score(x, y))
    print(lr_model.coef_)
    print(lr_model.intercept_)

    plt.plot(x, lr_model.predict(x))
    plt.scatter(x, y)
    plt.xlabel('Normalized Industrial Distance')
    plt.ylabel('Median BG Income')
    plt.savefig(
        'micropolis_income' + str(i) + '.png',
        format='png', bbox_inches='tight'
    )
    plt.close()


def network_plot(model, i):
    ax = wntr.graphics.plot_network(
        model.wn,
        node_attribute=model.income_comb.loc[:, 'income'].groupby(level=0).mean(),
        node_size=5
    )
    plt.savefig('income_map_' + str(i) + '.png',
                format='png', bbox_inches='tight')
    plt.close()


def dist_hist(ind_dist, i):
    plt.hist(ind_dist)
    plt.savefig('ind_dist_hist_' + str(i) + '.png',
                format='png', bbox_inches='tight')
    plt.close()


for i in range(10):
    model = ConsumerModel(
        4606,
        'micropolis',
        days=1,
        id=i,
        seed=i,
        bbn_models=[],
        verbose=0.5
    )
    # households = [h.node for i, hs in model.households.items() for h in hs]
    ind_dist = np.array(
        [h.ind_dist for i, hs in model.households.items() for h in hs]
    )

    # box_plots(ind_dist, model, i)

    scatter_plot(ind_dist, model, i)

    # dist_hist(ind_dist, i)

    # network_plot(model, i)

    high_income.append(np.percentile(np.array(model.income), 90))
    med_income.append(np.percentile(np.array(model.income), 50))
    low_income.append(model.income_comb['level'].value_counts()[1])

print(sum(high_income) / len(high_income))
print(sum(med_income) / len(med_income))
print(sum(low_income) / len(low_income))
