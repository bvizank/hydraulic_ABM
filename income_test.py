from Hydraulic_abm_SEIR import ConsumerModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr


high_income = list()
med_income = list()
low_income = list()
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
    # income = pd.DataFrame.from_dict(
    #     {'level': model.income_comb['level'], 'dist': ind_dist},
    #     orient='index', columns=households
    # )
    # print(income)
    print(model.income_comb['level'] == 1)
    low = (model.income_comb['income']).to_numpy()[(ind_dist <= 0.33)]
    med = (model.income_comb['income']).to_numpy()[
        (ind_dist > 0.33) & (ind_dist <= 0.67)
    ]
    high = (model.income_comb['income']).to_numpy()[ind_dist > 0.67]

    plt.boxplot([low, med, high])
    plt.show()

    high_income.append(np.percentile(np.array(model.income), 90))
    med_income.append(np.percentile(np.array(model.income), 50))
    low_income.append(model.income_comb['level'].value_counts()[1])

print(sum(high_income) / len(high_income))
print(sum(med_income) / len(med_income))
print(sum(low_income) / len(low_income))
