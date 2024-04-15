from Hydraulic_abm_SEIR import ConsumerModel
import numpy as np


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

    high_income.append(np.percentile(np.array(model.income), 90))
    med_income.append(np.percentile(np.array(model.income), 50))
    low_income.append(model.income_comb['level'].value_counts()[1])

print(sum(high_income) / len(high_income))
print(sum(med_income) / len(med_income))
print(sum(low_income) / len(low_income))
