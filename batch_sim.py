import warnings
warnings.simplefilter("ignore", UserWarning)
from Hydraulic_abm_SEIR import ConsumerModel
from wntr_1 import *
from Char_micropolis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

output_file = 'lag_period_22-28days.xlsx'
start = 22
end = 29 # need one more than actual end value.
count = 1

f = Micro_pop

for i in range(start,end):
    model = ConsumerModel(f, seed = 123, lag_period = i)

    for t in range(24*90):
        model.step()

    print(model.status_tot)

    if i == start:
        status_tot = model.status_tot
        param_out = model.param_out
    else:
        for index,col_name in enumerate(model.status_tot.columns):
            status_tot.insert(count+index*(count+1), (str(col_name) + str(count)), model.status_tot[col_name])
        param_out = param_out.append(model.param_out)
        count += 1

with pd.ExcelWriter(output_file) as writer:
        status_tot.to_excel(writer, sheet_name='seir_data')
        param_out.to_excel(writer, sheet_name='params')
