import warnings
warnings.simplefilter("ignore", UserWarning)
from Hydraulic_abm_SEIR import ConsumerModel
from wntr_1 import *
from Char_micropolis_static_loc import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime, perf_counter
import os

curr_dt = strftime("%Y-%m-%d_%H-%M", localtime())
output_loc = 'Output Files/' + curr_dt + '_results'
os.mkdir(output_loc)

output_file = 'datasheet.xlsx'

f = Micro_pop
days = 90
model = ConsumerModel(f, seed=123, days=days, wfh=True, wfh_lag=0.01)

start = perf_counter()

for t in range(24*days):
    model.step()

Demands_test = np.zeros(24*days)
for node in All_terminal_nodes:
    add_demands = model.demand_matrix[node]
    Demands_test = np.add(Demands_test, add_demands)

stop = perf_counter()

print('Time to complete: ', stop - start)

plt.plot(Demands_test)
# plt.plot(Demands_test, label='Total Demand')
plt.xlabel("Time (sec)")
plt.ylabel("Demand (ML)")
plt.legend(loc='best')
plt.savefig(output_loc + '/' + 'demands.png')
plt.close()
print('Total demands are ' + str(Demands_test.sum(axis = 0)))

# print(Demands_test[:,0])

model.status_tot['t'] = Micro_pop * model.status_tot['t'] / 24

plt.plot('t', 'S', data = model.status_tot, label = 'Susceptible')
plt.plot('t', 'E', data = model.status_tot, label = 'Exposed')
plt.plot('t', 'I', data = model.status_tot, label = 'Infected')
plt.plot('t', 'R', data = model.status_tot, label = 'Recovered')
plt.plot('t', 'D', data = model.status_tot, label = 'Dead')
plt.xlabel('Time (days)')
plt.ylabel('Percent Population')
plt.legend()
plt.savefig(output_loc + '/' + 'seir.png')
plt.close()

model.status_tot['I'] = Micro_pop * model.status_tot['I']
model.status_tot['sum_I'] = Micro_pop * model.status_tot['sum_I']

plt.plot('t', 'I', data = model.status_tot, label = 'Infected')
plt.plot('t', 'sum_I', data = model.status_tot, label = 'Cumulative I')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.savefig(output_loc + '/' + 'infected.png')
plt.close()

''' Save the model outputs '''
model.status_tot['t'] = model.status_tot['t'] * 24 * 3600
# model.status_tot['t'] = pd.to_numeric(model.status_tot['t'],downcast="integer")
# model.status_tot = model.status_tot.set_index('t')
# model.status_tot = pd.concat([model.status_tot, Demands_test], axis=1)

with pd.ExcelWriter(output_loc + '/' + output_file) as writer:
    model.status_tot.to_excel(writer, sheet_name='seir_data')
    model.param_out.to_excel(writer, sheet_name='params')
    model.demand_matrix.to_excel(writer, sheet_name='demand')
    model.pressure_matrix.to_excel(writer, sheet_name='pressure')
    model.age_matrix.to_excel(writer, sheet_name='age')
