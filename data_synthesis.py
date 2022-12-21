import os
import pandas as pd
from utils import read_data

data_dir = 'Output Files/no_pb_30/'
read_list = ['seir', 'demand', 'age', 'pressure', 'agent', 'flow']

tot_demand = pd.DataFrame()
tot_seir = pd.DataFrame()
tot_age = pd.DataFrame()
tot_pressure = pd.DataFrame()
tot_agent = pd.DataFrame()
tot_flow = pd.DataFrame()

for i, folder in enumerate(next(os.walk(data_dir))[1]):
    output = read_data(data_dir + folder + '/', read_list)
    for item in read_list:
        globals()['tot_' + item] = pd.concat((globals()['tot_' + item], output[item]))

for item in read_list:
    globals()['tot_'+item+'_row'] = globals()['tot_'+item].groupby(globals()['tot_'+item].index)
    globals()['tot_'+item+'_row'].mean().to_pickle(data_dir+'avg_'+item+'.pkl')
    globals()['tot_'+item+'_row'].std().to_pickle(data_dir+'sd_'+item+'.pkl')
