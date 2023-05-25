import os
import pandas as pd
from utils import read_data

# data_dir = 'Output Files/30_wfh/'
data_dir = '../../OneDrive - North Carolina State University/Research/Code/ABM/Excess Data/no_pm_30/'
read_list = [
    'seir_data',
    'demand',
    'age',
    'pressure',
    'agent_loc',
    'flow',
    'cov_ff',
    'cov_pers',
    'dine',
    'groc',
    'media',
    'ppe',
    'wfh'
]

tot_demand = pd.DataFrame()
tot_seir_data = pd.DataFrame()
tot_age = pd.DataFrame()
tot_pressure = pd.DataFrame()
tot_agent_loc = pd.DataFrame()
tot_flow = pd.DataFrame()
tot_cov_ff = pd.DataFrame()
tot_cov_pers = pd.DataFrame()
tot_dine = pd.DataFrame()
tot_groc = pd.DataFrame()
tot_media = pd.DataFrame()
tot_ppe = pd.DataFrame()
tot_wfh = pd.DataFrame()

for i, folder in enumerate(next(os.walk(data_dir))[1]):
    print(f"Reading files for {i} simluation..........")
    files = os.listdir(os.path.join(data_dir, folder))
    # for file in files:
        # if file.endswith('.pkl'):
        #     os.remove(os.path.join(data_dir, folder, file))
    output = read_data(data_dir + folder + '/', read_list)
    output['cov_pers'] = output['cov_pers'].replace([7, 1], [0, 1])
    for item in read_list:
        globals()['tot_' + item] = pd.concat((globals()['tot_' + item], output[item]))

for item in read_list:
    globals()['tot_'+item+'_row'] = globals()['tot_'+item].groupby(globals()['tot_'+item].index)
    globals()['tot_'+item+'_row'].mean().to_pickle(data_dir+'avg_'+item+'.pkl')
    globals()['tot_'+item+'_row'].std().to_pickle(data_dir+'sd_'+item+'.pkl')
