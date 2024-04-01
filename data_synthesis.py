import os
import pandas as pd
from utils import read_data
from mpi4py import MPI

''' MPI initialization '''
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


# data_dir = 'Output Files/30_wfh/'
data_dir = '/Users/vizan/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/Research/Equity/excess_data/30_base_equity'
# data_dir = 'D:/OneDrive - North Carolina State University/Research/Code/ABM/Excess Data/wfh_30/'

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

# tot_demand = pd.DataFrame()
# tot_seir_data = pd.DataFrame()
# tot_age = pd.DataFrame()
# tot_pressure = pd.DataFrame()
# tot_agent_loc = pd.DataFrame()
# tot_flow = pd.DataFrame()
# tot_cov_ff = pd.DataFrame()
# tot_cov_pers = pd.DataFrame()
# tot_dine = pd.DataFrame()
# tot_groc = pd.DataFrame()
# tot_media = pd.DataFrame()
# tot_ppe = pd.DataFrame()
# tot_wfh = pd.DataFrame()

'''
For each file in the read_list we need to read in the data from each of
of the 30 folders then export the mean and variance
'''
for i, file in read_list:
    print(f'Reading data for {file}')
    curr_data = pd.DataFrame()
    # iterate through folders and append the data from each run to the 
    # curr_data var
    for folder in next(os.walk(data_dir))[1]:
        curr_data = pd.concat((
            curr_data,
            read_data(os.path.join(data_dir, folder, ''), [file])[file]
        ))
    
    # group values by index and then export mean and var data
    curr_data = curr_data.groupby(curr_data.index)
    curr_data.mean().to_pickle(data_dir + '/' + 'avg_' + file + '.pkl')
    curr_data.var().to_pickle(data_dir + '/' + 'var_' + file + '.pkl')

# for i, folder in enumerate(next(os.walk(data_dir))[1]):
#     print(f"Reading files for {i} simluation..........")
#     files = os.listdir(os.path.join(data_dir, folder))
#     # for file in files:
#         # if file.endswith('.pkl'):
#         #     os.remove(os.path.join(data_dir, folder, file))
#     output = read_data(data_dir + folder + '/', read_list)
#     output['cov_pers'] = output['cov_pers'].replace([7, 1], [0, 1])
#     for item in read_list:
#         globals()['tot_' + item] = pd.concat((globals()['tot_' + item], output[item]))

# for item in read_list:
#     globals()['tot_'+item+'_row'] = globals()['tot_'+item].groupby(globals()['tot_'+item].index)
#     globals()['tot_'+item+'_row'].mean().to_pickle(data_dir+'avg_'+item+'.pkl')
#     globals()['tot_'+item+'_row'].var().to_pickle(data_dir+'var_'+item+'.pkl')
