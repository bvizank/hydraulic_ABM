import os
import pandas as pd
from utils import read_data
from mpi4py import MPI
import logging
import shutil

''' MPI initialization '''
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/log'+str(rank), filemode='w', level=logging.INFO)

data_dir = 'Output Files/30_base-bw_equity/'
# data_dir = '/Users/vizan/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/Research/Equity/excess_data/30_base_equity'
# data_dir = 'D:/OneDrive - North Carolina State University/Research/Code/ABM/Excess Data/wfh_30/'

read_list = [
    'age',
    'agent_loc',
    'burden',
    'bw_cost',
    'bw_demand',
    'cook',
    'cov_ff',
    'cov_pers',
    'demand',
    'dine',
    'drink',
    'flow',
    'groc',
    'hygiene',
    'income',
    'media',
    'ppe',
    'pressure',
    'seir_data',
    'traditional',
    'tw_cost',
    'wfh'
]

avg_list = [
    'income',
    'bw_cost',
    'bw_demand',
    'tw_cost',
    'cook',
    'hygiene',
    'drink',
    'burden',
    'traditional'
]

'''
For each file in the read_list we need to read in the data from each of
of the 30 folders then export the mean and variance
'''
for i in range(0+rank, len(read_list), nprocs):
    file = read_list[i]
    logger.debug(f'Reading data for {file}')
    curr_data = pd.DataFrame()
    # iterate through folders and append the data from each run to the 
    # curr_data var
    for folder in next(os.walk(data_dir))[1]:
        if folder == 'hh_results':
            continue
        new_data = read_data(os.path.join(data_dir, folder, ''), [file])[file]
        if file in avg_list:
            # new_data = new_data.T.groupby(level=0).mean()
            # new_data = new_data.T
            shutil.copy(data_dir + folder + '/' + file + '.pkl', data_dir + 'hh_results/' + file + '_' + folder + '.pkl')
        else:
            logger.info('Current data')
            logger.info(f'{curr_data}')
            logger.info('New data')
            logger.info(f'{new_data}')
            logger.info('Duplicated')
            logger.debug(new_data.loc[:,new_data.columns.duplicated()])
            curr_data = pd.concat([
                curr_data,
                new_data
            ])

    # group values by index and then export mean and var data
    if file not in avg_list:
        logger.info(f'{curr_data}')
        curr_data = curr_data.groupby(curr_data.index)
        curr_data.mean().to_pickle(data_dir + '/' + 'avg_' + file + '.pkl')
        curr_data.var().to_pickle(data_dir + '/' + 'var_' + file + '.pkl')
