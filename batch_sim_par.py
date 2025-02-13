from run_sim import run_sim
from mpi4py import MPI
import os
import pandas as pd
from utils import read_data, delete_contents
import shutil
import data as dt
import logging
import random


# parameters
bw = False,
bbn_models = []
dist_income = False,
twa_process = 'absolute'
twa_mods = [130, 140, 150]
output_loc = 'Output Files/30_base/'

# delete all the handlers from the root logger
logger = logging.getLogger()
for hdlr in logger.handlers[:]:
    logger.removeHandler(hdlr)

# set up MPI
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

# check to see if the file directory exists, if not make it
if rank == 0:
    if os.path.isdir(output_loc):
        delete_contents(output_loc)
        os.mkdir(output_loc + 'hh_results')
    else:
        os.mkdir(output_loc)
        os.mkdir(output_loc + 'hh_results')

comm.Barrier()

# set a new file logger in place of the stream handler
# this will eliminate errors being sent to sys.stderr
fh = logging.FileHandler('logs/log' + str(rank), 'w')
fh.setLevel(logging.DEBUG)
formmater = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
fh.setFormatter(formmater)
logger.addHandler(fh)

for i in range(0+rank, 30, nprocs):
    seed = random.random()
    # run the simulation
    run_sim(
        id=i,
        days=180,
        seed=seed,
        wfh_lag=0,
        no_wfh_perc=0,
        bbn_models=bbn_models,
        daily_contacts=30,
        city='clinton',
        verbose=0,
        hyd_sim='monthly',
        warmup=True,
        twa_mods=twa_mods,
        bw=bw,
        ind_min_demand=0,
        dist_income=dist_income,
        output_loc=output_loc
    )
    os.remove('temp' + str(i) + '.bin')
    os.remove('temp' + str(i) + '.rpt')
    os.remove('temp' + str(i) + '.inp')

''' Make sure all procs have finished before moving on to data synthesis '''
comm.Barrier()

'''
For each file in the read_list we need to read in the data from each of
of the 30 folders then export the mean and variance
'''
for i in range(0+rank, len(dt.read_list), nprocs):
    file = dt.read_list[i]
    logger.debug(f'Reading data for {file}')
    curr_data = pd.DataFrame()
    # iterate through folders and append the data from each run to the
    # curr_data var
    for folder in next(os.walk(output_loc))[1]:
        if folder == 'hh_results':
            continue
        new_data = read_data(os.path.join(output_loc, folder, ''), [file])[file]
        if file in dt.avg_list:
            # new_data = new_data.T.groupby(level=0).mean()
            # new_data = new_data.T
            shutil.copy(
                output_loc + folder + '/' + file + '.pkl',
                output_loc + 'hh_results/' + file + '_' + folder + '.pkl'
            )
        else:
            logger.info('Current data')
            logger.info(f'{curr_data}')
            logger.info('New data')
            logger.info(f'{new_data}')
            logger.info('Duplicated')
            logger.debug(new_data.loc[:, new_data.columns.duplicated()])
            curr_data = pd.concat([
                curr_data,
                new_data
            ])

    # group values by index and then export mean and var data
    if file not in dt.avg_list:
        logger.info(f'{curr_data}')
        curr_data = curr_data.groupby(curr_data.index)
        curr_data.mean().to_pickle(output_loc + '/' + 'avg_' + file + '.pkl')
        curr_data.var().to_pickle(output_loc + '/' + 'var_' + file + '.pkl')
