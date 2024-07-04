from run_sim import run_sim
from mpi4py import MPI
import os
import logging


# delete all the handlers from the root logger
logger = logging.getLogger()
for hdlr in logger.handlers[:]:
    logger.removeHandler(hdlr)

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

# set a new file logger in place of the stream handler
# this will eliminate errors being sent to sys.stderr
fh = logging.FileHandler('logs/log' + str(rank), 'w')
fh.setLevel(logging.DEBUG)
formmater = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
fh.setFormatter(formmater)
logger.addHandler(fh)

for i in range(0+rank, 30, nprocs):

    # run the simulation
    run_sim(
        id=i,
        days=180,
        seed=i,
        wfh_lag=0,
        no_wfh_perc=0,
        bbn_models=[],
        daily_contacts=30,
        city='micropolis',
        verbose=0,
        hyd_sim='monthly',
        warmup=True,
        bw=False,
        ind_min_demand=0
    )
    os.remove('temp' + str(i) + '.bin')
    os.remove('temp' + str(i) + '.rpt')
    os.remove('temp' + str(i) + '.inp')
