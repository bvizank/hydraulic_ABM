from run_sim import run_sim
import logging

# delete all the handlers from the root logger
logger = logging.getLogger()
for hdlr in logger.handlers[:]:
    logger.removeHandler(hdlr)

# set a new file logger in place of the stream handler
# this will eliminate errors being sent to sys.stderr
fh = logging.FileHandler('log', 'w')
fh.setLevel(logging.DEBUG)
formmater = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
fh.setFormatter(formmater)
logger.addHandler(fh)

# run the simulation
run_sim(
    id=0,
    days=90,
    seed=0,
    write_inp=True,
    wfh_lag=0,
    no_wfh_perc=0,
    bbn_models=['all'],
    daily_contacts=30,
    city='micropolis',
    verbose=0.5,
    hyd_sim='monthly',
    warmup=True,
    bw=True,
    ind_min_demand=0,
    twa_process='absolute'
)
