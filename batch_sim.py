from run_sim import run_sim
from mpi4py import MPI

start = 10
end = 60 # need one more than actual end value.
count = 10

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

for i in range(start,10*size,count):
    run_sim(id=rank, seed=123, wfh_lag=0, no_wfh_perc=i/100, verbose=0)
