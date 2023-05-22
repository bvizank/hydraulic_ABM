from run_sim import run_sim
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

for i in range(0+rank, 30, nprocs):
    run_sim('micropolis', id=i, days=90, bbn_models=['all'], verbose=0)
