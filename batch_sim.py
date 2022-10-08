from run_sim import run_sim
from multiprocessing import Pool

start = 0
end = 60 # need one more than actual end value.
count = 10

def worker_wrapper(args):
    return run_sim(**args)


if __name__ == '__main__':
    jobs = []
    pool = Pool(6)
    for n in range(start,end,count):
        jobs.append({'id':n, 'seed':123, 'no_wfh_perc':n/100,'verbose':0})
    pool.map(worker_wrapper, jobs)
