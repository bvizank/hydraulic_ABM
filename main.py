from run_sim import run_sim

# run_sim(id=0, seed=123, wfh_lag=0, no_wfh_perc=0, bbn_models=[])
run_sim(id=0, days=90, seed=123, wfh_lag=0, no_wfh_perc=0, bbn_models=['all'],
        city='micropolis', verbose=1)
