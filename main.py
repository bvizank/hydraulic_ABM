from run_sim import run_sim
# import logging

# logger = logging.getLogger('wntr')

# for k, v in logging.Logger.manager.loggerDict.items():
#     print('+ [%s] {%s} ' % (str.ljust(k, 20), str(v.__class__)[8:-2]))
#     if not isinstance(v, logging.PlaceHolder):
#         for h in v.handlers:
#             print('     +++', str(h.__class__)[8:-2])

# run_sim(id=0, seed=123, wfh_lag=0, no_wfh_perc=0, bbn_models=[])
# logging.basicConfig(
#     filename='log.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
#     filemode='w'
# )

# logging.info('Starting simulation')
run_sim(id=0, days=200, seed=123, wfh_lag=0, no_wfh_perc=0, bbn_models=['all'],
        city='micropolis', verbose=0.5, hyd_sim=7)
