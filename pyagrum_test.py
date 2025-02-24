import pyAgrum as gum
import data as dt
import random
import math
import numpy as np
import bnlearn as bn
from timeit import default_timer as timer


n = 1000

wfh_gum = gum.loadBN('Input Files/data_driven_models/work_from_home.bif')
wfh_bn = bn.import_DAG('Input Files/data_driven_models/work_from_home.bif')

agent_list = np.empty((n, dt.bbn_params.shape[1]), dtype=np.int64)
for i in range(n):
    agent_list[i, :] = dt.bbn_params[
        random.randint(0, dt.bbn_params.shape[0] - 1), :
    ]

dag_nodes = [n for n in dt.bbn_param_list if wfh_gum.exists(n)]
print(dag_nodes)
# print(dag_nodes)
# dag_nodes = wfh_gum.names()

# print(agent_params_gum)


def predict_pyagrum(ev_list):
    ''' Make a prediction using pyagrum '''
    evs = {}
    for i, param in enumerate(dt.bbn_param_list):
        if param in dag_nodes:
            evs[param] = int(ev_list[i] - 1)

    # evs = {k: int(v) for k, v in agent_params.items() if k in dag_nodes}

    ie = gum.LazyPropagation(wfh_gum)
    ie.setEvidence(evs)
    ie.addTarget("work_from_home")
    ie.makeInference()

    return ie.posterior(wfh_gum.idFromName("work_from_home"))[1]


def predict_bnlearn(ev_list):
    # evs['COVIDeffect_4'] = math.floor(evs['COVIDeffect_4'])
    # evidence = dict()
    # for i, item in enumerate(wfh_nodes):
    #     if item != 'work_from_home':
    #         evidence[item] = evidence_agent[item]
    evs = {}
    for i, param in enumerate(dt.bbn_param_list):
        if param in dag_nodes:
            evs[param] = ev_list[i] - 1

    # evs = {k: v for k, v in agent_params.items() if k in dag_nodes}

    query = bn.inference.fit(wfh_bn,
                             variables=['work_from_home'],
                             evidence=evs,
                             verbose=0)
    return query.df['p'][1]


start = timer()
for i in range(n):
    gum_post = predict_pyagrum(agent_list[i, :])
end = timer()

print(f"pyAgrum time: {end - start}")

start = timer()
for i in range(n):
    bn_post = predict_bnlearn(agent_list[i, :])
end = timer()

print(f"bnlearn time: {end - start}")

