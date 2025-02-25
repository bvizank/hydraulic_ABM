import os
import sys

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pyAgrum as gum
import data as dt
import random
import math
import numpy as np
import bnlearn as bn
from timeit import default_timer as timer


n = 1000

wfh_gum = gum.loadBN("Input Files/data_driven_models/work_from_home.bif")
wfh_bn = bn.import_DAG("Input Files/data_driven_models/work_from_home.bif")
mask_gum = gum.loadBN("Input Files/data_driven_models/net_files/mask.net")
mask_bn = bn.import_DAG("Input Files/data_driven_models/mask.bif")

agent_list = np.empty((n, dt.bbn_params.shape[1]), dtype=np.int64)
for i in range(n):
    agent_list[i, :] = dt.bbn_params[random.randint(0, dt.bbn_params.shape[0] - 1), :]

# print(dag_nodes)
# print(dag_nodes)
# dag_nodes = wfh_gum.names()

# print(agent_params_gum)


def predict_pyagrum(ev_list, target, dag):
    """Make a prediction using pyagrum"""
    dag_nodes = [n for n in dt.bbn_param_list if dag.exists(n)]
    evs = {}
    for i, param in enumerate(dt.bbn_param_list):
        if param in dag_nodes:
            evs[param] = int(ev_list[i] - 1)

    # evs = {k: int(v) for k, v in agent_params.items() if k in dag_nodes}

    ie = gum.LazyPropagation(dag)
    ie.setNumberOfThreads(1)
    ie.setEvidence(evs)
    ie.addTarget(target)
    ie.makeInference()

    return ie.posterior(dag.idFromName(target))[1]


def predict_bnlearn(ev_list, target, dag):
    # evs['COVIDeffect_4'] = math.floor(evs['COVIDeffect_4'])
    # evidence = dict()
    # for i, item in enumerate(wfh_nodes):
    #     if item != 'work_from_home':
    #         evidence[item] = evidence_agent[item]
    dag_nodes = dag["adjmat"].columns
    evs = {}
    for i, param in enumerate(dt.bbn_param_list):
        if param in dag_nodes:
            evs[param] = ev_list[i] - 1

    # evs = {k: v for k, v in agent_params.items() if k in dag_nodes}

    query = bn.inference.fit(dag, variables=[target], evidence=evs, verbose=0)
    return query.df["p"][1]


start = timer()
gum_post = list()
for i in range(n):
    gum_post.append(predict_pyagrum(agent_list[i, :], "mask", mask_gum))
end = timer()

print(f"pyAgrum time: {end - start}")

start = timer()
bn_post = list()
for i in range(n):
    bn_post.append(predict_bnlearn(agent_list[i, :], "mask", mask_bn))
end = timer()

print(f"bnlearn time: {end - start}")

print(f"diff: {np.sum(np.array(bn_post) - np.array(gum_post))}")
