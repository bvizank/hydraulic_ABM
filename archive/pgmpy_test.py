from pgmpy.readwrite import BIFReader
from pgmpy.inference import ApproxInference
from pgmpy.inference import VariableElimination
from time import perf_counter
import bnlearn as bn


wfh = BIFReader('Input Files/data_driven_models/work_from_home.bif')
wfh_dag = bn.import_DAG('Input Files/data_driven_models/work_from_home.bif', verbose=0)
print(wfh.get_states())

evidence = {
    'DemEdu': '1',
    'COVIDeffect_2': '4',
    'MediaExp_7': '1',
    'DemHealthcare': '2',
    'Longitude_3': '2',
    'MediaExp_6': '1',
    'FinitePool_6': '5'
}

# ap_start = perf_counter()
# infer = ApproxInference(wfh.get_model())

# query = infer.query(
#     ['work_from_home'],
#     evidence=evidence,
#     show_progress=False
# )
# print(f"Approximate time: {perf_counter() - ap_start}")
# print(query)


ex_start = perf_counter()
infer = VariableElimination(wfh.get_model())

query = infer.query(
    ['work_from_home'],
    evidence=evidence,
    show_progress=False
)
print(f"Exact time: {perf_counter() - ex_start}")
print(query.cardinality)

bn_start = perf_counter()
query = bn.inference.fit(
    wfh_dag,
    variables=['work_from_home'],
    evidence=evidence,
    verbose=0
)

print(f"Bnlearn time: {perf_counter() - bn_start}")
print(query)
