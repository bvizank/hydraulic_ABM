import wntr
import pickle

inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
wn.options.quality.parameter = 'AGE'

f = open('wn.pickle', 'wb')
pickle.dump(wn, f)
f.close()

sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

print(results.node['quality'])

f = open('wn.pickle', 'rb')
wn2 = pickle.load(f)
f.close()

sim = wntr.sim.EpanetSimulator(wn2)
results = sim.run_sim()

print(results.node['quality'])
