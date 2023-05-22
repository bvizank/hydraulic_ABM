# from Char_micropolis import *
import wntr
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", UserWarning)

# Create a water network model
inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
# MICROPOLIS_v1_orig_consumers.inp
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
# wntr.graphics.plot_network(wn, title=wn.name)
# plt.show()

#Simulate hydraulics
#wn.options.time.duration = 23*3600
#sim = wntr.sim.EpanetSimulator(wn)
#results = sim.run_sim()

# Create graph
#G = wn.get_graph() # directed multigraph

# List of nodes
lst_nodes = wn.node_name_list
# creating list of base demands
lst_base_demands = []
for node in lst_nodes:
    junction = wn.get_node(node)
    try:
        lst_base_demands.append(float(junction.base_demand))
    except:
        lst_base_demands.append(0) # In case its a source like e.g. "River" (so a string), the demand will be set to zero
