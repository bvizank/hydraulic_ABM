from wntr_1 import *
import re
import pandas as pd
import numpy as np
import math

#Population Micropolis:
Micro_pop = 4606

# Find demand patterns for each node
node_patterns = {}

for node in lst_nodes:
    junc = wn.get_node(node)
    try:
        f = junc.demand_timeseries_list
        f1 = str(f)
        s = re.findall(r'\d+', f1)
        pattern = int(s[-1])
        node_patterns[node] = pattern
    except:
        node_patterns[node] = 'None'

# Next: Max amount of Agents per nodes.

data = pd.read_excel(r'Input Files/Micropolis_pop_at_node.xlsx')
Micro_node_id = data['Node'].tolist()
Micro_maxpop_node = data['Max Population'].tolist()

#Creating dictionary with max pop at each terminal (resid?) node

Max_pop_pnode_resid = dict(zip(Micro_node_id,Micro_maxpop_node))

#Node kinds are:(Pattern number - Kind of node)
#1: Commercial – Cafe
#2: Residential
#3: Industrial, factory with 3 shifts
#4: Commercial – Dairy Queen
#5: Commercial – Curches, schools, city hall, post office
# Only terminal nodes count. So only nodes with prefix 'TN'

# Cafe nodes (only 1)
Nodes_comm_cafe = []
for k,v in node_patterns.items():
    if v == 1 and k[0:2]== 'TN':
        Nodes_comm_cafe.append(k)
    else:
        continue

# residential nodes
Nodes_resident = []
for k,v in node_patterns.items():
    if v == 2 and k[0:2]== 'TN':
        Nodes_resident.append(k)
    else:
        continue

# Industrial nodes
Nodes_industr = []
for k,v in node_patterns.items():
    if v == 3 and k[0:2]== 'TN':
        Nodes_industr.append(k)
    else:
        continue

# Nodes dairy queen
Nodes_comm_dairy = []
for k,v in node_patterns.items():
    if v == 4 and k[0:2]== 'TN':
        Nodes_dairy.append(k)
    else:
        continue

# Rest of commercial nodes like schools, churches etc.
Nodes_comm_rest = []

for k,v in node_patterns.items():
    if v == 5 and k[0:2]== 'TN':
        Nodes_comm_rest.append(k)
    else:
        continue

terminal_nodes = len(Nodes_comm_rest)+len(Nodes_comm_dairy)+len(Nodes_industr)+len(Nodes_resident)+1
All_terminal_nodes = Nodes_comm_rest+ Nodes_comm_dairy +Nodes_industr + Nodes_resident + Nodes_comm_cafe

# Export the various lists
#data = ','.join(Nodes_resident)
#fh = open('myfile.csv', 'w+')
#fh.write(data)
#fh.close()


#Import table for timesteps per nodetypes per hour
Nodes_distribution_ph = pd.read_excel(r'Input Files/Micropolis_pop_at_node_kind.xlsx')
#print(Nodes_distribution_ph)
Industr_distr_ph = Nodes_distribution_ph['indust.'].tolist()
Resident_distr_ph = Nodes_distribution_ph['resident'].tolist()
Comm_rest_distr_ph = Nodes_distribution_ph['rest.'].tolist()
Comm_distr_ph = Nodes_distribution_ph['comm.'].tolist()
Sum_distr_ph = Nodes_distribution_ph['sum'].tolist()


# Example residents per residential node at first timestep
#lst_t1_resident_pn = {}
# Calculating percentage of node "capacity
Resident_percentage = np.array(Resident_distr_ph)/np.array(Sum_distr_ph)
Industr_percentage= np.array(Industr_distr_ph)/np.array(Sum_distr_ph)
Comm_percentage = np.array(Comm_distr_ph)/np.array(Sum_distr_ph)
Comm_rest_percentage = np.array(Comm_rest_distr_ph)/np.array(Sum_distr_ph)

#print(Resident_percentage)


## Clearance of nodes for every iteration step
Cleared_nodes_iterations = pd.read_excel(r'Input Files/Cleared_node_names.xlsx')
Cleared_nodes_iteration_1 = Cleared_nodes_iterations['Iteration_1'].tolist()
Cleared_nodes_iteration_2 = Cleared_nodes_iterations['Iteration_2'].tolist()
Cleared_nodes_iteration_2= [i for i in Cleared_nodes_iteration_2 if type(i) is str]
Cleared_nodes_iteration_3 = Cleared_nodes_iterations['Iteration_3'].tolist()
Cleared_nodes_iteration_3= [i for i in Cleared_nodes_iteration_3 if type(i) is str]
Cleared_nodes_iteration_4 = Cleared_nodes_iterations['Iteration_4'].tolist()
Cleared_nodes_iteration_4= [i for i in Cleared_nodes_iteration_4 if type(i) is str]
Cleared_nodes_iteration_5 = Cleared_nodes_iterations['Iteration_5'].tolist()
Cleared_nodes_iteration_5= [i for i in Cleared_nodes_iteration_5 if type(i) is str]
Cleared_nodes_iteration_6 = Cleared_nodes_iterations['Iteration_6'].tolist()
Cleared_nodes_iteration_6= [i for i in Cleared_nodes_iteration_6 if type(i) is str]

Cleared_nodes_all = Cleared_nodes_iteration_1
Cleared_nodes_all.extend(Cleared_nodes_iteration_2)
Cleared_nodes_all.extend(Cleared_nodes_iteration_3)
Cleared_nodes_all.extend(Cleared_nodes_iteration_4)
Cleared_nodes_all.extend(Cleared_nodes_iteration_5)
Cleared_nodes_all.extend(Cleared_nodes_iteration_6)

## All cleared Terminal Nodes

Cleared_nodes_terminal = []

for node in  Cleared_nodes_all:
    if node[0:2] == 'TN':
        Cleared_nodes_terminal.append(node)
    else:
        continue

Endangered_nodes_terminal = []

for node in All_terminal_nodes:
    if node in Cleared_nodes_terminal:
        continue
    else:
        Endangered_nodes_terminal.append(node)



#Load TV, RADIO, SLEEP data

Sleep_data = pd.read_excel(r'Input Files/sleep_data.xlsx')
Sleep_distr = Sleep_data['sleep_data'].tolist()

radio_data = pd.read_excel(r'Input Files/Radio_data.xlsx')
radio_distr = radio_data['radio_data'].tolist()

TV_data = pd.read_excel(r'Input Files/TV_data.xlsx')
TV_distr = TV_data['tv_data'].tolist()

# Load agent parameters for BBN predictions
bbn_params = pd.read_csv(r'Input Files/all_bbn_data.csv')

# Load in the new residential patterns from Lakewood data
wfh_patterns = pd.read_csv(r'Input Files/res_patterns/normalized_res_patterns.csv')
