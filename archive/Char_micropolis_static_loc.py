import pandas as pd
import numpy as np
import wntr
import copy


def setup(network):
    # Create a water network model
    if network == "micropolis":
        inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
        data = pd.read_excel(r'Input Files/Micropolis_pop_at_node.xlsx')
    elif network == "mesopolis":
        inp_file = 'Input Files/Mesopolis.inp'
        data = pd.read_excel(r'Input Files/Mesopolis_pop_at_node.xlsx')
    base_demands, pattern_list = init_wntr(inp_file)

    # input the number of agents required at each node type at each time
    node_id = data['Node'].tolist()
    maxpop_node = data['Max Population'].tolist()
    if network == "mesopolis":
        house_num = data['HOUSE'].tolist()
    else:
        house_num = None

    # Creating dictionary with max pop at each terminal (resid?) node

    node_capacity = dict(zip(node_id, maxpop_node))

    node_dict = dict()
    if network == "micropolis":
        # Node kinds are:(Pattern number - Kind of node)
        # 1: Commercial – Cafe
        # 2: Residential
        # 3: Industrial, factory with 3 shifts
        # 4: Commercial – Dairy Queen
        # 5: Commercial – Curches, schools, city hall, post office
        # Only terminal nodes count. So only nodes with prefix 'TN'

        # Cafe nodes (only 1)
        node_dict['cafe'] = find_nodes(1, pattern_list)
        # residential nodes
        node_dict['res'] = find_nodes(2, pattern_list)
        # Industrial nodes
        node_dict['ind'] = find_nodes(3, pattern_list)
        # Nodes dairy queen
        node_dict['dq'] = find_nodes(4, pattern_list)
        # Rest of commercial nodes like schools, churches etc.
        node_dict['com'] = find_nodes(5, pattern_list)
    elif network == "mesopolis":
        # pattern types: air, com, res, ind, nav
        node_dict['air'] = find_nodes('air', pattern_list)
        node_dict['com'] = find_nodes('com', pattern_list)
        node_dict['res'] = find_nodes('res', pattern_list)
        node_dict['ind'] = find_nodes('ind', pattern_list)
        node_dict['nav'] = find_nodes('nav', pattern_list)

    # finish setup process by loading distributions of agents at each node type
    # and media data.
    pop_dict = load_distributions(network)
    media, bbn_params, wfh_patterns = load_media()

    return (node_dict, node_capacity, house_num, pop_dict, media, bbn_params,
            wfh_patterns)


def init_wntr(inp_file):
    # initialize the water network model with wntr
    wn = wntr.network.WaterNetworkModel(inp_file)

    # List all nodes and all nodes with demand
    lst_nodes = wn.node_name_list
    dem_nodes = [node for node in lst_nodes if hasattr(wn.get_node(node), 'demand_timeseries_list')]
    # creating list of base demands
    lst_base_demands = dict()
    node_patterns = dict()
    for node in lst_nodes:
        junction = wn.get_node(node)
        time_list = junction.demand_timeseries_list[0]
        if node in dem_nodes:
            # get base demand value for each node with a demand
            lst_base_demands[node] = copy.deepcopy(time_list.base_value)

        # Find demand patterns for each node
        node_patterns[node] = copy.deepcopy(time_list.pattern_name)

    return (lst_base_demands, node_patterns)


def find_nodes(type, pattern_list):
    output = list()
    for k, v in pattern_list.items():
        if v == type and k[0:2] == 'TN':
            output.append(k)
        else:
            continue

    return output


def load_distributions(network):
    '''
    Load in the data on how many agents are supposed to be at each node type
    at each time of the day.

    THIS IS DEPENDENT ON THE NETWORK
    '''
    # Import table for timesteps per nodetypes per hour
    if network == "micropolis":
        node_pop = pd.read_excel(r'Input Files/Micropolis_pop_at_node_kind.xlsx')
    elif network == "mesopolis":
        node_pop = pd.read_excel(r'Input Files/Mesopolis_pop_at_node_kind.xlsx')

    pop_dict = dict()

    pop_dict['ind'] = node_pop['indust.'].tolist()
    pop_dict['res'] = node_pop['resident'].tolist()
    pop_dict['res'] = node_pop['rest.'].tolist()
    pop_dict['com'] = node_pop['comm.'].tolist()
    pop_dict['sum'] = node_pop['sum'].tolist()

    # Example residents per residential node at first timestep
    #lst_t1_resident_pn = {}
    # Calculating percentage of node "capacity
    Resident_percentage = np.array(pop_dict['res'])/np.array(pop_dict['sum'])
    Industr_percentage = np.array(pop_dict['ind'])/np.array(pop_dict['sum'])
    Comm_percentage = np.array(pop_dict['cafe'])/np.array(pop_dict['sum'])
    Comm_rest_percentage = np.array(pop_dict['com'])/np.array(pop_dict['sum'])

    return (pop_dict)


def load_clearance():
    # Clearance of nodes for every iteration step
    Cleared_nodes_iterations = pd.read_excel(r'Input Files/Cleared_node_names.xlsx')
    Cleared_nodes_iteration_1 = Cleared_nodes_iterations['Iteration_1'].tolist()
    Cleared_nodes_iteration_2 = Cleared_nodes_iterations['Iteration_2'].tolist()
    Cleared_nodes_iteration_2 = [i for i in Cleared_nodes_iteration_2 if type(i) is str]
    Cleared_nodes_iteration_3 = Cleared_nodes_iterations['Iteration_3'].tolist()
    Cleared_nodes_iteration_3 = [i for i in Cleared_nodes_iteration_3 if type(i) is str]
    Cleared_nodes_iteration_4 = Cleared_nodes_iterations['Iteration_4'].tolist()
    Cleared_nodes_iteration_4 = [i for i in Cleared_nodes_iteration_4 if type(i) is str]
    Cleared_nodes_iteration_5 = Cleared_nodes_iterations['Iteration_5'].tolist()
    Cleared_nodes_iteration_5 = [i for i in Cleared_nodes_iteration_5 if type(i) is str]
    Cleared_nodes_iteration_6 = Cleared_nodes_iterations['Iteration_6'].tolist()
    Cleared_nodes_iteration_6 = [i for i in Cleared_nodes_iteration_6 if type(i) is str]

    Cleared_nodes_all = Cleared_nodes_iteration_1
    Cleared_nodes_all.extend(Cleared_nodes_iteration_2)
    Cleared_nodes_all.extend(Cleared_nodes_iteration_3)
    Cleared_nodes_all.extend(Cleared_nodes_iteration_4)
    Cleared_nodes_all.extend(Cleared_nodes_iteration_5)
    Cleared_nodes_all.extend(Cleared_nodes_iteration_6)

    # All cleared Terminal Nodes

    Cleared_nodes_terminal = []

    for node in Cleared_nodes_all:
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


def load_media():
    '''
    Load in the necessary media data including sleep, TV, and radio data
    Also load the BBN parameters from Dryhurst et al. 2020 and the Lakewood
    COVID-19 residential patterns.

    THIS DATA IS NOT DEPENDENT ON NETWORK
    '''

    # Load TV, RADIO, SLEEP data
    sleep_data = pd.read_excel(r'Input Files/sleep_data.xlsx')
    sleep_distr = sleep_data['sleep_data'].tolist()

    radio_data = pd.read_excel(r'Input Files/Radio_data.xlsx')
    radio_distr = radio_data['radio_data'].tolist()

    tv_data = pd.read_excel(r'Input Files/TV_data.xlsx')
    tv_distr = tv_data['tv_data'].tolist()

    media = {'sleep': sleep_distr,
             'radio': radio_distr,
             'tv': tv_distr}

    # Load agent parameters for BBN predictions
    bbn_params = pd.read_csv(r'Input Files/all_bbn_data.csv')

    # Load in the new residential patterns from Lakewood data
    wfh_patterns = pd.read_csv(r'Input Files/res_patterns/normalized_res_patterns.csv')

    return (media, bbn_params, wfh_patterns)
