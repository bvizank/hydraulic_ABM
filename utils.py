import pandas as pd
import numpy as np
import math
import wntr
import copy
import os
import matplotlib.pyplot as plt


def setup(network):
    # Create a water network model
    if network == "micropolis":
        inp_file = 'Input Files/micropolis/MICROPOLIS_v1_inc_rest_consumers.inp'
        data = pd.read_excel(r'Input Files/micropolis/Micropolis_pop_at_node.xlsx')
    elif network == "mesopolis":
        inp_file = 'Input Files/mesopolis/Mesopolis.inp'
        data = pd.read_excel(r'Input Files/mesopolis/Mesopolis_pop_at_node.xlsx')
    base_demands, pattern_list, wn = init_wntr(inp_file)

    # input the number of agents required at each node type at each time
    node_id = data['Node'].tolist()
    maxpop_node = data['Max Population'].tolist()
    if network == "mesopolis":
        house_num = data['HOUSE'].tolist()
        # create dictionary with the number of houses per node
        house_num = dict(zip(node_id, house_num))
    else:
        house_num = None

    # Creating dictionary with max pop at each terminal node
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
        node_dict['cafe'] = find_nodes(1, pattern_list, network)
        # residential nodes
        node_dict['res'] = find_nodes(2, pattern_list, network)
        # Industrial nodes
        node_dict['ind'] = find_nodes(3, pattern_list, network)
        # Nodes dairy queen
        node_dict['dq'] = find_nodes(4, pattern_list, network)
        # Rest of commercial nodes like schools, churches etc.
        node_dict['com'] = find_nodes(5, pattern_list, network)
        node_dict['com'] = node_dict['com'] + find_nodes(6, pattern_list, network)
    elif network == "mesopolis":
        # pattern types: air, com, res, ind, nav
        node_dict['air'] = find_nodes('air', pattern_list, network)
        node_dict['com'] = find_nodes('com', pattern_list, network)
        node_dict['res'] = find_nodes('res', pattern_list, network)
        node_dict['ind'] = find_nodes('ind', pattern_list, network)
        node_dict['nav'] = find_nodes('nav', pattern_list, network)
        node_dict['cafe'] = find_nodes('cafe', pattern_list, network)

    terminal_nodes = list()
    for key in node_dict:
        terminal_nodes += node_dict[key]

    # finish setup process by loading distributions of agents at each node type,
    # media data, and the distance between residential nodes and closest ind.
    # node.
    pop_dict = load_distributions(network)
    ind_node_dist, na = calc_industry_distance(
        wn, node_dict['ind'], nodes=node_dict['res'])

    return (node_dict, node_capacity, house_num, pop_dict,
            terminal_nodes, wn, ind_node_dist)


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
        if node in dem_nodes:
            time_list = junction.demand_timeseries_list[0]
            # get base demand value for each node with a demand
            lst_base_demands[node] = copy.deepcopy(time_list.base_value)

            # Find demand patterns for each node
            node_patterns[node] = copy.deepcopy(time_list.pattern_name)

    return (lst_base_demands, node_patterns, wn)


def find_nodes(type, pattern_list, network):
    output = list()
    for k, v in pattern_list.items():
        if v != 'DefPat' and network == "micropolis":
            # print(f"v: {v}; type: {type}; are they equal {int(v) == int(type)}")
            if int(v) == int(type) and k[0:2] == 'TN':
                output.append(k)
            else:
                continue
        elif network == "mesopolis":
            if v == type and k[0:2] == 'TN':
                output.append(k)

    return output


def load_distributions(network):
    '''
    Load in the data on how many agents are supposed to be at each node type
    at each time of the day.

    THIS IS DEPENDENT ON THE NETWORK
    '''
    # Import table for timesteps per nodetypes per hour
    if network == "micropolis":
        node_pop = pd.read_excel(r'Input Files/micropolis/Micropolis_pop_at_node_kind.xlsx')
    elif network == "mesopolis":
        node_pop = pd.read_excel(r'Input Files/micropolis/Mesopolis_pop_at_node_kind.xlsx')

    pop_dict = dict()

    pop_dict['ind'] = node_pop['indust.'].tolist()
    pop_dict['res'] = node_pop['resident'].tolist()
    pop_dict['com'] = node_pop['comm.'].tolist()
    pop_dict['cafe'] = node_pop['rest.'].tolist()
    if network == "mesopolis":
        pop_dict['nav'] = node_pop['navy'].tolist()
        pop_dict['air'] = node_pop['air'].tolist()
    pop_dict['sum'] = node_pop['sum'].tolist()

    # Example residents per residential node at first timestep
    #lst_t1_resident_pn = {}
    # Calculating percentage of node "capacity
    # Resident_percentage = np.array(pop_dict['res'])/np.array(pop_dict['sum'])
    # Industr_percentage = np.array(pop_dict['ind'])/np.array(pop_dict['sum'])
    # Comm_percentage = np.array(pop_dict['cafe'])/np.array(pop_dict['sum'])
    # Comm_rest_percentage = np.array(pop_dict['com'])/np.array(pop_dict['sum'])

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


def calc_industry_distance(wn, ind_nodes, nodes=None):
    '''
    Function to calculate the distance to the nearest industrial node.
    '''
    if nodes is None:
        nodes = [name for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name != '3']

    ind_distances = dict()
    close_node = dict()
    for node in nodes:
        curr_node = wn.get_node(node)
        curr_node_dis = dict()
        for ind_node in ind_nodes:
            curr_ind_node = wn.get_node(ind_node)
            curr_node_dis[ind_node] = calc_distance(curr_node, curr_ind_node)
        # find the key with the min value, i.e. the node with the lowest distance
        ind_distances[node] = min(curr_node_dis.values())
        close_node[node] = min(curr_node_dis, key=curr_node_dis.get)

    return (ind_distances, close_node)


def calc_distance(node1, node2):
    p1x, p1y = node1.coordinates
    p2x, p2y = node2.coordinates

    return math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)


# def read_comp_data(loc, read_list, error):
#     out_dict = dict()
#     for item in read_list:
#         out_dict['avg_'+item] = pd.read_pickle(loc + 'avg_' + item + '.pkl')
#         out_dict['sd_'+item] = pd.read_pickle(loc + 'sd_' + item + '.pkl')
#         if error == 'ci95':
#             out_dict['sd_'+item] = out_dict['sd_'+item] * 1.96 / math.sqrt(30)
#         elif error == 'se':
#             out_dict['sd_'+item] = out_dict['sd_'+item] / math.sqrt(30)
#         else:
#             pass

#     return out_dict


# def read_data(loc, read_list, data_file=None):
#     ''' Function to read in data from either excel or pickle '''
#     output = dict()
#     # data_file = loc + 'datasheet.xlsx'
#     pkls = [file for file in os.listdir(loc) if file.endswith(".pkl")]

#     for name in read_list:
#         index_col = 0
#         print("Reading " + name + " data")
#         if name == 'seir':
#             sheet_name = 'seir_data'
#             index_col = 1
#         elif name == 'agent':
#             sheet_name = 'agent locations'
#         else:
#             sheet_name = name

#         file_name = name + '.pkl'
#         if file_name not in pkls:
#             print("No pickle file found, importing from excel")
#             locals()[name] = pd.read_excel(data_file,
#                                            sheet_name=sheet_name,
#                                            index_col=index_col)
#             if name == 'seir':
#                 locals()[name].index = locals()[name].index.astype("int64")

#             locals()[name].to_pickle(loc + file_name)
#         else:
#             print("Pickle file found, unpickling")
#             locals()[name] = pd.read_pickle(loc + name + '.pkl')

#         output[name] = locals()[name]

#     return output


def read_comp_data(loc, read_list):
    out_dict = dict()
    for item in read_list:
        out_dict['avg_' + item] = pd.read_pickle(loc + 'avg_' + item + '.pkl')
        out_dict['var_' + item] = pd.read_pickle(loc + 'var_' + item + '.pkl')

    return out_dict


def read_data(loc, read_list):
    out_dict = dict()
    for item in read_list:
        out_dict[item] = pd.read_pickle(loc + item + '.pkl')

    return out_dict


def calc_error(var_data, error):
    '''
    Calculate the appropriate error values given the variance values

    parameter:
        var_data   (pd.DataFrame): variance data
        error               (str): the error to calculate

    '''

    std_data = np.sqrt(var_data)
    if error == 'ci95':
        output = std_data * 1.96 / math.sqrt(30)
    elif error == 'se':
        output = std_data / math.sqrt(30)
    elif error == 'sd':
        output = std_data
    else:
        raise NotImplementedError(f"{error} is not yet implemented.")

    return output


def clean_epanet(folder):
    '''
    Clean all input, bin, and rpt files from given folder.
    '''

    files = os.listdir(folder)
    for file in files:
        if file.endswith('.bin') or \
           file.endswith('.rpt') or \
           file.endswith('.inp'):
            os.remove(os.path.join(folder, file))


def make_lorenz(data):
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    scaled_prefix_sum = data.cumsum() / data.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    x_vals = np.insert(scaled_prefix_sum, 0, 0)
    fig = plt.figure()
    # plot the straight line perfect equality curve
    plt.plot([0, 1], [0, 1])
    # we need the X values to be between 0.0 to 1.0
    plt.plot(np.linspace(0.0, 1.0, x_vals.size), x_vals)

    return fig


def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def output_age_data(file):
    loc = 'Output Files/' + file + '/'
    data = read_data(loc, ['age'])
    data = data['age'].iloc[-100:].mean(axis=0)
    print(data)
    print(data.mean() / 3600)
    data.to_pickle('hot_start_age_data_2024-03-08_12-10_200days_results.pkl')
