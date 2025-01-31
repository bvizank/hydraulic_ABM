import pandas as pd
import numpy as np
import math
import random
import wntr
import copy
import os
import shutil
import matplotlib.pyplot as plt


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

    return (node_patterns, wn)


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

    return pop_dict


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


def read_comp_data(loc, read_list, days, truncate_list):
    out_dict = dict()
    for item in read_list:
        out_dict['avg_' + item] = pd.read_pickle(loc + 'avg_' + item + '.pkl')
        out_dict['var_' + item] = pd.read_pickle(loc + 'var_' + item + '.pkl')

        if item in truncate_list:
            x_len = days * 24
            out_dict['avg_' + item] = out_dict['avg_' + item].iloc[168:x_len, :]
            out_dict['var_' + item] = out_dict['var_' + item].iloc[168:x_len, :]

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
    data = data['age'].mean(axis=0)
    print(data.mean() / 3600)
    data.to_pickle('hot_start_age_data_2024-03-08_12-10_200days_results.pkl')


def calc_clinton_ind_dists():
    '''
    Calculate the average minimum distance to industrial for each residential
    node in Clinton and average by block group.
    '''
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy
    pd.options.mode.copy_on_write = True
    # import the data the includes residential and industrial nodes and spatial data
    col_names = ['lon', 'lat', 'val', 'struct', 'sec', 'group', 'bg', 'city']
    data = pd.read_csv(
        'Input Files/clinton_data.csv',
        delimiter=',',
        names=col_names
    )

    # convert lat and lon to radians
    data['lat'] = data['lat'] * np.pi / 180
    data['lon'] = data['lon'] * np.pi / 180

    # ind_nodes = data[(data['sec'] == 3) & (data['city'] == 1)]
    # res_nodes = data[(data['sec'] == 1) & (data['city'] == 1)]
    ind_nodes = data[(data['sec'] == 3)]
    res_nodes = data[(data['sec'] == 1)]
    res_nodes = res_nodes[res_nodes.loc[:, 'struct'] == 1]

    # make a dict of industrial parcel locations
    ind_loc = dict()
    for i, row in ind_nodes.iterrows():
        ind_loc[row.name] = (row['lat'], row['lon'])

    for i, key in enumerate(ind_loc):
        '''
        Using haversine formula to calculate distance

        More information found here:
        https://www.movable-type.co.uk/scripts/latlong.html
        '''
        res_nodes = res_nodes.assign(del_lat=res_nodes['lat'] - ind_loc[key][0])
        res_nodes = res_nodes.assign(del_lon=res_nodes['lon'] - ind_loc[key][1])
        res_nodes = res_nodes.assign(a=(
            np.sin(res_nodes['del_lat']/2) ** 2 +
            np.cos(res_nodes['lat']) * np.cos(ind_loc[key][0]) *
            np.sin(res_nodes['del_lon']/2) ** 2
        ))
        res_nodes.loc[:, str(key)] = (
            2 * np.arctan2(
                    np.sqrt(res_nodes['a']), np.sqrt(1 - res_nodes['a'])
                ) *
            6371000
        )

    col_names.extend(['del_lat', 'del_lon', 'a'])
    res_nodes.loc[:, 'min'] = res_nodes.loc[:, ~res_nodes.columns.isin(col_names)].min(axis=1)

    return res_nodes


def delete_contents(loc):
    '''
    Delete everything in provided file location.

    parameters:
        loc   (str): location to delete files and folders
    '''

    for filename in os.listdir(loc):
        file_path = os.path.join(loc, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def income_list(data, n_house, model, s=None):
    '''
    Create a list of incomes that is representative of the data
    provided and containing >= n_house values.

    parameters:
    ----------
        data   (dict): income data formatted as key: income bracket,
                       value: percentage of population in income bracket
        n_house (int): size of synthetic dataset (number of households)
        s       (int): seed for RNG
    '''

    # if the seed exists, set the seed
    if s:
        random.seed(s)

    income = list()
    index = 0
    for i, key in enumerate(data):
        '''
        Get a set of samples for the given income range that is 100
        times longer than the percentage given by the data.
        e.g. for $0 - $10,000, we want 76 samples uniformly distributed
        between 0 and 10000.
        '''
        if i != (len(data) - 1):
            for j in range(math.ceil(data[key]/1000*n_house)):
                income.append(model.random.uniform(
                    list(data.keys())[i], list(data.keys())[i+1]
                ))
        else:
            # at the end we need to arbitrarily set an upper bound
            for j in range(math.ceil(data[key]/1000*n_house)):
                income.append(model.random.uniform(
                    list(data.keys())[i], list(data.keys())[i]*3
                ))
        index += data[key]

    # shuffle the income list
    model.random.shuffle(income)

    # convert the income list to a numpy array
    # income = np.array(income)

    return income
