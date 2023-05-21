import pandas as pd
import numpy as np
import numba as nb
import os
import wntr
import copy


__all__ = ['setup', 'read_data', 'choose']


def setup(network):
    # Create a water network model
    if network == "micropolis":
        inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
        data = pd.read_excel(r'Input Files/Micropolis_pop_at_node.xlsx')
    elif network == "mesopolis":
        inp_file = 'Input Files/Mesopolis.inp'
        data = pd.read_excel(r'Input Files/Mesopolis_pop_at_node.xlsx')
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

    # finish setup process by loading distributions of agents at each node type
    # and media data.
    pop_dict = load_distributions(network)
    media, bbn_params, wfh_patterns = load_media()

    return (node_dict, node_capacity, house_num, pop_dict, media, bbn_params,
            wfh_patterns, terminal_nodes, wn)


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
        node_pop = pd.read_excel(r'Input Files/Micropolis_pop_at_node_kind.xlsx')
    elif network == "mesopolis":
        node_pop = pd.read_excel(r'Input Files/Mesopolis_pop_at_node_kind.xlsx')

    pop_dict = dict()

    pop_dict['ind'] = node_pop['indust.'].to_numpy()
    pop_dict['res'] = node_pop['resident'].to_numpy()
    pop_dict['com'] = node_pop['comm.'].to_numpy()
    pop_dict['cafe'] = node_pop['rest.'].to_numpy()
    if network == "mesopolis":
        pop_dict['nav'] = node_pop['navy'].to_numpy()
        pop_dict['air'] = node_pop['air'].to_numpy()
    pop_dict['sum'] = node_pop['sum'].to_numpy()

    return (pop_dict)


def load_media():
    '''
    Load in the necessary media data including sleep, TV, and radio data
    Also load the BBN parameters from Dryhurst et al. 2020 and the Lakewood
    COVID-19 residential patterns.

    THIS DATA IS NOT DEPENDENT ON NETWORK
    '''

    # Load TV, RADIO, SLEEP data
    sleep_data = pd.read_excel(r'Input Files/sleep_data.xlsx')
    sleep_distr = sleep_data['sleep_data'].to_numpy()

    radio_data = pd.read_excel(r'Input Files/Radio_data.xlsx')
    radio_distr = radio_data['radio_data'].to_numpy()

    tv_data = pd.read_excel(r'Input Files/TV_data.xlsx')
    tv_distr = tv_data['tv_data'].to_numpy()

    media = {'sleep': sleep_distr,
             'radio': radio_distr,
             'tv': tv_distr}

    # Load agent parameters for BBN predictions
    bbn_params = pd.read_csv(r'Input Files/all_bbn_data.csv')

    # Load in the new residential patterns from Lakewood data
    wfh_patterns = pd.read_csv(r'Input Files/res_patterns/normalized_res_patterns.csv')

    return (media, bbn_params, wfh_patterns)


def read_data(loc, read_list):
    ''' Function to read in data from either excel or pickle '''
    output = dict()
    data_file = loc + 'datasheet.xlsx'
    pkls = [file for file in os.listdir(loc) if file.endswith(".pkl")]

    for name in read_list:
        index_col = 0
        print("Reading " + name + " data")
        if name == 'seir':
            sheet_name = 'seir_data'
            index_col = 1
        elif name == 'agent':
            sheet_name = 'agent locations'
        else:
            sheet_name = name

        file_name = name + '.pkl'
        if file_name not in pkls:
            print("No pickle file found, importing from excel")
            locals()[name] = pd.read_excel(data_file,
                                           sheet_name=sheet_name,
                                           index_col=index_col)
            if name == 'seir':
                locals()[name].index = locals()[name].index.astype("int64")

            locals()[name].to_pickle(loc + file_name)
        else:
            print("Pickle file found, unpickling")
            locals()[name] = pd.read_pickle(loc + name + '.pkl')

        output[name] = locals()[name]

    return output


@nb.njit((nb.int32, nb.int32)) # Numba hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose(5, 2) # choose 2 out of 5 people with equal probability (without repeats)
    '''

    return np.random.choice(max_n, n, replace=False)


def sample(dist=None, par1=None, par2=None, size=None, **kwargs):
    '''
    Draw a sample from the distribution specified by the input. The available
    distributions are:

    - 'uniform'       : uniform distribution from low=par1 to high=par2; mean is equal to (par1+par2)/2
    - 'normal'        : normal distribution with mean=par1 and std=par2
    - 'lognormal'     : lognormal distribution with mean=par1 and std=par2 (parameters are for the lognormal distribution, *not* the underlying normal distribution)
    - 'normal_pos'    : right-sided normal distribution (i.e. only positive values), with mean=par1 and std=par2 *of the underlying normal distribution*
    - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
    - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
    - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
    - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞

    Args:
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)
        kwargs (dict): passed to individual sampling functions

    Returns:
        A length N array of samples

    **Examples**::

        cv.sample() # returns Unif(0,1)
        cv.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
        cv.sample(dist='lognormal_int', par1=5, par2=3) # returns a lognormally distributed set of values with mean 5 and std 3

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
        function assumes the user wants to specify the mean and std of the lognormal distribution.

        Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
        (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
        distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
        of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
        large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
        the mean.
    '''

    # Some of these have aliases, but these are the "official" names
    choices = [
        'uniform',
        'normal',
        'normal_pos',
        'normal_int',
        'lognormal',
        'lognormal_int',
    ]

    # Ensure it's an integer
    if size is not None:
        size = int(size)

    # Compute distribution parameters and draw samples
    # NB, if adding a new distribution, also add to choices above
    if   dist in ['unif', 'uniform']: samples = np.random.uniform(low=par1, high=par2, size=size, **kwargs)
    elif dist in ['norm', 'normal']:  samples = np.random.normal(loc=par1, scale=par2, size=size, **kwargs)
    elif dist == 'normal_pos':        samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs))
    elif dist == 'normal_int':        samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs)))
    elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
        if par1 > 0:
            mean = np.log(par1**2 / np.sqrt(par2**2 + par1**2)) # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size, **kwargs)
        else:
            samples = np.zeros(size)
        if '_int' in dist:
            samples = np.round(samples)
    else:
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
        raise NotImplementedError(errormsg)

    return samples

