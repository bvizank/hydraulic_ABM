from mesa import Model
import pandas as pd
import utils as ut
import city_info as ci
import data as dt
import networkx as nx
import bnlearn as bn
import wntr
import os
from mesa.time import RandomActivation
from mesa.space import NetworkGrid


class Parameters(Model):
    '''
    Create the list of parameters that will be used for the given
    simulation.
    '''

    def __init__(self, N, city, days, id, seed, **kwargs):
        super().__init__()
        ''' Set the default parameters '''
        self.days = days
        self.id = id
        self.num_agents = N
        self.network = city
        self.t = 0
        self.schedule = RandomActivation(self)
        self.timestep = 0
        self.timestep_day = 0
        self.timestepN = 0
        # self.base_demands_previous = {}
        self.swn = nx.watts_strogatz_graph(
            n=self.num_agents, p=0.2, k=6, seed=seed
        )
        self.snw_agents = {}
        # self.nodes_endangered = All_terminal_nodes
        self.demand_test = []

        ''' COVID-19 exposure pars '''
        self.exposure_rate = 0.05  # infection rate per contact per day in households
        self.exposure_rate_large = 0.01  # infection rate per contact per day in workplaces
        self.e2i = (4.5, 1.5)  # mean and sd number of days before infection shows
        self.i2s = (1.1, 0.9)  # time after viral shedding before individual shows sypmtoms
        self.s2sev = (6.6, 4.9)  # time after symptoms start before individual develops potential severe covid
        self.sev2c = (1.5, 2.0)  # time after severe symptoms before critical status
        self.c2d = (10.7, 4.8)  # time between critical dianosis to death
        self.recTimeAsym = (8.0, 2.0)  # time for revovery for asymptomatic cases
        self.recTimeMild = (8.0, 2.0)  # mean and sd number of days for recovery: mild cases
        self.recTimeSev = (18.1, 6.3)
        self.recTimeC = (18.1, 6.3)

        self.covid_exposed = int(0.001 * self.num_agents)
        self.cumm_infectious = self.covid_exposed
        self.daily_contacts = 10
        self.verbose = 1

        self.res_pat_select = 'lakewood'
        self.wfh_lag = 0
        self.no_wfh_perc = 0.5
        self.wfh_thres = False  # whether wfh lag has been reached

        self.bbn_models = ['wfh', 'dine', 'grocery', 'ppe']

        ''' Import the four DAGs for the BBNs '''
        self.wfh_dag = bn.import_DAG('Input Files/data_driven_models/work_from_home.bif', verbose=0)
        self.dine_less_dag = bn.import_DAG('Input Files/pmt_models/dine_out_less_pmt-6.bif', verbose=0)
        self.grocery_dag = bn.import_DAG('Input Files/pmt_models/shop_groceries_less_pmt-6.bif', verbose=0)
        self.ppe_dag = bn.import_DAG('Input Files/data_driven_models/mask.bif', verbose=0)

        '''
        This value comes from this paper:
        https://www.cdc.gov/mmwr/volumes/71/wr/mm7106e1.htm#T3_down

        Odds that you will get covid if you wear a mask is
        66% less likely than without a mask, therefore, the new
        exposure rate is 34% of the original.
        '''
        self.ppe_reduction = 0.34

        '''
        hyd_sim represents the way the hydraulic simulation is to take place
        options are 'eos' (end of simulation) and 'hourly' with the default 'eos'
        '''
        self.hyd_sim = 'eos'
        '''
        The warmup input dictates whether a warmup period is run to reach steady
        state water age values. Default is true
        '''
        self.warmup = True
        '''
        Warmup tolerance is the threshold for when warmup period is complete
        '''
        self.tol = 0.001
        '''
        bw dictates whether bottled water buying is modeled. Defaults to True
        '''
        self.bw = True
        '''
        twa_mods are the modulators to the twa thresholds passed to households
        '''
        self.twa_mods = [130, 140, 150]
        '''
        self.ind_min_demand is the minimum industrial demand as a percentage
        '''
        self.ind_min_demand = 0
        '''
        dist_income is whether income is industrial distance based
        '''
        self.dist_income = True
        '''
        twa_process dictates whether the twas are represented as absolute
        reductions or percentage reductions

        two options are absolute and percentage
        '''
        self.twa_process = 'absolute'

        ''' Data collectors '''
        self.agent_matrix = dict()

        ''' Initialize the COVID state variable collectors '''
        self.cov_pers = dict()
        self.cov_ff = dict()
        self.media_exp = dict()

        ''' Initialize the PM adoption collectors '''
        self.wfh_dec = dict()
        self.dine_dec = dict()
        self.groc_dec = dict()
        self.ppe_dec = dict()

        ''' Initialize household income and COW collectors '''
        self.bw_cost = dict()
        self.tw_cost = dict()
        self.bw_demand = dict()
        self.traditional = dict()
        self.burden = dict()

        self.hygiene = dict()
        self.drink = dict()
        self.cook = dict()

        ''' Initialize status collectors for COVID-19 '''
        status_tot = [
            0,
            self.num_agents-self.covid_exposed,
            0,
            self.covid_exposed,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            self.cumm_infectious,
            len(self.agents_wfh())
        ]
        status_tot = [i / self.num_agents for i in status_tot]
        self.status_tot = {0: status_tot}

        ''' Update the above parameters with the supplied kwargs '''
        self.update_pars(**kwargs)

    def update_pars(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                if key == 'bbn_models' and value == 'all':
                    value = ['wfh', 'dine', 'grocery', 'ppe']
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid attribute")

    def setup_real(self, name, wn_name=None):
        '''
        Import the required data for a real city with a synthetic WDN

        1. Import the water network model using WNTR
        2. Import the demand patterns for residential, cafe, commercial,
           and industrial nodes.
        3. Import the relative distribution of agents amongst node types.
        4. Assign each building a node in the WDN.
        5. Instantiate a household for each residential building.

        Parameters:
        -----------
            name :: str
                The name of the city. There is logic to weed out names
                without data.
        '''
        city_dir = os.path.join('Input Files/cities', name)

        ''' Import the water network model using WNTR '''
        if wn_name:
            inp_file = os.path.join(city_dir, wn_name + '.inp')
            if not os.path.exists(inp_file):
                raise ValueError(f"File {wn_name + '.inp'} does not exist.")
        else:
            inp_file = os.path.join(city_dir, name + '.inp')
        self.wn = wntr.network.WaterNetworkModel(inp_file)

        ''' Import the demand patterns '''
        self.demand_patterns = pd.read_csv(
            os.path.join(city_dir, 'patterns.csv'),
            delimiter=','
        )

        ''' Import the distributions of agents '''
        self.node_distributions = pd.read_csv(
            os.path.join(city_dir, 'hourly_population.csv'),
            delimiter=','
        )

        ''' Assign each building a node in the WDN '''
        self.node_buildings = ci.make_building_list(self.wn, name, city_dir)
        print(self.node_buildings)

        

        self.setup_grid()

    def setup_virtual(self, network):
        if network == "micropolis":
            inp_file = 'Input Files/micropolis/MICROPOLIS_v1_inc_rest_consumers.inp'
            data = pd.read_excel(r'Input Files/micropolis/Micropolis_pop_at_node.xlsx')
        elif network == "mesopolis":
            inp_file = 'Input Files/mesopolis/Mesopolis.inp'
            data = pd.read_excel(r'Input Files/mesopolis/Mesopolis_pop_at_node.xlsx')
        pattern_list, self.wn = ut.init_wntr(inp_file)

        # input the number of agents required at each node type at each time
        node_id = data['Node'].tolist()
        maxpop_node = data['Max Population'].tolist()
        if network == "mesopolis":
            house_num = data['HOUSE'].tolist()
            # create dictionary with the number of houses per node
            self.house_num = dict(zip(node_id, house_num))
        else:
            self.house_num = None

        # Creating dictionary with max pop at each terminal node
        self.node_capacity = dict(zip(node_id, maxpop_node))
        self.age_nodes = [n for n in self.nodes_capacity if self.nodes_capacity[n] != 0]

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
            self.cafe_nodes = ut.find_nodes(1, pattern_list, network)
            # residential nodes
            self.res_nodes = ut.find_nodes(2, pattern_list, network)
            # Industrial nodes
            self.ind_nodes = ut.find_nodes(3, pattern_list, network)
            # Rest of commercial nodes like schools, churches etc.
            self.com_nodes = ut.find_nodes(5, pattern_list, network)
            self.com_nodes = self.com_nodes + ut.find_nodes(6, pattern_list, network)
            self.nav_nodes = []
            self.air_nodes = []
        elif network == "mesopolis":
            # pattern types: air, com, res, ind, nav
            self.air_nodes = ut.find_nodes('air', pattern_list, network)
            self.com_nodes = ut.find_nodes('com', pattern_list, network)
            self.res_nodes = ut.find_nodes('res', pattern_list, network)
            self.ind_nodes = ut.find_nodes('ind', pattern_list, network)
            self.nav_nodes = ut.find_nodes('nav', pattern_list, network)
            self.cafe_nodes = ut.find_nodes('cafe', pattern_list, network)

        self.terminal_nodes = (
            self.cafe_nodes +
            self.res_nodes +
            self.ind_nodes +
            self.com_nodes +
            self.air_nodes +
            self.nav_nodes
        )

        # finish setup process by loading distributions of agents at each node type,
        # media data, and the distance between residential nodes and closest ind.
        # node.
        pop_dict = ut.load_distributions(network)
        self.res_dist = pop_dict['res']  # residential capacities at each hour
        self.com_dist = pop_dict['com']  # commercial capacities at each hour
        self.ind_dist = pop_dict['ind']  # industrial capacities at each hour
        self.sum_dist = pop_dict['sum']  # sum of capacities
        self.cafe_dist = pop_dict['cafe']  # restaurant capacities at each hour
        if network == 'micropolis':
            # self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
            self.nav_dist = [0]  # placeholder for agent assignment
        if network == 'mesopolis':
            self.air_dist = pop_dict['air']
            self.nav_dist = pop_dict['nav']
        self.ind_node_dist, na = ut.calc_industry_distance(
            self.wn, node_dict['ind'], nodes=node_dict['res'])

        self.setup_grid()

    def setup_grid(self):
        ''' Set up the grid for agent movement '''
        self.G = self.wn.get_graph()
        self.grid = NetworkGrid(self.G)
        self.num_nodes = len(self.G.nodes)
        for i in range(dt.wfh_patterns.shape[1]):
            self.wn.add_pattern(
                'wk'+str(i+1), dt.wfh_patterns[:, i]
            )

    def save_pars(self):
        """
        Save parameters to a DataFrame, param_out, to save at the end of the
        simulation. This helps with data organization.
        """
        self.param_out = pd.DataFrame(columns=['Param', 'value1', 'value2'])
        covid_exp = pd.DataFrame([['covid_exposed', self.covid_exposed]],
                                 columns=['Param', 'value1'])
        hh_rate = pd.DataFrame([['household_rate', self.exposure_rate]],
                               columns=['Param', 'value1'])
        wp_rate = pd.DataFrame([['workplace_rate', self.exposure_rate_large]],
                               columns=['Param', 'value1'])
        inf_time = pd.DataFrame([['infection_time', self.e2i[0], self.e2i[1]]],
                                columns=['Param', 'value1', 'value2'])
        syp_time = pd.DataFrame([['symptomatic_time', self.i2s[0],
                                  self.i2s[1]]],
                                columns=['Param', 'value1', 'value2'])
        sev_time = pd.DataFrame([['severe_time', self.s2sev[0],
                                  self.s2sev[1]]],
                                columns=['Param', 'value1', 'value2'])
        crit_time = pd.DataFrame([['critical_time', self.sev2c[0],
                                   self.sev2c[1]]],
                                 columns=['Param', 'value1', 'value2'])
        death_time = pd.DataFrame([['death_time', self.c2d[0], self.c2d[1]]],
                                  columns=['Param', 'value1', 'value2'])
        asymp_rec_time = pd.DataFrame([['asymp_recovery_time',
                                        self.recTimeAsym[0],
                                        self.recTimeAsym[1]]],
                                      columns=['Param', 'value1', 'value2'])
        mild_rec_time = pd.DataFrame([['mild_recovery_time',
                                       self.recTimeMild[0],
                                       self.recTimeMild[1]]],
                                     columns=['Param', 'value1', 'value2'])
        sev_rec_time = pd.DataFrame([['severe_recovery_time',
                                      self.recTimeSev[0],
                                      self.recTimeSev[1]]],
                                    columns=['Param', 'value1', 'value2'])
        crit_rec_time = pd.DataFrame([['critical_recovery_time',
                                       self.recTimeC[0],
                                       self.recTimeC[1]]],
                                     columns=['Param', 'value1', 'value2'])
        daily_cont = pd.DataFrame([['daily_contacts', self.daily_contacts]],
                                  columns=['Param', 'value1'])
        bbn_mod = pd.DataFrame([['bbn_models', self.bbn_models]],
                               columns=['Param', 'value1'])
        res_pat = pd.DataFrame([['res pattern', self.res_pat_select]],
                               columns=['Param', 'value1'])
        wfh_lag = pd.DataFrame([['wfh_lag', self.wfh_lag]],
                               columns=['Param', 'value1'])
        no_wfh = pd.DataFrame([['percent ind no wfh', self.no_wfh_perc]],
                              columns=['Param', 'value1'])
        ppe_reduc = pd.DataFrame([['ppe_reduction', self.ppe_reduction]],
                                 columns=['Param', 'value1'])

        self.param_out = pd.concat([covid_exp, hh_rate, wp_rate, inf_time,
                                   syp_time, sev_time, crit_time, death_time,
                                   asymp_rec_time, mild_rec_time, sev_rec_time,
                                   crit_rec_time, daily_cont, bbn_mod, res_pat,
                                   wfh_lag, no_wfh, ppe_reduc])
