# Consumer Model WNTR- MESA
import warnings
from mesa import Model
from mesa.time import RandomActivation
from utils import setup
from hydraulic import EpanetSimulator_Stepwise
import networkx as nx
from mesa.space import NetworkGrid
from agent_model import Household
import data as dt
import math
import time
import bnlearn as bn
import numpy as np
import pandas as pd
from copy import deepcopy as dcp
import wntr
from wntr.epanet.util import EN

import matplotlib.pyplot as plt

warnings.simplefilter("ignore", UserWarning)


class ConsumerModel(Model):
    '''
    ABM model class. Contains all methods to run the ABM simulation

    Parameters
    ----------
    N : int
        number of agents to create
    city : string
        Name of city to be modeled. Options include micropolis and mesopolis
    days : int
        number of days the simulation will run
    id : int
        Unique identifer for model
    seed : int
        RNG seed
    '''

    def __init__(self,
                 N,   # number of people simulated
                 city,
                 days=90,
                 id=0,    # id of simulation
                 seed=None,
                 **kwargs):
        super().__init__()
        init_start = time.perf_counter()
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

        if 'start_inf' in kwargs:
            self.covid_exposed = kwargs['start_inf'] #round(0.001*N) # number of starting infectious
        else:
            self.covid_exposed = int(0.001 * self.num_agents)

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

        if 'daily_contacts' in kwargs:
            self.daily_contacts = kwargs['daily_contacts']
        else:
            self.daily_contacts = 10
        self.cumm_infectious = self.covid_exposed

        ''' Import the four DAGs for the BBNs '''
        self.wfh_dag = bn.import_DAG('Input Files/data_driven_models/work_from_home.bif', verbose=0)
        self.dine_less_dag = bn.import_DAG('Input Files/pmt_models/dine_out_less_pmt-6.bif', verbose=0)
        self.grocery_dag = bn.import_DAG('Input Files/pmt_models/shop_groceries_less_pmt-6.bif', verbose=0)
        self.ppe_dag = bn.import_DAG('Input Files/data_driven_models/mask.bif', verbose=0)

        if 'res_pat_select' in kwargs:
            self.res_pat_select = kwargs['res_pat_select']
        else:
            self.res_pat_select = 'lakewood'

        if 'wfh_lag' in kwargs:
            self.wfh_lag = kwargs['wfh_lag']  # infection percent before work from home allowed
        else:
            self.wfh_lag = 0
        self.wfh_thres = False  # whether wfh lag has been reached

        if 'no_wfh_perc' in kwargs:
            self.no_wfh_perc = kwargs['no_wfh_perc']
        else:
            self.no_wfh_perc = 0.5

        if 'bbn_models' in kwargs:
            if 'all' in kwargs['bbn_models']:
                self.bbn_models = ['wfh', 'dine', 'grocery', 'ppe']
            else:
                self.bbn_models = kwargs['bbn_models']
        else:
            self.bbn_models = ['wfh', 'dine', 'grocery', 'ppe']

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 1

        if 'ppe_reduction' in kwargs:
            self.ppe_reduction = kwargs['ppe_reduction']
        else:
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
        if 'hyd_sim' in kwargs:
            self.hyd_sim = kwargs['hyd_sim']
        else:
            self.hyd_sim = 'eos'

        '''
        The warmup input dictates whether a warmup period is run to reach steady
        state water age values. Default is true
        '''
        if 'warmup' in kwargs:
            self.warmup = kwargs['warmup']
        else:
            self.warmup = True

        '''
        Warmup tolerance is the threshold for when warmup period is complete
        '''
        if 'tol' in kwargs:
            self.tol = kwargs['tol']
        else:
            self.tol = 0.001

        '''
        bw dictates whether bottled water buying is modeled. Defaults to True
        '''
        if 'bw' in kwargs:
            self.bw = kwargs['bw']
        else:
            self.bw = True

        '''
        twa_mods are the modulators to the twa thresholds passed to households
        '''
        if 'twa_mods' in kwargs:
            self.twa_mods = kwargs['twa_mods']
        else:
            self.twa_mods = [130, 150, 140]

        '''
        self.ind_min_demand is the minimum industrial demand as a percentage
        '''
        if 'ind_min_demand' in kwargs:
            self.ind_min_demand = kwargs['ind_min_demand']
        else:
            self.ind_min_demand = 0

        '''
        dist_income is whether income is industrial distance based
        '''
        if 'dist_income' in kwargs:
            self.dist_income = kwargs['dist_income']
        else:
            self.dist_income = True

        '''
        twa_process dictates whether the twas are represented as absolute
        reductions or percentage reductions

        two options are absolute and percentage
        '''
        if 'twa_process' in kwargs:
            self.twa_process = kwargs['twa_process']
        else:
            self.twa_process = 'absolute'

        ''' Setup and mapping of variables from various sources. For more information
        see utils.py '''
        setup_out = setup(city)

        ''' need to check whether these are mapped correctly. '''
        self.res_nodes = setup_out[0]['res']
        self.ind_nodes = setup_out[0]['ind']
        self.com_nodes = setup_out[0]['com']
        self.cafe_nodes = setup_out[0]['cafe']  # There is no node assigned to "dairy queen" so it was neglected
        if city == 'micropolis':
            self.nav_nodes = []  # placeholder for agent assignment
        if city == 'mesopolis':
            self.air_nodes = setup_out[0]['air']
            self.nav_nodes = setup_out[0]['nav']

        self.nodes_capacity = setup_out[1]
        self.house_num = setup_out[2]

        self.age_nodes = [n for n in self.nodes_capacity if self.nodes_capacity[n] != 0]

        self.res_dist = setup_out[3]['res']  # residential capacities at each hour
        self.com_dist = setup_out[3]['com']  # commercial capacities at each hour
        self.ind_dist = setup_out[3]['ind']  # industrial capacities at each hour
        self.sum_dist = setup_out[3]['sum']  # sum of capacities
        self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
        if city == 'micropolis':
            # self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
            self.nav_dist = [0]  # placeholder for agent assignment
        if city == 'mesopolis':
            self.air_dist = setup_out[3]['air']
            self.nav_dist = setup_out[3]['nav']

        # self.sleep = setup_out[4]['sleep']
        # self.radio = dcp(setup_out[4]['radio'])
        # self.tv = dcp(setup_out[4]['tv'])
        # self.bbn_params = setup_out[5]  # pandas dataframe of bbn parameters
        # wfh_patterns = setup_out[6]
        self.terminal_nodes = setup_out[4]
        self.wn = setup_out[5]
        self.ind_node_dist = setup_out[6]  # distance between res nodes and closest ind node

        self.G = self.wn.get_graph()
        self.grid = NetworkGrid(self.G)
        self.num_nodes = len(self.G.nodes)
        for i in range(dt.wfh_patterns.shape[1]):
            self.wn.add_pattern(
                'wk'+str(i+1), dt.wfh_patterns[:, i]
            )

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

        ''' Initialize the hydraulic information collectors '''
        # these are nodes with demands (there are also nodes without demand):
        self.nodes_w_demand = [
            node for node in self.grid.G.nodes
            if hasattr(self.wn.get_node(node), 'demand_timeseries_list')
        ]

        if self.hyd_sim == 'eos':
            self.daily_demand = np.empty((24, len(self.nodes_w_demand)))
            self.demand_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                              columns=self.G.nodes)
            self.pressure_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                                columns=self.G.nodes)
            self.age_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                           columns=self.G.nodes)
            self.flow_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                            columns=[name for name, link in self.wn.links()])
        elif self.hyd_sim == 'hourly':
            self.current_demand = pd.Series(
                0,
                index=self.G.nodes
            )
        elif self.hyd_sim == 'monthly':
            self.daily_demand = np.empty((24, len(self.nodes_w_demand)))
            self.current_age = None

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

        # initialization methods
        self.base_demand_list()
        self.set_age()
        if self.hyd_sim == 'hourly' or self.hyd_sim == 'monthly':
            self.wn.options.time.pattern_timestep = 3600
            self.wn.options.time.hydraulic_timestep = 3600
            # self.wn.options.time.quality_timestep = 900
            self.wn.options.quality.parameter = 'AGE'
            self.sim = EpanetSimulator_Stepwise(self.wn,
                                                file_prefix='temp' + str(self.id))
            self.sim.initialize(file_prefix='temp' + str(self.id))
            # self.current_demand = self.sim._results.node['demand']

        # water age slope to determine the warmup period end
        self.water_age_slope = 1

        # if we are running the simulation hourly, we need to have
        # demand patterns that are as long as the simulation, which
        # is what update_patterns does.
        if self.hyd_sim == 'hourly':
            self.update_patterns()
        self.create_node_list()
        if self.verbose == 0.5:
            print("Creating agents ..............")
        self.create_agents()
        if self.verbose == 0.5:
            print("Setting agent attributes ...............")
        self.set_attributes()
        self.dag_nodes()
        if self.res_pat_select == 'pysimdeum':
            self.create_demand_houses()
        self.create_comm_network()

        # # Create dictionary of work from home probabilities using bnlearn.
        # self.rp_wfh_probs = {}
        # for i in range(7):
        #     query = bn.inference.fit(self.wfh_dag,
        #                              variables = ['work_from_home'],
        #                              evidence = {'risk_perception_r':(i)},
        #                              verbose = 0)
        #     self.rp_wfh_probs[i] = query.df['p'][1]
        #
        # print(self.rp_wfh_probs)

        init_stop = time.perf_counter()

        if self.verbose == 1:
            print('Time to initialize: ', init_stop - init_start)

    def dag_nodes(self):
        self.wfh_nodes = dcp(self.wfh_dag['adjmat'].columns)
        self.dine_nodes = dcp(self.dine_less_dag['adjmat'].columns)
        self.grocery_nodes = dcp(self.grocery_dag['adjmat'].columns)
        self.ppe_nodes = dcp(self.ppe_dag['adjmat'].columns)

    def base_demand_list(self):
        self.base_demands = dict()
        self.base_pattern = dict()
        self.node_index = dict()
        # self.demand_multiplier = dict()
        for node in self.nodes_w_demand:
            node_1 = self.wn.get_node(node)
            self.base_demands[node] = node_1.demand_timeseries_list[0].base_value

            ''' Make a pattern for each node for hydraulic simulation '''
            if self.hyd_sim == 'eos' or self.hyd_sim == 'monthly':
                curr_pattern = dcp(node_1.demand_timeseries_list[0].pattern)
                self.wn.add_pattern('node_'+node, curr_pattern.multipliers)
                # set the demand pattern for the node to the new pattern
                node_1.demand_timeseries_list[0].pattern_name = 'node_'+node
            # elif self.hyd_sim == 'hourly' or isinstance(self.hyd_sim, int):
            #     self.node_index[node] = dcp(self.sim._en.ENgetnodeindex(node))
            self.base_pattern[node] = (
                dcp(node_1.demand_timeseries_list[0].pattern_name),
                dcp(self.wn.get_pattern(node_1.demand_timeseries_list[0].pattern_name))
            )

            # set demand multipliers for each node for the tap water avoidance
            # modeling
            # self.demand_multiplier[node] = 1

    def set_age(self):
        ''' Set initial water age '''
        init_age = pd.read_pickle('hot_start_age_data_2024-03-08_12-10_200days_results.pkl')
        for node in self.grid.G.nodes:
            curr_node = self.wn.get_node(node)
            curr_node.initial_quality = float(init_age.loc[[node]].values[0])

    def node_list(self, list, nodes):
        list_out = []
        for node in nodes:
            for i in range(int(list[node])):
                list_out.append(node)
        return list_out

    def create_node_list(self):
        nodes_industr_2x = self.ind_nodes + self.ind_nodes
        nodes_nav_2x = self.nav_nodes + self.nav_nodes
        self.work_loc_list = self.node_list(self.nodes_capacity, nodes_industr_2x + nodes_nav_2x)
        self.res_loc_list = self.node_list(self.nodes_capacity, self.res_nodes)
        # self.rest_loc_list = node_list(self.nodes_capacity, self.cafe_nodes)
        # self.comm_loc_list = node_list(self.nodes_capacity, self.com_nodes)

    def create_agents(self):
        ''' Creating lists of nodes where employers have decided not to allow
        working from home or jobs that are "essential". '''

        ''' Potentially change this to include navy nodes '''
        self.total_no_wfh = self.random.choices(
                                population=self.ind_nodes,
                                k=int(len(self.ind_nodes)*self.no_wfh_perc)
                            )

        # create dictionary of households
        self.households = dict()
        self.household_n = dict()
        # no_wfh_comm_nodes = self.random.choices(population=self.com_nodes,
        #                                         k=int(len(self.com_nodes)*0.2))
        # no_wfh_rest_nodes = self.random.choices(population=self.cafe_nodes,
        #                                         k=int(len(self.cafe_nodes)*0.2))

        self.work_agents = (max(self.ind_dist) + max(self.nav_dist)) * 2
        # rest_agents = max(self.cafe_dist)
        # comm_agents = max(self.com_dist)

        # CREATING AGENTS
        res_nodes = dcp(self.res_nodes)
        self.random.shuffle(res_nodes)
        ids = 0
        max_node_dist = max(self.ind_node_dist.values())

        '''
        Place all the agents in the residential nodes, each filled up to
        its capacity

        The self.households and self.households_n are filled in the household
        methods (micro_ and meso_)
        '''
        for node in res_nodes:
            # distance to closest industrial node relative to max distance
            # essentially a normalized distance
            node_dist = self.ind_node_dist[node] / max_node_dist
            # for spot in range(int(self.nodes_capacity[node])):
            if self.network == 'micropolis':
                ids = self.micro_household(ids, node, node_dist)
            elif self.network == 'mesopolis':
                ids = self.meso_household(ids, node)

        # collect income and income level from each household that was just created
        self.income = [h.income for n, i in self.households.items() for h in i]
        self.income_level = [h.income_level for n, i in self.households.items() for h in i]
        self.hh_size = [len(h.agent_ids) for n, i in self.households.items() for h in i]
        self.income_comb = pd.DataFrame(
            data={
                'income': self.income,
                'level': self.income_level,
                'hh_size': self.hh_size
            },
            index=[h.node for n, i in self.households.items() for h in i]
        )

        if self.network == 'mesopolis':
            '''
            Need to initialize the rest of the agents because there are only
            139,654 residential spots. Currently we are assuming that the
            remaining 7062 agents are at the airport, but this is a bad
            assumption because the max capacity of the airport is 429.
            '''
            for pop in range(self.num_agents - ids):
                a = ConsumerAgent(ids, self)
                self.schedule.add(a)
                a.home_node = 'TN1372'
                ids += 1

    def micro_household(self, init_id, curr_node, node_dist):
        ''' Assigns each agent's housemates based on the number
        of agents at the current node.

        If the residential node is larger than 6, then we need to
        artificially make households of 6 or less. '''
        node_cap = dcp(self.nodes_capacity[curr_node])
        node_cap_static = node_cap
        if node_cap == 0:
            return init_id

        curr_ids = init_id

        house_list = list()

        '''
        Need to account for multifamily housing, so iterating through
        residential nodes and placing agents that way and then storing their
        housemates in the agent object.

        Iterate through the agents at the current res node
        and add them to households of 1 to 6 agents
        '''
        while node_cap > 6:
            # pick a random size for current household
            home_size = self.random.choice(range(1, 7))
            prev_id = curr_ids
            curr_ids += home_size
            # make the household and append it to a list of households
            house_list.append(Household(
                prev_id, curr_ids, curr_node, node_dist, self.twa_mods, self
            ))
            node_cap -= home_size
        else:
            # if the node size is 6 or fewer, we only need to do this once
            house_list.append(Household(
                curr_ids,
                node_cap_static + init_id,
                curr_node,
                node_dist,
                self.twa_mods,
                self
            ))

        self.households[curr_node] = house_list
        self.household_n[curr_node] = len(house_list)

        return init_id + node_cap

    def meso_household(self, curr_node):
        ''' Assign each agent's housemates based on the current node.

        The current plan is to fill all residential node spots and then
        the rest of the agents are travellers at the airport. '''
        # need to rectify the difference between the number of
        # residential spots (139,654) and the total population
        # (146,716)
        ''' Iterate through the agents at the current res node
        and add them to households of 1 to 6 agents '''
        while len(curr_node) > 6:
            # pick a random size for current household
            # a uniform distribution between 1 and 7 will average to 4
            home_size = int(self.random.uniform(1, 8))
            # pick the agents that will occupy this household
            curr_housemates = self.random.choices(curr_node, k=home_size)
            # remake the curr_node variable without the agents just chosen
            curr_node = [a for a in curr_node if a not in curr_housemates]
            ''' Assign the current list of housemates to each agent
            so they each know who their housemates are '''

            for mate in curr_housemates:
                agent = self.schedule._agents[mate]
                # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                agent.housemates = dcp(curr_housemates)  # this includes current agent

        for mate in curr_node:
            agent = self.schedule._agents[mate]
            # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
            agent.housemates = dcp(curr_node)

    def create_demand_houses(self):
        ''' Create houses using pysimdeum for stochastic demand simulation '''
        self.res_houses = list()
        for node in self.res_nodes:
            agents_at_node = len(self.grid.G.nodes[node]['agent'])
            if agents_at_node > 5 or agents_at_node < 2:
                pass
            elif agents_at_node == 1:
                house = pysimdeum.built_house(house_type='one_person')
                house.id = node
                for user in house.users:
                    user.age = 'work_ad'
                    user.job = True
                self.res_houses.append(house)
            else:
                house = pysimdeum.built_house(house_type='family', user_num=agents_at_node)
                house.id = node
                for user in house.users:
                    user.age = 'work_ad'
                    user.job = True
                self.res_houses.append(house)

    def check_houses(self):
        for house in self.res_houses:
            node = house.id
            agents_at_node = self.grid.G.nodes[node]['agent']
            for agent in agents_at_node:
                if agent.wfh == 1 or agent.work_node == None:
                    user.age = 'home_ad'
                    user.job = True

    def set_attributes(self):
        '''
        Assign agents an age 1 = 0-19, 2 = 20-29, 3 = 30-39, 4 = 40-49,
        5 = 50-59, 6 = 60-69, 7 = 70-79, 8 = 80-89, 9 = 90+
        '''
        ages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        age_weights = [0.25, 0.18, 0.15, 0.14, 0.12, 0.08, 0.05, 0.01, 0.01]

        '''
        Assign agents either susceptible or infected to COVID based on initial
        infected number.
        '''
        exposed_sample = self.random.sample(
            [i for i, a in enumerate(self.schedule.agents)],
            self.covid_exposed
        )
        for i, agent in enumerate(self.schedule.agents):
            ''' Set age of each agent '''
            agent.age = self.random.choices(population=ages,
                                            weights=age_weights,
                                            k=1)[0]

            ''' Set covid infection for each agent '''
            if i in exposed_sample:
                agent.covid = 'infectious'
            else:
                agent.covid = 'susceptible'

            ''' Set bbn parameters for each agent '''
            agent_set_params = dt.bbn_params[
                self.random.randint(0, dt.bbn_params.shape[0] - 1), :
            ]
            # agent_set_params = dt.bbn_params.sample()
            # agent.agent_params['risk_perception_r'] = int(agent_set_params['risk_perception_r']) - 1
            for i, param in enumerate(dt.bbn_param_list):
                agent.agent_params[param] = agent_set_params[i] - 1
                # try:
                #     # if param == "COVIDeffect_4":
                #     #     pass
                #     if param == "DemEdu":
                #         if int(agent_set_params[param]) == 9:
                #             agent.agent_params[param] = 5
                #         else:
                #             agent.agent_params[param] = int(agent_set_params[param]) - 1
                #     elif param == "Ethnicmin":
                #         if int(agent_set_params[param]) == 4:
                #             agent.agent_params[param] = 2
                #         else:
                #             agent.agent_params[param] = int(agent_set_params[param]) - 1
                #     elif param == "COVIDexp":
                #         agent.agent_params[param] = 7
                #     elif param == "MediaExp_3":
                #         agent.agent_params[param] = 1
                #     else:
                #         agent.agent_params[param] = int(agent_set_params[param]) - 1
                # except:
                #     pass

    def create_comm_network(self):  # a random set of agents from household (all agents live at the same node)
        '''
        CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
        Assigning Agents randomly to nodes in SWN
        '''
        for agent in self.schedule.agents:
            agent.friends = self.swn.neighbors(agent.unique_id)
        # self.snw_agents_node = {}
        # Nodes_in_snw = list(range(1, Micro_pop + 1))

        # # Create dictionairy with dict[agents]= Node
        # for agent in self.schedule.agents:
        #     node_to_agent = self.random.choice(Nodes_in_snw)
        #     self.snw_agents_node[agent] = node_to_agent
        #     Nodes_in_snw.remove(node_to_agent)

        # # Create dictionairy with dict[Nodes]= agent
        # self.snw_node_agents = {y: x for x, y in self.snw_agents_node.iteritems()}
        # self.snw_node_agents = dict(zip(self.snw_agents_node.values(), self.snw_agents_node.keys()))
        # for key in self.snw_node_agents:
        #     print(self.snw_node_agents[key])

    def num_status(self):
        """
        Function to calculate the number of agents in each compartment (Susceptible,
        exposed, infectious, recovered, and dead), the number of either symptomatic
        and asymptomatic, and the number of agents that are in each severity
        class (mild, severe, and critical). This information is printed every
        hour step.
        """
        self.stat_tot = [
            self.timestep,
            0,
            0,
            0,
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

        for i, agent in enumerate(self.schedule.agents):
            if agent.covid == 'susceptible':
                self.stat_tot[1] += 1
            elif agent.covid == 'exposed':
                self.stat_tot[2] += 1
            elif agent.covid == 'infectious':
                self.stat_tot[3] += 1
            elif agent.covid == 'recovered':
                self.stat_tot[4] += 1
            elif agent.covid == 'dead':
                self.stat_tot[5] += 1
            else:
                pass

            if agent.symptomatic == 1:
                self.stat_tot[6] += 1
            elif agent.symptomatic == 0:
                self.stat_tot[7] += 1
            else:
                pass

            if agent.inf_severity == 1:
                self.stat_tot[8] += 1
            elif agent.inf_severity == 2:
                self.stat_tot[9] += 1
            elif agent.inf_severity == 3:
                self.stat_tot[10] += 1
            else:
                pass

        self.stat_tot[12] = int(len(self.agents_wfh()))

        self.stat_tot = [i / self.num_agents for i in self.stat_tot]
        self.status_tot[self.timestep] = dcp(self.stat_tot)

    def contact(self, agent_to_move, node_type):
        """
        Function to test whether a specific agent has exposed the other agents
        in their node to covid. This is currently called everytime an agent moves
        to a new location and NOT every hour.
        """
        agents_at_node = self.grid.G.nodes[agent_to_move.pos]['agent']
        if node_type == 'residential':
            agents_to_expose = [a for a in agent_to_move.household.agent_obs
                                if a in agents_at_node]
        elif node_type == 'workplace':
            if len(agents_at_node) > self.daily_contacts:
                agents_to_expose = self.random.sample(population=agents_at_node,
                                                      k=self.daily_contacts)
            else:
                agents_to_expose = agents_at_node

        for agent in agents_to_expose:
            if agent.covid == 'susceptible':
                if node_type == 'workplace':
                    if agent_to_move.ppe == 0:
                        if self.random.random() < self.exposure_rate_large:
                            agent.covid = 'exposed'
                        else:
                            pass
                    else:
                        if self.random.random() < self.exposure_rate_large * self.ppe_reduction:
                            agent.covid = 'exposed'
                        else:
                            pass
                elif node_type == 'residential':
                    if agent_to_move.ppe == 0:
                        if self.random.random() < self.exposure_rate:
                            agent.covid = 'exposed'
                        else:
                            pass
                    else:
                        if self.random.random() < self.exposure_rate * self.ppe_reduction:
                            agent.covid = 'exposed'
                        else:
                            pass
                else:
                    print('Warning: node type: ' + str(node_type) + ' does not exist.')
            else:
                pass

    def check_status(self):
        """
        Function to check status of agents and increase timers for each disease
        state. Runs every hour.
        """
        for i, agent in enumerate(self.schedule.agents):
            """
            Add time to exp_time if agent covid status is exposed, add time to
            infectious_time if the agent is infectious.
            """
            if agent.covid == 'exposed':
                agent.exposed_time += 1
            elif agent.covid == 'infectious':
                agent.infectious_time += 1
                # Add time to sev_time and crit_time if agent is in severe or
                # critical state. Also, reset sev_time if agent is critical, to
                # remove the chance of the agent being considered in both categories.
                if agent.inf_severity == 2:
                    agent.sev_time += 1
                elif agent.inf_severity == 3:
                    agent.crit_time += 1
                    agent.sev_time = 0
                else:
                    pass
            else:
                pass

            """
            Add time to symp_time if agent is symptomatic.
            """
            if agent.symptomatic == 1:
                agent.symp_time += 1
            else:
                pass

    def check_social_dist(self):
        for i, agent in enumerate(self.schedule.agents):
            if self.random.random() < self.wfh_probs[math.floor(agent.agent_params["COVIDeffect_4"])]:
                agent.wfh = 1

    def communication_utility(self):
        # Communication through TV and Radio
        # Percentage of people listening to radio:
        radio_reach = dt.radio[self.timestepN]/100
        tv_reach = dt.tv[self.timestepN]/100

        # Communication through radio
        for i, a in enumerate(self.schedule.agents):
            if self.random.random() < radio_reach:
                a.agent_params['MediaExp_3'] = 0
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

        # Communication through TV
        for i, a in enumerate(self.schedule.agents):
            if self.random.random() < tv_reach:
                a.agent_params['MediaExp_3'] = 0
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

    def move(self):
        """
        Move the correct number of agents to and from commercial nodes.
        """
        curr_comm_num = self.com_dist[self.timestepN]
        prev_comm_num = self.com_dist[self.timestepN - 1]
        delta_agents_comm = round(curr_comm_num - prev_comm_num)
        if delta_agents_comm > 0:
            Possible_Agents_to_move = [a for a in self.schedule.agents
                                       if a.pos in self.res_nodes]
                                       #and a.work_type == 'commercial']

            nodes_comm = list()
            for node in self.com_nodes:
                avail_spots = self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_comm.append(node)
            # print('Comm nodes: ' + str(len(nodes_comm)))

            # we want all the commercial nodes that have vacancies and how many
            # vacancies...

            for i in range(min(delta_agents_comm, len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                location = self.random.choice(nodes_comm)
                # work_node = Agent_to_move.work_node
                # while len(self.grid.G.nodes[work_node]['agent']) > self.nodes_capacity[work_node]:
                #     Agent_to_move = self.random.choice(Possible_Agents_to_move)
                    # work_node = Agent_to_move.work_node
                # if (Agent_to_move.wfh == 1 and
                #     self.wfh_thres and
                #     Agent_to_move.can_wfh == True):
                #     pass
                # else:
                # if (Agent_to_move.work_type != None and
                #     Agent_to_move in self.grid.G.nodes[Agent_to_move.work_node]['agent']):
                #     print(f"Agent {Agent_to_move} is at work.")
                if self.base_pattern[location][0] == '6':
                    if Agent_to_move.less_groceries == 1:
                        continue
                    self.grid.move_agent(Agent_to_move, location)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    nodes_comm.remove(location)
                    self.infect_agent(Agent_to_move, 'workplace')
                else:
                    ''' The agents in this arm are considered workers '''
                    if Agent_to_move.wfh == 1:
                        continue
                    self.grid.move_agent(Agent_to_move, location)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    nodes_comm.remove(location)
                    self.infect_agent(Agent_to_move, 'workplace')

        elif delta_agents_comm < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
            Possible_Agents_to_move = self.commercial_agents()
            for i in range(min(abs(delta_agents_comm), len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move.remove(Agent_to_move)
                self.infect_agent(Agent_to_move, 'residential')

        """
        Move the correct number of agents to and from cafe nodes.
        """
        curr_rest_num = self.cafe_dist[self.timestepN]
        prev_rest_num = self.cafe_dist[self.timestepN - 1]
        delta_agents_rest = round(curr_rest_num - prev_rest_num)
        if delta_agents_rest > 0:
            Possible_Agents_to_move = [a for a in self.schedule.agents
                                       if a.pos in self.res_nodes]
                                       # and a.work_type == 'restaurant']

            nodes_cafe = list()
            for node in self.cafe_nodes:
                avail_spots = self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_cafe.append(node)
            # print('Cafe nodes: ' + str(len(nodes_cafe)))

            for i in range(min(delta_agents_rest,len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                location = self.random.choice(nodes_cafe)
                # if (Agent_to_move.wfh == 1 and
                #     self.wfh_thres and
                #     Agent_to_move.can_wfh == True):
                #     pass
                # else:
                if Agent_to_move.no_dine == 1:
                    continue
                self.grid.move_agent(Agent_to_move, location)
                Possible_Agents_to_move.remove(Agent_to_move)
                nodes_cafe.remove(location)
                self.infect_agent(Agent_to_move, 'workplace')

        elif delta_agents_rest < 0:
            Possible_Agents_to_move = self.rest_agents()
            for i in range(min(abs(delta_agents_rest), len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move.remove(Agent_to_move)
                self.infect_agent(Agent_to_move, 'residential')

    # Moving Agents from and to work in Industrial Nodes. Every 8 hours half the Agents in industrial nodes
    # are being replaced with Agents from Residential nodes. At 1, 9 and 17:00
    def move_indust(self):
        # Moving Agents from Industrial nodes back home to residential home nodes
        Possible_Agents_to_move_home = self.industry_agents()
        if self.network == 'micropolis':
            Agents_to_home = int(min(1092/2, len(Possible_Agents_to_move_home)))
            Agents_to_work = int(1092/2) if self.timestep != 0 else 1092
        elif self.network == 'mesopolis':
            Agents_to_home = int(min(65228/2, len(Possible_Agents_to_move_home)))
            Agents_to_work = int(65228/2) if self.timestep != 0 else 65228

        for i in range(Agents_to_home):
            Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
            self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
            Possible_Agents_to_move_home.remove(Agent_to_move)
            self.infect_agent(Agent_to_move, 'residential')

        # Agents from Residential nodes to Industrial
        Possible_Agents_to_move_to_work = [a for a in self.schedule.agents
                                           if a.pos in self.res_nodes
                                           and a.work_type == 'industrial']

        self.agents_moved = list()
        for i in range(Agents_to_work):
            Agent_to_move = self.random.choice(Possible_Agents_to_move_to_work)
            work_node = Agent_to_move.work_node
            while len(self.grid.G.nodes[work_node]['agent']) > self.nodes_capacity[work_node]:
                Agent_to_move = self.random.choice(Possible_Agents_to_move_to_work)
                work_node = Agent_to_move.work_node
            if (Agent_to_move.wfh == 1 and
                self.wfh_thres and
                Agent_to_move.can_wfh == True):
                continue
            self.agents_moved.append(Agent_to_move)
            self.grid.move_agent(Agent_to_move, Agent_to_move.work_node)
            Possible_Agents_to_move_to_work.remove(Agent_to_move)
            self.infect_agent(Agent_to_move, 'workplace')

    def set_patterns(self, node):
        house = [house for house in self.res_houses if house.id == node.name][0]
        consumption = house.simulate(num_patterns=10)
        tot_cons = consumption.sum(['enduse', 'user']).mean([ 'patterns'])
        hourly_cons = tot_cons.groupby('time.hour').mean()
        self.wn.add_pattern(
            'hs_' + house.id,
            np.array(np.divide(hourly_cons, hourly_cons.mean()))
        )
        node.demand_timeseries_list[0].pattern_name = 'hs_' + house.id

    def collect_demands(self):
        step_demand = np.empty(len(self.nodes_w_demand))
        step_agents = list()
        for i, node in enumerate(self.nodes_w_demand):
            # we need to iterate through all nodes with demand to be able to
            # run the hydraulic simualation.
            if node in self.nodes_capacity:
                capacity_node = self.nodes_capacity[node]
                # node_1 = self.wn.get_node(node)
                agents_at_node_list = self.grid.G.nodes[node]['agent']
                agents_at_node = len(agents_at_node_list)
                step_agents.append(agents_at_node)
                # agents_wfh = len([a for a in agents_at_node_list if a.wfh == 1])
                if capacity_node != 0:
                    multiplier = agents_at_node / capacity_node
                    if node in self.ind_nodes:
                        ''' Industrial demand multiplier should include some
                        portion that is not based on agent movement.
                        This is accounted for by self.ind_min_demand '''
                        step_demand[i] = (
                            multiplier * (1 - self.ind_min_demand)
                            + self.ind_min_demand
                        )
                    else:
                        ''' Non-industrial nodes are solely based on the
                        number of agents at them '''
                        step_demand[i] = multiplier

                        ''' Count the number of agents at each household '''
                        # if node in self.households.keys():
                        #     for house in self.households[node]:
                        #         house.count_agents()
                else:
                    step_demand[i] = 0
            else:
                step_demand[i] = 0

            # try:
            #     # determine demand reduction
            #     demand_reduction_node = 0
            #     # for agent in self.grid.G.nodes[node]['agent']: # determine demand reduction for every agent at that node
            #     #     if agent.compliance == 1:
            #     #         rf = self.random.randint(0.035 * 1000, 0.417 * 1000) / 1000  # Reduction factor for demands
            #     #         demand_reduction_node += node_1.demand_timeseries_list[0].base_value * rf /  agents_at_node     # Calculating demand reduction per agent per node
            #     #     else:
            #     #         continue
            #
            #     # Save first base demand so later assign it back to Node after simulation
            #     # self.base_demands_previous[node] = node_1.demand_timeseries_list[0].base_value
            #     # print(self.base_demands_previous[node])
            #     # node_1.demand_timeseries_list[0].base_value = node_1.demand_timeseries_list[0].base_value * agents_at_node/ Capacity_node - demand_reduction_node
            #     # check how many agents are working from home at current node
            #     # if more than 50%, changes pattern
            #
                # if self.res_pat_select == 'lakewood':
                #     perc_wfh = agents_wfh / agents_at_node
                #     if perc_wfh > 0.5 and node in self.res_nodes:
                #         node_1.demand_timeseries_list[0].pattern_name = 'wk1'
                # elif self.res_pat_select == 'pysimdeum':
                #     if node in self.res_nodes:
                #         self.set_patterns(node_1)
                # else:
                #    pass
            # except:
            #     pass

        assert len(step_agents) == len(self.nodes_capacity)

        self.daily_demand[self.timestepN, :] = step_demand
        self.agent_matrix[self.timestep] = step_agents

    # this how set demand at each node, given what was collected in function above:
    def change_demands(self):
        '''
        Add the current days demand pattern to each nodes demand pattern.
        This appends a new 24 hour pattern based on the demands at each node.
        '''

        for i, node in enumerate(self.nodes_w_demand):
            curr_node = self.wn.get_node(node)
            curr_demand = curr_node.demand_timeseries_list[0].base_value

            ''' list of demand multipliers from agent locations '''
            new_mult = self.daily_demand[:, i]  # np.array

            ''' multiply the location multiplier by the demand multiplier
            calculated by tap water avoidance behaviors '''
            # new_mult = new_mult * self.demand_multiplier[node]
            agents_at_node = self.grid.G.nodes[node]['agent']
            agents_wfh = len([a for a in agents_at_node if a.wfh == 1])
            if self.res_pat_select == 'lakewood' and len(agents_at_node) != 0:
                perc_wfh = agents_wfh / len(agents_at_node)
                if perc_wfh > 0.5 and node in self.res_nodes:
                    base_pat = self.wn.get_pattern('wk1')
                else:
                    base_pat = self.base_pattern[node][1]
            elif self.res_pat_select == 'pysimdeum':
                if node in self.res_nodes:
                    self.set_patterns(curr_node)
            else:
                base_pat = self.base_pattern[node][1]

            node_pat = self.wn.get_pattern('node_'+node)
            if node in self.households.keys():
                ''' demand for the next 24 hours, not including reduction or
                agent multiplier '''
                curr_multipliers = (
                    base_pat.multipliers * new_mult
                )

                daily_demand = curr_multipliers.sum() * curr_demand * 1000000

                reduction_val = 0
                # number of agents assigned to this node over all households
                house_agents = sum([len(h.agent_ids) for h in self.households[node]])
                for house in self.households[node]:
                    # calculate the percent of this nodes demand is caused by
                    # this household
                    agent_percent = len(house.agent_ids) / house_agents
                    # average agent multiplier
                    avg_agent_multiplier = sum(new_mult) / len(new_mult)
                    # add the demand from this household to the tap_demand

                    if self.twa_process == 'absolute':
                        # reduction value needs to be offset by the agent reduction
                        # which is the average agent multiplier
                        house.tap_demand += (
                            daily_demand * agent_percent -
                            house.reduction * avg_agent_multiplier
                        )
                        # iterate the bottle_demand as well
                        house.bottle_demand += house.reduction * avg_agent_multiplier
                        # increase the total reduction value for this node
                        reduction_val += house.reduction * avg_agent_multiplier
                    elif self.twa_process == 'percentage':
                        # daily demand already has information about the number
                        # of agents at this house from new_mult
                        house.tap_demand += (
                            daily_demand * agent_percent * house.change
                        )
                        # iterate the bottle_demand as well
                        house.bottle_demand += (
                            daily_demand * agent_percent * (1 - house.change)
                        )
                        # increase the total reduction value for this node
                        reduction_val += house.bottle_demand

                # should check if the demand for the node is the same as the
                # total from all the households
                house_sum = sum([h.tap_demand for h in self.households[node]])
                if (daily_demand - reduction_val) - house_sum > 0.001:
                    msg = f"Total demand {daily_demand - reduction_val} does not equal the sum of houses {house_sum} for household of {len(self.households[node])}"
                    raise RuntimeError(msg)

                # this is the demand we want for this node. It includes the
                # reduction for agents at the node and for bw use.
                desired_demand = daily_demand - reduction_val
                new_demand_multiplier = desired_demand / daily_demand

                new_mult = new_mult * new_demand_multiplier
            else:
                curr_multipliers = base_pat.multipliers * new_mult

            # add the last 24 hours of multipliers to the exisiting pattern
            if self.timestep_day == 1:
                node_pat.multipliers = curr_multipliers
            else:
                node_pat.multipliers = np.concatenate(
                    (node_pat.multipliers, curr_multipliers)
                )

            # del curr_node.demand_timeseries_list[0]
            # curr_node.demand_timeseries_list.append((curr_demand, new_pat))

    def run_hyd_hour(self):
        '''
        Run the hydraulic simulation for one hour at a time.
        '''
        # update demands for each node in the network based on the number of
        # agents at the node
        step_agents = list()
        for i, node in enumerate(self.nodes_w_demand):
            # we need to iterate through all nodes with demand to be able to
            # run the hydraulic simualation.
            if node in self.nodes_capacity:
                # curr_node = self.wn.get_node(node)
                capacity_node = self.nodes_capacity[node]
                agents_at_node_list = self.grid.G.nodes[node]['agent']
                agents_at_node = len(agents_at_node_list)
                step_agents.append(agents_at_node)
                if capacity_node != 0:
                    self.sim._en.ENsetnodevalue(
                      self.node_index[node],
                      EN.BASEDEMAND,
                      self.base_demands[node] * agents_at_node / capacity_node
                    )
                    # curr_node.demand_timeseries_list.base_value = (
                    #     self.base_demands[node] * agents_at_node / capacity_node
                    # )
                # else:
                #     # if the node does not have a capacity then we don't need
                #     # to change its demand
                #     pass

        # need to save the agent list
        self.agent_matrix[self.timestep] = step_agents

        # TO DO add daily check to see if agents are working from home
        # and change demand pattern accordingly.

        self.sim.set_next_stop_time(3600 * (self.timestep + 1))
        success = False
        while not success:
            success, stop_conditions = self.sim.run_sim()

        if ((self.timestep + 1) / 24) % 7 == 0:
            self.check_water_age()
            # print(self.water_age_slope)

        # print(3600 * (self.timestep+1))
        # print(self.sim._results.node['demand'])
        # print(3600 * (self.timestep))
        # if self.timestep != 0:
        # self.current_demand = (
        #     self.sim._results.node['demand'].loc[3600 * (self.timestep+1), :]
        # )

    def run_hyd_monthly(self):
        '''
        Run the hydraulic simulation for a month. This method handles all funcs
        necessary to run the monthly simulation, including collecting the demand
        values for each node at each hour, making the new demand patterns each
        day, and running the simulation every month.
        '''
        # first we need to collect the nodal demands at each hour. That means
        # we run self.collect_demands() each time this method is called.
        self.collect_demands()
        # if the timestep is a day then we need to update the demand patterns
        if (self.timestep + 1) % 24 == 0 and self.timestep != 0:
            self.change_demands()
        # if the timestep is the beginning of a week then we want to run the sim
        # also run the sim at the end of the simulation
        if (((self.timestep + 1) / 24) % 30 == 0 and self.timestep != 0 or
           (self.timestep + 1) / 24 == self.days):
            # first set the demand patterns for each node
            for node in self.nodes_w_demand:
                if node in self.nodes_capacity:
                    node_pattern = self.wn.get_pattern('node_'+node)
                    self.sim._en.ENsetpattern(
                        self.sim._en.ENgetpatternindex('node_'+node),
                        node_pattern.multipliers,
                    )
                    # pIndex = self.sim._en.ENgetnodevalue(
                    #     self.sim._en.ENgetnodeindex(node),
                    #     EN.PATTERN
                    # )
                    # demand = self.sim._en.ENgetnodevalue(
                    #     self.sim._en.ENgetnodeindex(node),
                    #     EN.BASEDEMAND
                    # )
                    # print(f"Node {node} has pattern {self.sim._en.ENgetpatternid(int(pIndex))}")
                    # print(f"Node {node} has basedemand {demand}")

            # run the simulation
            self.sim.set_next_stop_time(3600 * (self.timestep + 1))
            success = False
            while not success:
                success, stop_conditions = self.sim.run_sim()
            self.check_water_age()
            # print(self.water_age_slope)

            # update household avoidance behaviors and demand values
            # we don't want to update behaviors during the warmup period
            if not self.warmup and self.bw:
                for node, houses in self.households.items():
                    # demand_list = list()
                    for house in houses:
                        node_age = self.sim._results.node['quality'].loc[:, node]
                        # print(node_age.iloc[-1])
                        house.update_household(node_age.iloc[-1] / 3600)
                        # demand_list.append(house.change)

                    ''' set the demand multiplier based on the average
                    household demand at the node '''
                    # self.demand_multiplier[node] = (
                    #     sum(demand_list) / len(demand_list)
                    # )
                # print(self.demand_multiplier)

                ''' collect household level data '''
                self.collect_household_data()
                # self.traditional[self.timestep], self.burden[self.timestep] = self.calc_equity_metrics(
                #     np.array(self.income),
                #     np.array(self.bw_cost[self.timestep] + self.tw_cost[self.timestep])
                # )

            # if we aren't allowing bottled water buying then we still need to
            # calculate the cost of tap water
            if not self.warmup and not self.bw:
                step_tw_cost = list()
                # for each house, calculate the demand and cost of tap water
                for node, houses in self.households.items():
                    for house in houses:
                        # house.calc_demand()
                        house.calc_tap_cost()
                        house.tap_demand = 0
                        step_tw_cost.append(dcp(house.tap_cost))
                self.tw_cost[self.timestep] = step_tw_cost

    def run_hydraulic(self):
        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim('id' + str(self.id))

        demand = results.node['demand'] * 1000000
        demand = demand.astype('int')
        self.demand_matrix = demand
        self.pressure_matrix = results.node['pressure']
        self.age_matrix = results.node['quality']
        flow = results.link['flowrate'] * 1000000
        flow = flow.astype('int')
        self.flow_matrix = flow
        # results.to_pickle(str(self.id) + 'out_results.pkl')

    def inform_status(self):
        info_stat_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].information == 1:
                info_stat_all += 1
            else:
                pass
        if self.verbose == 1:
            print('\tPeople informed: ' + str(info_stat_all))

    def compliance_status(self):
        compl_stat_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].compliance == 1:
                compl_stat_all += 1
            else:
                pass

        if self.verbose == 1:
            print('\tPeople complying: ' + str(compl_stat_all))

    def check_agent_change(self):
        ''' Function to check each agent for a change in household COVID '''
        for agent in self.schedule.agents:
            # assumes the person talks with household members immediately after
            # finding out about COVID infection
            if agent.infectious_time == 0:
                pass
            elif agent.infectious_time == 1 or agent.infectious_time % (24*5) == 0:
                if agent.home_node != 'TN1372':
                    agent.change_house_adj()

    def check_agent_loc(self):
        self.wrong_node = 0
        for i, agent in enumerate(self.schedule.agents):
            if agent.pos == agent.work_node:
                if agent not in self.grid.G.nodes[agent.work_node]['agent']:
                    self.wrong_node += 1
                    print(f"Agent {agent} is not at its work node.")
            elif agent.pos == agent.home_node:
                if agent not in self.grid.G.nodes[agent.home_node]['agent']:
                    self.wrong_node += 1
                    print(agent in self.agents_moved)
                    print(f"Agent {agent} is not at its home node.")

        print(self.wrong_node)

    def check_covid_change(self):
        covid_change = 0
        for agent in self.schedule.agents:
            if agent.adj_covid_change == 1 and agent.wfh == 0:
                covid_change += 1

        return covid_change

    def update_patterns(self):
        '''
        Lengthen the 5 built-in patterns to match the number of days in the 
        simulation
        '''
        for i in range(1, 6, 1):
            curr_pat = self.wn.get_pattern(str(i))
            curr_mult = dcp(curr_pat.multipliers)
            for j in range(self.days):
                curr_pat.multipliers = np.concatenate((curr_pat.multipliers, curr_mult))

    def plot_water_age(self, age):
        '''
        Saves a plot of the water age.

        Parameters
        ----------
        age : list
            age values
        '''
        age = age.mean(axis=1).rolling(24).mean()
        age.plot()
        plt.savefig(
            'water_age_'+str(self.timestep)+'.png',
            format='png',
            bbox_inches='tight'
        )
        plt.close()

    def check_water_age(self):
        '''
        Check the difference in water age between the first and last timestep
        in the current hydraulic results
        '''
        # get the most recent results
        curr_results = self.sim.get_results()
        # parse last and last water age lists
        mean_age = curr_results.node['quality'].loc[:, self.age_nodes].mean(axis=1) / 3600
        # print(mean_age)
        last_age = mean_age.iloc[-1]
        first_age = mean_age.iloc[1]
        # print(first_age)
        # print(len(mean_age))
        # self.plot_water_age(curr_results.node['quality'].loc[:, self.age_nodes] / 3600)
        # calculate the difference between last and penultimate timesteps
        # and sum a total error.
        # for node in self.nodes_w_demand:
        #     error = last_age[node] - pen_age[node]
        #     total_error += error
        self.water_age_slope = (last_age - first_age) / len(mean_age)

    def collect_agent_data(self):
        ''' BBN input containers '''
        step_cov_pers = list()
        step_cov_ff = list()
        step_media = list()

        ''' BBN output containers '''
        step_wfh = list()
        step_dine = list()
        step_groc = list()
        step_ppe = list()

        for agent in self.schedule.agents:
            step_cov_pers.append(dcp(agent.agent_params['COVIDexp']))
            step_cov_ff.append(dcp(agent.agent_params['COVIDeffect_4']))
            step_media.append(dcp(agent.agent_params['MediaExp_3']))

            step_wfh.append(dcp(agent.wfh))
            step_dine.append(dcp(agent.no_dine))
            step_groc.append(dcp(agent.less_groceries))
            step_ppe.append(dcp(agent.ppe))

        self.cov_pers[self.timestep] = step_cov_pers
        self.cov_ff[self.timestep] = step_cov_ff
        self.media_exp[self.timestep] = step_media

        self.wfh_dec[self.timestep] = step_wfh
        self.dine_dec[self.timestep] = step_dine
        self.groc_dec[self.timestep] = step_groc
        self.ppe_dec[self.timestep] = step_ppe

    def collect_household_data(self):
        ''' Income output containers '''
        step_bw_cost = list()
        step_tw_cost = list()
        step_bw_demand = list()
        step_hygiene = list()
        step_drink = list()
        step_cook = list()

        for node, houses in self.households.items():
            for house in houses:
                step_bw_cost.append(dcp(house.bottle_cost))
                step_tw_cost.append(dcp(house.tap_cost))
                step_bw_demand.append(dcp(house.bottle_demand))
                hygiene = 1 if 'hygiene' in house.bottle else 0
                drink = 1 if 'drink' in house.bottle else 0
                cook = 1 if 'cook' in house.bottle else 0
                step_hygiene.append(hygiene)
                step_drink.append(drink)
                step_cook.append(cook)

        self.bw_cost[self.timestep] = step_bw_cost
        self.tw_cost[self.timestep] = step_tw_cost
        self.bw_demand[self.timestep] = step_bw_demand

        self.hygiene[self.timestep] = step_hygiene
        self.drink[self.timestep] = step_drink
        self.cook[self.timestep] = step_cook

    def calc_equity_metrics(self, income, cow):
        '''
        Calculate the equity metrics for the given time period.

        See:
        https://nicholasinstitute.duke.edu/water-affordability/affordability/about_dashboard.html#metrics

        Parameters
        ----------
        income : np.array
            income for each household

        cow : np.array
            cost of water for each household for the given timestep

        '''
        # scale the income to the number of days passed in the simulation
        scaled_income = income * self.timestep_day / 365

        # calculate the median and 20th percentile income values
        median_i = np.median(scaled_income)
        bot20_i = np.quantile(scaled_income, 0.2)

        # calculate the median water bill
        median_cow = np.median(cow)

        # caculate traditional and household burden
        traditional = median_cow / median_i
        household_burden = median_cow / bot20_i

        return traditional, household_burden

    def daily_tasks(self):
        ''' Increment day time step '''
        self.timestep_day += 1

        if not self.warmup:
            ''' Check status of agents with COVID exposure and infections '''
            for i, agent in enumerate(self.schedule.agents):
                if agent.covid == 'exposed':
                    agent.check_infectious()
                elif agent.covid == 'infectious':
                    agent.check_symptomatic()
                    if agent.inf_severity >= 1:
                        agent.check_severity()
                    else:
                        pass
                    agent.check_recovered()
                    agent.check_death()

                if agent.adj_covid_change == 1:  # this means that agents only check wfh/no_dine/grocery, a max of 5 times throughout the 90 days
                    if agent.wfh == 0 and 'wfh' in self.bbn_models:
                        agent.predict_wfh()

                    if agent.no_dine == 0 and 'dine' in self.bbn_models:
                        agent.predict_dine_less()

                    if agent.less_groceries == 0 and 'grocery' in self.bbn_models:
                        agent.predict_grocery()

                    if agent.ppe == 0 and 'ppe' in self.bbn_models:
                        agent.predict_ppe()

            # collect the bbn and sv data for each agent
            self.collect_agent_data()

            ''' Set the wfh threshold if lag time has been reached '''
            if self.stat_tot[3] > self.wfh_lag and not self.wfh_thres:
                self.wfh_thres = True

    def eos_tasks(self):
        ''' End of simulation tasks '''
        # collect the demands from each node each hour
        self.collect_demands()
        if (self.timestep + 1) / 24 == self.days:
            # setup demand patterns so they extend self.days
            self.update_patterns()
            self.wn.options.time.duration = 3600 * 24 * self.days
            self.wn.options.quality.parameter = 'AGE'
            # run the hydraulic simulation and collect results
            self.run_hydraulic()
        if (self.timestep + 1) % 24 == 0:
            # update the demand patterns with the current days demand
            self.change_demands()


        # you can add here for running by 1hour:


    def industry_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.ind_nodes]

    def commercial_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.com_nodes]

    def rest_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.cafe_nodes]

    def resident_agents(self):
        return [a for a in self.schedule.agents if a.pos == a.home_node]

    def infect_agent(self, agent, next_loc):
        if agent.covid == 'infectious' and not self.warmup:
            self.contact(agent, next_loc)

    def agents_wfh(self):
        return [a for a in self.schedule.agents if a.wfh == 1]

    def print_func(self):
        p1 = ['t', 'S', 'E', 'I', 'R', 'D']
        p2 = ['Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I', 'wfh']
        print('Hour step: ' + str(self.timestepN))
        print('Time step: ' + str(self.timestep))
        print('Day step: ' + str(self.timestep_day))
        print('\n')
        out_1 = ''
        out_2 = ''
        counter = 0
        for i, item in enumerate(self.stat_tot):
            if i < 6:
                out_1 = out_1 + ' ' + p1[i] + ': ' + '{:.2f}'.format(item)
            else:
                out_2 = out_2 + ' ' + p2[counter] + ': ' + '{:.2f}'.format(item)
                counter += 1

        print('\t', out_1)
        print('\t', out_2)
        # print('\tStatus (%): ', ['{:.2f}'.format(i) for i in self.stat_tot])
        # print('\tStatus (#): ', ['{:.3f}'.format(i) * self.num_agents for i in self.stat_tot])
        print('\n')
        print('\tAgents at industrial nodes: ' + str(len(self.industry_agents())))
        print('\tAgents at commercial nodes: ' + str(len(self.commercial_agents())))
        print('\tAgents at restaurant nodes: ' + str(len(self.rest_agents())))
        print('\n')
        print('\tAgents at home: ' + str(len(self.resident_agents())))
        print('\n')
        print('\tAgents with close COVID: ' + str(self.check_covid_change()))

    def step(self):
        self.schedule.step()
        if self.timestep != 0:
            self.move()
        # self.check_agent_loc()
        # self.check_covid_change()
        # BV: changed times to 6, 14, and 22 because I think this is more representative
        # of a three shift schedule. Unsure if this changes anything with water
        # patterns, but I suspect it might.
        if self.timestep == 0 or self.timestepN == 5 or self.timestepN == 13 or self.timestepN == 21:
            self.move_indust()

        # COVID related methods are not run during warmup
        if not self.warmup:
            self.check_status()
            self.check_agent_change()
            if self.res_pat_select == 'pysimdeum':
                self.check_houses()
            self.communication_utility()
            # self.collect_agent_data()

        # daily updating is done during warmup
        if (self.timestep + 1) % 24 == 0 and self.timestep != 0:
            self.daily_tasks()

        if self.hyd_sim == 'eos':
            self.eos_tasks()
        elif self.hyd_sim == 'hourly':
            self.run_hyd_hour()
        elif self.hyd_sim == 'monthly':
            self.run_hyd_monthly()
        else:
            NotImplementedError(f"Hydraulic simultion {self.hyd_sim} not set up.")

        self.timestepN = self.timestep + 1 - self.timestep_day * 24
        # self.inform_status()
        # self.compliance_status()
        if not self.warmup:
            self.num_status()
        if self.verbose == 1:
            self.print_func()
        self.timestep += 1

        if self.water_age_slope < self.tol:
            self.warmup = False
