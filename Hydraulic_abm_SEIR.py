# Consumer Model WNTR- MESA
import warnings
from mesa import Model
from mesa.time import RandomActivation
from utils import setup
import networkx as nx
from mesa.space import NetworkGrid
from agent_model import ConsumerAgent
import math
import time
import bnlearn as bn
import numpy as np
import pandas as pd
# from pysimdeum import pysimdeum
import copy
import wntr

warnings.simplefilter("ignore", UserWarning)


class ConsumerModel(Model):

    def __init__(self,
                 N,
                 city,
                 days=90,
                 id=0,
                 **kwargs):

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
        if 'seed' in kwargs:
            seed = kwargs['seed']
            self.swn = nx.watts_strogatz_graph(n=self.num_agents, p=0.2, k=6, seed=seed)
        else:
            self.swn = nx.watts_strogatz_graph(n=self.num_agents, p=0.2, k=6, seed=919)
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

        self.sleep = setup_out[4]['sleep']
        self.radio = copy.deepcopy(setup_out[4]['radio'])
        self.tv = copy.deepcopy(setup_out[4]['tv'])
        self.bbn_params = setup_out[5]  # pandas dataframe of bbn parameters
        wfh_patterns = setup_out[6]
        self.terminal_nodes = setup_out[7]
        self.wn = setup_out[8]

        # set up water network
        # if city == 'micropolis':
        #     inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
        # elif city == 'mesopolis':
        #     inp_file = 'Input Files/Mesopolis.inp'

        self.G = self.wn.get_graph()
        self.grid = NetworkGrid(self.G)
        self.num_nodes = len(self.G.nodes)
        # wn.options.time.duration = 0
        self.wn.options.time.hydraulic_timestep = 3600
        self.wn.options.time.pattern_timestep = 3600
        self.wn.options.quality.parameter = 'AGE'
        for i in range(wfh_patterns.shape[1]):
            self.wn.add_pattern('wk'+str(i+1), np.array(wfh_patterns.iloc[:, i]))

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
        self.nodes_w_demand = [
            node for node in self.grid.G.nodes
            if hasattr(self.wn.get_node(node), 'demand_timeseries_list')
        ]
        self.daily_demand = pd.DataFrame(0, index = np.arange(0, 86400, 3600), columns = self.nodes_w_demand)
        self.demand_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                          columns=self.G.nodes)
        # self.agent_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[node for node in self.nodes_w_demand if node in self.nodes_capacity])
        self.pressure_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                            columns=self.G.nodes)
        self.age_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                       columns=self.G.nodes)
        self.flow_matrix = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600),
                                        columns=[name for name, link in self.wn.links()])
        # self.daily_demand = list()
        # self.demand_matrix = dict()
        self.agent_matrix = dict()

        ''' Initialize the COVID state variable collectors '''
        # self.cov_pers = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        # self.cov_ff = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        # self.media_exp = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        self.cov_pers = dict()
        self.cov_ff = dict()
        self.media_exp = dict()

        ''' Initialize the PM adoption collectors '''
        # self.wfh_dec = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        # self.dine_dec = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        # self.groc_dec = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        # self.ppe_dec = pd.DataFrame(0, index=np.arange(0, 86400*days, 3600), columns=[str(i) for i in range(self.num_agents)])
        self.wfh_dec = dict()
        self.dine_dec = dict()
        self.groc_dec = dict()
        self.ppe_dec = dict()

        # Set values for susceptibility based on age. From https://doi.org/10.1371/journal.pcbi.1009149
        self.susDict = {1: [0.525, 0.001075, 0.000055, 0.00002],
                        2: [0.6, 0.0072, 0.00036, 0.0001],
                        3: [0.65, 0.0208, 0.00104, 0.00032],
                        4: [0.7, 0.0343, 0.00216, 0.00098],
                        5: [0.75, 0.07650, 0.00933, 0.00265],
                        6: [0.8, 0.1328, 0.03639, 0.00766],
                        7: [0.85, 0.20655, 0.08923, 0.02439],
                        8: [0.9, 0.2457, 0.1742, 0.08292],
                        9: [0.9, 0.2457, 0.1742, 0.1619]}

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
        # self.status_tot = np.array([self.status_tot])
        #  ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I', 'wfh'])

        self.base_demand_list()
        self.create_node_list()
        if self.verbose == 1:
            print("Creating agents ..............")
        self.create_agents()
        if self.verbose == 1:
            print("Setting agent attributes ...............")
        self.set_attributes()
        self.dag_nodes()
        if self.res_pat_select == 'pysimdeum':
            self.create_demand_houses()
        # self.create_comm_network()

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
        self.wfh_nodes = copy.deepcopy(self.wfh_dag['adjmat'].columns)
        self.dine_nodes = copy.deepcopy(self.dine_less_dag['adjmat'].columns)
        self.grocery_nodes = copy.deepcopy(self.grocery_dag['adjmat'].columns)
        self.ppe_nodes = copy.deepcopy(self.ppe_dag['adjmat'].columns)

    def base_demand_list(self):
        self.base_demands = dict()
        self.base_pattern = dict()
        for node in self.nodes_w_demand:
            node_1 = self.wn.get_node(node)
            self.base_demands[node] = node_1.demand_timeseries_list[0].base_value

            ''' Make a pattern for each node for hydraulic simulation '''
            curr_pattern = copy.deepcopy(node_1.demand_timeseries_list[0].pattern)
            # curr_pattern.name = 'node_'+node
            self.wn.add_pattern('node_'+node, curr_pattern.multipliers)
            self.base_pattern[node] = copy.deepcopy(node_1.demand_timeseries_list[0].pattern_name)

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
        no_wfh_ind_nodes = self.random.choices(population=self.ind_nodes,
                                               k=int(len(self.ind_nodes)*self.no_wfh_perc))
        # no_wfh_comm_nodes = self.random.choices(population=self.com_nodes,
        #                                         k=int(len(self.com_nodes)*0.2))
        # no_wfh_rest_nodes = self.random.choices(population=self.cafe_nodes,
        #                                         k=int(len(self.cafe_nodes)*0.2))
        total_no_wfh = no_wfh_ind_nodes# + no_wfh_comm_nodes + no_wfh_rest_nodes

        work_agents = (max(self.ind_dist) + max(self.nav_dist)) * 2
        # rest_agents = max(self.cafe_dist)
        # comm_agents = max(self.com_dist)
        # CREATING AGENTS
        ''' Needed to account for multifamily housing, so iterating through
        residential nodes and placing agents that way and then storing their
        housemates in the agent object. '''
        res_nodes = copy.deepcopy(self.res_nodes)
        self.random.shuffle(res_nodes)
        ids = 0
        for node in res_nodes:
            curr_node = list()
            for spot in range(int(self.nodes_capacity[node])):
                a = ConsumerAgent(ids, self)
                self.schedule.add(a)
                if work_agents != 0:
                    a.work_node = self.random.choice(self.work_loc_list)
                    a.home_node = node
                    # a.home_node = self.random.choice(self.res_loc_list)
                    self.work_loc_list.remove(a.work_node)
                    # self.res_loc_list.remove(a.home_node)
                    if a.work_node in self.nav_nodes:
                        a.work_type = 'navy'
                    elif a.work_node in self.ind_nodes:
                        a.work_type = 'industrial'
                    work_agents -= 1
                # elif rest_agents != 0:
                #     a.work_node = self.random.choice(self.rest_loc_list)
                #     a.home_node = self.random.choice(self.res_loc_list)
                #     self.rest_loc_list.remove(a.work_node)
                #     self.res_loc_list.remove(a.home_node)
                #     a.work_type = 'restaurant'
                #     rest_agents -= 1
                # elif comm_agents != 0:
                #     a.work_node = self.random.choice(self.comm_loc_list)
                #     a.home_node = self.random.choice(self.res_loc_list)
                #     self.comm_loc_list.remove(a.work_node)
                #     self.res_loc_list.remove(a.home_node)
                #     a.work_type = 'commercial'
                #     comm_agents -= 1
                else:
                    a.home_node = node
                    # self.res_loc_list.remove(a.home_node)

                if a.work_node in total_no_wfh:
                    a.can_wfh = False
                self.grid.place_agent(a, a.home_node)
                curr_node.append(a.unique_id)
                ids += 1

            if self.network == 'micropolis':
                self.micro_household(curr_node)
            elif self.network == 'mesopolis':
                self.meso_household(curr_node)

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

    def micro_household(self, curr_node):
        ''' Assigns each agent's housemates based on the number
        of agents at the current node.

        If the residential node is larger than 6, then we need to
        artificially make households of 6 or less. '''
        if len(curr_node) > 6:  # multifamily housing
            ''' Iterate through the agents at the current res node
            and add them to households of 1 to 6 agents '''
            while len(curr_node) > 6:
                # pick a random size for current household
                home_size = self.random.choice(range(1, 7))
                # pick the agents that will occupy this household
                curr_housemates = self.random.choices(curr_node, k=home_size)
                # remake the curr_node variable without the agents just chosen
                curr_node = [a for a in curr_node if a not in curr_housemates]
                # pick an income
                curr_income = 
                ''' Assign the current list of housemates to each agent
                so they each know who their housemates are '''
                for mate in curr_housemates:
                    agent = self.schedule._agents[mate]
                    # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                    agent.housemates = copy.deepcopy(curr_housemates)  # this includes current agent

            for mate in curr_node:
                agent = self.schedule._agents[mate]
                # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                agent.housemates = copy.deepcopy(curr_node)
        else:
            for agent in curr_node:
                curr_agent = self.schedule._agents[agent]
                # agent = [a for a in self.schedule.agents if a.unique_id == agent][0]
                curr_agent.housemates = copy.deepcopy(curr_node)

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
                agent.housemates = copy.deepcopy(curr_housemates)  # this includes current agent

        for mate in curr_node:
            agent = self.schedule._agents[mate]
            # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
            agent.housemates = copy.deepcopy(curr_node)

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
        exposed_sample = self.random.sample([i for i, a in enumerate(self.schedule.agents)],
                                               self.covid_exposed)
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
            agent_set_params = self.bbn_params.sample()
            # agent.agent_params['risk_perception_r'] = int(agent_set_params['risk_perception_r']) - 1
            for param in self.bbn_params:
                try:
                    # if param == "COVIDeffect_4":
                    #     pass
                    if param == "DemEdu":
                        if int(agent_set_params[param]) == 9:
                            agent.agent_params[param] = 5
                        else:
                            agent.agent_params[param] = int(agent_set_params[param]) - 1
                    elif param == "Ethnicmin":
                        if int(agent_set_params[param]) == 4:
                            agent.agent_params[param] = 2
                        else:
                            agent.agent_params[param] = int(agent_set_params[param]) - 1
                    elif param == "COVIDexp":
                        agent.agent_params[param] = 7
                    elif param == "MediaExp_3":
                        agent.agent_params[param] = 1
                    else:
                        agent.agent_params[param] = int(agent_set_params[param]) - 1
                except:
                    pass

    def create_comm_network(self):
        '''
        CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
        Assigning Agents randomly to nodes in SNW
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
        self.stat_tot = [self.timestep, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.cumm_infectious, len(self.agents_wfh())]

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
        # step_status = pd.DataFrame([self.stat_tot], columns = ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I', 'wfh'])
        self.status_tot[self.timestep] = copy.deepcopy(self.stat_tot)

    def contact(self, agent_to_move, node_type):
        """
        Function to test whether a specific agent has exposed the other agents
        in their node to covid. This is currently called everytime an agent moves
        to a new location and NOT every hour.
        """
        if len(self.grid.G.nodes[agent_to_move.pos]['agent']) > 6:
            agents_at_node = self.grid.G.nodes[agent_to_move.pos]['agent']
            agents_to_expose = self.random.choices(population = agents_at_node,
                                                   k = self.daily_contacts)
        else:
            agents_to_expose = self.grid.G.nodes[agent_to_move.pos]['agent']

        for agent in agents_to_expose:
            # agent.adj_covid_change = 1
            # if agent.agent_params["COVIDeffect_4"] < 6 and node_type == 'residential':
            #     agent.agent_params["COVIDeffect_4"] += 0.1
            # else:
            #     pass

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
                agent.exp_time += 1
            elif agent.covid == 'infectious':
                agent.infectious_time += 1
                # Add time to sev_time and crit_time if agent is in severe or
                # critical state. Also, reset sev_time if agent is critical, to
                # remove the chance of the agent being considered in both categories.
                if agent.inf_severity == 2:
                    agent.sev_time += 1
                elif agent.inf_severity == 3:
                    agent.crit_time +=1
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

    def check_infectious(self, agent):
        """
        Function to check if agent has been exposed for sufficient time to be
        infectious. Agent is not symptomatic yet, but can still transmit disease.

        Function is typically run from loop over all agents at each DAY step.
        """
        if agent.exp_time >= 24 * math.log(self.random.lognormvariate(self.e2i[0],self.e2i[1])):
            agent.covid = 'infectious'
            self.cumm_infectious += 1
            agent.exp_time = 0
            agent.agent_params["COVIDexp"] = 1
        else:
            pass

    def check_symptomatic(self, agent):
        """
        Function to check if agent has been infectious for sufficient time to be
        either symptomatic or asymptomatic. Once asymptomatic, there is a set time
        until recovered, otherwise, moves to mild, severe, or critical depending
        on age.

        Function is typically run from loop over all agents at each DAY step.
        """
        inf_time = 24 * math.log(self.random.lognormvariate(self.i2s[0],self.i2s[1]))
        if agent.symptomatic == None and agent.infectious_time >= inf_time:
            if self.random.random() < self.susDict[agent.age][0]:
                agent.symptomatic = 1
                agent.inf_severity = 1
            else:
                agent.symptomatic = 0
        else:
            pass

    def check_severity(self, agent):
        """
        Function to check the agents infection severity. Severity is based on the
        time the agent has been symptomatic and their age.

        Function is typically run from loop over all agents at each DAY step.
        """
        sev_time = 24 * math.log(self.random.lognormvariate(self.s2sev[0],self.s2sev[1]))
        crit_time = 24 * math.log(self.random.lognormvariate(self.sev2c[0],self.sev2c[1]))
        sev_prob = self.susDict[agent.age][1]
        crit_prob = self.susDict[agent.age][2]
        if agent.inf_severity == 1 and agent.symp_time >= sev_time and self.random.random() < sev_prob:
            agent.inf_severity = 2
        elif agent.inf_severity == 2 and agent.sev_time >= crit_time and self.random.random() < crit_prob:
            agent.inf_severity = 3
        else:
            pass

    def check_recovered(self, agent):
        """
        Function to check if agent has been infectious for sufficient time to be
        recorvered. Depends on whether the agent was asymptomatic or symptomatic.

        Function is typically run from loop over all agents at each DAY step.
        """
        asymp_time = 24 * math.log(self.random.lognormvariate(self.recTimeAsym[0],self.recTimeAsym[1]))
        mild_time = 24 * math.log(self.random.lognormvariate(self.recTimeMild[0],self.recTimeMild[1]))
        sevRec_time = 24 * math.log(self.random.lognormvariate(self.recTimeSev[0],self.recTimeSev[1]))
        critRec_time = 24 * math.log(self.random.lognormvariate(self.recTimeC[0],self.recTimeC[1]))
        if agent.symptomatic == 0 and agent.infectious_time >= asymp_time:
            agent.covid = 'recovered'
            agent.infectious_time = 0
            agent.symptomatic = None
        else:
            pass

        if agent.inf_severity == 1 and agent.symp_time >= mild_time:
            agent.covid = 'recovered'
            agent.infectious_time = 0
            agent.symptomatic = None
        elif agent.inf_severity == 2 and agent.sev_time >= sevRec_time:
            agent.covid = 'recovered'
            agent.infectious_time = 0
            agent.symptomatic = None
        elif agent.inf_severity == 3 and agent.crit_time >= critRec_time:
            agent.covid = 'recovered'
            agent.infectious_time = 0
            agent.symptomatic = None
        else:
            pass

    def check_death(self, agent):
        """
        Function to check if agent has been critical for sufficient time to be
        potentially dead. Depends on agents age. Removes agent from grid and
        sets agents position to None.

        Function is typically run from loop over all agents at each DAY step.
        """
        death_prob = self.susDict[agent.age][3]
        death_time = 24 * math.log(self.random.lognormvariate(self.c2d[0],self.c2d[1]))
        if agent.inf_severity == 3 and agent.crit_time >= death_time and self.random.random() < death_prob:
            agent.covid = 'dead'
            agent.symptomatic = None
            self.grid.remove_agent(agent)
        else:
            pass

    def check_social_dist(self):
        for i, agent in enumerate(self.schedule.agents):
            if self.random.random() < self.wfh_probs[math.floor(agent.agent_params["COVIDeffect_4"])]:
                agent.wfh = 1

    def communication_utility(self):
        # Communication through TV and Radio
        # Percentage of people listening to radio:
        radio_reach = self.radio[self.timestepN]/100
        tv_reach = self.tv[self.timestepN]/100

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
                if self.base_pattern[location] == '6':
                    if Agent_to_move.less_groceries == 1:
                        pass
                    else:
                        self.grid.move_agent(Agent_to_move, location)
                        Possible_Agents_to_move.remove(Agent_to_move)
                        nodes_comm.remove(location)
                        self.infect_agent(Agent_to_move, 'workplace')
                else:
                    ''' The agents in this arm are considered workers '''
                    if Agent_to_move.wfh == 0:
                        self.grid.move_agent(Agent_to_move, location)
                        Possible_Agents_to_move.remove(Agent_to_move)
                        nodes_comm.remove(location)
                        self.infect_agent(Agent_to_move, 'workplace')
                    else:
                        pass

        elif delta_agents_comm < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
            Possible_Agents_to_move = self.commercial_agents()
            for i in range(min(abs(delta_agents_comm), len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move.remove(Agent_to_move)
                self.infect_agent(Agent_to_move, 'residential')
        else:
            pass

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
                    pass
                else:
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
        else:
            pass

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
                pass
            else:
                self.agents_moved.append(Agent_to_move)
                self.grid.move_agent(Agent_to_move, Agent_to_move.work_node)
                Possible_Agents_to_move_to_work.remove(Agent_to_move)
                self.infect_agent(Agent_to_move, 'workplace')

    def set_patterns(self, node):
        house = [house for house in self.res_houses if house.id == node.name][0]
        consumption = house.simulate(num_patterns=10)
        tot_cons = consumption.sum(['enduse', 'user']).mean([ 'patterns'])
        hourly_cons = tot_cons.groupby('time.hour').mean()
        self.wn.add_pattern('hs_' + house.id,
                       np.array(np.divide(hourly_cons, hourly_cons.mean())))
        node.demand_timeseries_list[0].pattern_name = 'hs_' + house.id

    def collect_demands(self):
        step_demand = list()
        step_agents = dict()
        for node in self.nodes_w_demand:
            if node in self.nodes_capacity:
                Capacity_node = self.nodes_capacity[node]
                # node_1 = self.wn.get_node(node)
                agents_at_node_list = self.grid.G.nodes[node]['agent']
                agents_at_node = len(agents_at_node_list)
                step_agents[node] = agents_at_node
                # agents_wfh = len([a for a in agents_at_node_list if a.wfh == 1])
                if Capacity_node != 0:
                    step_demand.append(agents_at_node/Capacity_node)
                else:
                    step_demand.append(0)
            else:
                step_demand.append(0)

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

        hourly_agents = [agent for key, agent in step_agents.items()]
        hourly_demands = pd.DataFrame(data=[step_demand], index=[0])
        self.daily_demand[self.timestepN:(self.timestepN+1)] = hourly_demands
        self.agent_matrix[self.timestep] = hourly_agents

    def change_demands(self):
        '''
        Add the current days demand pattern to each nodes demand pattern.
        This appends a new 24 pattern based on the demands at each node.
        '''

        for node in self.nodes_w_demand:
            curr_node = self.wn.get_node(node)
            curr_demand = curr_node.demand_timeseries_list[0].base_value
            new_mult = self.daily_demand[node]
            agents_at_node = self.grid.G.nodes[node]['agent']
            agents_wfh = len([a for a in agents_at_node if a.wfh == 1])
            if self.res_pat_select == 'lakewood' and len(agents_at_node) != 0:
                perc_wfh = agents_wfh / len(agents_at_node)
                if perc_wfh > 0.5 and node in self.res_nodes:
                    old_pat = self.wn.get_pattern('wk1')
                else:
                    old_pat = self.wn.get_pattern(self.base_pattern[node])
            elif self.res_pat_select == 'pysimdeum':
                if node in self.res_nodes:
                    self.set_patterns(node_1)
            else:
                old_pat = self.wn.get_pattern(self.base_pattern[node])
            new_pat = self.wn.get_pattern('node_'+node)
            if self.timestep_day == 1:
                new_pat.multipliers = old_pat.multipliers * new_mult
            else:
                new_pat.multipliers = np.concatenate((new_pat.multipliers, old_pat.multipliers * new_mult))

            del curr_node.demand_timeseries_list[0]
            curr_node.demand_timeseries_list.append((curr_demand, new_pat))

    def run_hydraulic(self):
        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim('id' + str(self.id))

        # Assigning first base demand again to individual Nodes so WNTR doesnt add all BD up
        # for node, base_demand in self.base_demands.items():
        #
        #     node_1 = self.wn.get_node(node)
        #     node_1.demand_timeseries_list[0].base_value = base_demand

        # SAVING CURRENT DEMAND TIMESTEP IN DEMAND MATRIX
        # self.demand_matrix[self.timestep-23: self.timestep+1] = results.node['demand'][0:24]
        # self.pressure_matrix[self.timestep-23: self.timestep+1] = results.node['pressure'][0:24]
        # self.age_matrix[self.timestep-23: self.timestep+1] = results.node['quality'][0:24]
        # flow = results.link['flowrate'][0:24] * 1000000
        # flow = flow.astype('int')
        # self.flow_matrix[self.timestep-23: self.timestep+1] = flow

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

    def predict_wfh(self, agent):
        # agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
        # for agent in agents_not_wfh:
        agent.adj_covid_change = 0
        evidence_agent = copy.deepcopy(agent.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.wfh_nodes):
            if item != 'work_from_home':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.wfh_dag,
                                 variables = ['work_from_home'],
                                 evidence = evidence,
                                 verbose = 0)
        if self.random.random() < query.df['p'][1]:
        # if self.random.random() < self.rp_wfh_probs[agent.agent_params['risk_perception_r']]:
            agent.wfh = 1

    def predict_dine_less(self, agent):
        # agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
        # for agent in agents_not_wfh:
        agent.adj_covid_change = 0
        evidence_agent = copy.deepcopy(agent.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.dine_nodes):
            if item != 'dine_out_less':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.dine_less_dag,
                                 variables = ['dine_out_less'],
                                 evidence = evidence,
                                 verbose = 0)
        if self.random.random() < query.df['p'][1]:
        # if self.random.random() < self.rp_wfh_probs[agent.agent_params['risk_perception_r']]:
            agent.no_dine = 1

    def predict_grocery(self, agent):
        # agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
        # for agent in agents_not_wfh:
        agent.adj_covid_change = 0
        evidence_agent = copy.deepcopy(agent.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.grocery_nodes):
            if item != 'shop_groceries_less':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.grocery_dag,
                                 variables=['shop_groceries_less'],
                                 evidence=evidence,
                                 verbose=0)
        if self.random.random() < query.df['p'][1]:
        # if self.random.random() < self.rp_wfh_probs[agent.agent_params['risk_perception_r']]:
            agent.less_groceries = 1

    def predict_ppe(self, agent):
        # agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
        # for agent in agents_not_wfh:
        agent.adj_covid_change = 0
        evidence_agent = copy.deepcopy(agent.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.ppe_nodes):
            if item != 'mask':
                if evidence_agent[item] < 10 and evidence_agent[item] >= 0:
                    evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.ppe_dag,
                                 variables = ['mask'],
                                 evidence = evidence,
                                 verbose = 0)
        if self.random.random() < query.df['p'][1]:
        # if self.random.random() < self.rp_wfh_probs[agent.agent_params['risk_perception_r']]:
            agent.ppe = 1

    def change_house_adj(self, agent):
        ''' Function to check whether agents in a given agents node have become
        infected with COVID '''
        # node = agent.home_node
        # agents_at_node = copy.deepcopy(self.grid.G.nodes[node]['agent'])
        # if len(agents_at_node) > 6:
        # print(agent.housemates)
        agents_in_house = copy.deepcopy(agent.housemates)
        agents_friends = [n for n in self.swn.neighbors(agent.unique_id)]
        agents_in_network = agents_in_house + agents_friends
        # print(agent)
        # print(agents_in_house)
        agents_in_network.remove(agent.unique_id)
        for name in agents_in_network:
            adj_agent = self.schedule._agents[name]
            # agent = [a for a in self.schedule.agents if a.unique_id == agent][0]
            adj_agent.adj_covid_change = 1
            if adj_agent.agent_params["COVIDeffect_4"] < 6:
                adj_agent.agent_params["COVIDeffect_4"] += 1
            else:
                pass
        # else:
        #     agents_at_node.remove(agent)
        #     for a in agents_at_node:
        #         a.adj_covid_change == 1
        #         if a.agent_params["COVIDeffect_4"] < 6:
        #             a.agent_params["COVIDeffect_4"] += 0.1
        #         else:
        #             pass

    def check_agent_change(self):
        ''' Function to check each agent for a change in household COVID '''
        for i, agent in enumerate(self.schedule.agents):
            # assumes the person talks with household members immediately after
            # finding out about COVID infection
            if agent.infectious_time == 0:
                pass
            elif agent.infectious_time == 1 or agent.infectious_time % (24*5) == 0:
                if agent.home_node != 'TN1372':
                    self.change_house_adj(agent)

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
        for i in range(1, 6, 1):
            curr_pat = self.wn.get_pattern(str(i))
            curr_mult = copy.deepcopy(curr_pat.multipliers)
            for j in range(90):
                curr_pat.multipliers = np.concatenate((curr_pat.multipliers, curr_mult))
            # print(curr_pat.multipliers)

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
            step_cov_pers.append(copy.deepcopy(agent.agent_params['COVIDexp']))
            step_cov_ff.append(copy.deepcopy(agent.agent_params['COVIDeffect_4']))
            step_media.append(copy.deepcopy(agent.agent_params['MediaExp_3']))

            step_wfh.append(copy.deepcopy(agent.wfh))
            step_dine.append(copy.deepcopy(agent.no_dine))
            step_groc.append(copy.deepcopy(agent.less_groceries))
            step_ppe.append(copy.deepcopy(agent.ppe))

        self.cov_pers[self.timestep] = step_cov_pers
        self.cov_ff[self.timestep] = step_cov_ff
        self.media_exp[self.timestep] = step_media

        self.wfh_dec[self.timestep] = step_wfh
        self.dine_dec[self.timestep] = step_dine
        self.groc_dec[self.timestep] = step_groc
        self.ppe_dec[self.timestep] = step_ppe

    def change_time_model(self):
        self.timestep += 1
        if self.timestep % 24 == 0:
            # self.wn.options.time.duration = 0
            ''' Increment day time step '''
            self.timestep_day += 1

            ''' Check status of agents with COVID exposure and infections '''
            for i, agent in enumerate(self.schedule.agents):
                if agent.covid == 'exposed':
                    self.check_infectious(agent)
                elif agent.covid == 'infectious':
                    self.check_symptomatic(agent)
                    if agent.inf_severity >= 1:
                        self.check_severity(agent)
                    else:
                        pass
                    self.check_recovered(agent)
                    self.check_death(agent)

                if agent.adj_covid_change == 1:  # this means that agents only check wfh/no_dine/grocery, a max of 5 times throughout the 90 days
                    if agent.wfh == 0 and 'wfh' in self.bbn_models:
                        self.predict_wfh(agent)

                    if agent.no_dine == 0 and 'dine' in self.bbn_models:
                        self.predict_dine_less(agent)

                    if agent.less_groceries == 0 and 'grocery' in self.bbn_models:
                        self.predict_grocery(agent)

                    if agent.ppe == 0 and 'ppe' in self.bbn_models:
                        self.predict_ppe(agent)

            # self.check_social_dist()
            if self.stat_tot[3] > self.wfh_lag and not self.wfh_thres:
                self.wfh_thres = True
            self.change_demands()
        # elif (self.timestep + 1) % 24 == 0 and self.timestep != 0:
            # ''' Clear EPANET files and run hydraulic for the day '''
            # curr_dir = os.getcwd()
            # files_in_dir = os.listdir(curr_dir)
            #
            # for file in files_in_dir:
            #     if file.endswith(".rpt") or file.endswith(".bin") or file.endswith(".inp"):
            #         os.remove(os.path.join(curr_dir, file))
            # self.run_hydraulic()
        # else:
            # pass

        if self.timestep_day == self.days:
            self.update_patterns()
            self.wn.options.time.duration = 3600 * 24 * self.days
            self.run_hydraulic()
        self.timestepN = self.timestep - self.timestep_day * 24
        # print(self.timestep)

    def industry_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.ind_nodes]

    def commercial_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.com_nodes]

    def rest_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.cafe_nodes]

    def resident_agents(self):
        return [a for a in self.schedule.agents if a.pos == a.home_node]

    def infect_agent(self, agent, next_loc):
        if agent.covid == 'infectious':
            self.contact(agent, next_loc)
        else:
            pass

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
            # self.move_wfh()
        # self.check_agent_loc()
        # self.check_covid_change()
        # BV: changed times to 6, 14, and 22 because I think this is more representative
        # of a three shift schedule. Unsure if this changes anything with water
        # patterns, but I suspect it might.
        if self.timestep == 0 or self.timestepN == 6 or self.timestepN == 14 or self.timestepN == 22:
            self.move_indust()
            # self.move_indust_wfh()
        self.check_status()
        self.check_agent_change()
        if self.res_pat_select == 'pysimdeum':
            self.check_houses()
        self.communication_utility()
        self.collect_demands()
        self.collect_agent_data()
        self.change_time_model()
        # self.inform_status()
        # self.compliance_status()
        self.num_status()
        if self.verbose == 1:
            self.print_func()
