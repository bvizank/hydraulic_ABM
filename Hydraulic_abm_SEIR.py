# Consumer Model WNTR- MESA
import warnings
warnings.simplefilter("ignore", UserWarning)
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
# from wntr_1 import *
from Char_micropolis_static_loc import *
import networkx as nx
from mesa.space import NetworkGrid
import random
from agent_model import *
import math
import time
import bnlearn as bn
import multiprocessing as mp
import os
import numpy as np
# from pysimdeum import pysimdeum
import copy

inp_file = 'Input Files/MICROPOLIS_v1_orig_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()
wn.options.time.duration = 0
wn.options.time.hydraulic_timestep = 3600
wn.options.time.pattern_timestep = 3600
wn.options.quality.parameter = 'AGE'
for i in range(wfh_patterns.shape[1]):
    wn.add_pattern('wk'+str(i+1), np.array(wfh_patterns.iloc[:,i]))


class ConsumerModel(Model):
    """A Model with some number of Agents"""
    def __init__(self,
                 N,
                 num_nodes = len(G.nodes),
                 nodes_capacity = Max_pop_pnode_resid,
                 nodes_resident = Nodes_resident,
                 nodes_industr = Nodes_industr,
                 nodes_cafe = Nodes_comm_cafe,
                 nodes_rest = Nodes_comm_rest,
                 all_terminal_nodes = All_terminal_nodes,
                 sum_distr_ph = Sum_distr_ph,
                 Comm_distr_ph = Comm_distr_ph,
                 Comm_rest_distr_ph = Comm_rest_distr_ph,
                 Resident_distr_ph = Resident_distr_ph,
                 industr_distr_ph = Industr_distr_ph,
                 bbn_params = bbn_params,
                 seed = None,
                 days = 90,
                 start_inf = 5,
                 daily_contacts = 10,
                 lag_period = 7,
                 wfh = False,
                 res_pat_select = 'lakewood',
                 wfh_lag = 0):

        init_start = time.perf_counter()
        self.num_agents = N
        self.G = G
        self.num_nodes = num_nodes
        self.nodes_resident = nodes_resident
        self.nodes_industr = nodes_industr
        self.nodes_cafe = nodes_cafe # There is no node assigned to "dairy queen" so it was neglected
        self.nodes_rest = nodes_rest
        self.terminal_nodes = All_terminal_nodes
        self.nodes_capacity = nodes_capacity
        self.resid_distr_ph = Resident_distr_ph
        self.comm_rest_distr_ph = Comm_rest_distr_ph
        self.comm_distr_ph = Comm_distr_ph
        self.industr_distr_ph = industr_distr_ph
        self.sum_distr_ph = Sum_distr_ph
        self.grid = NetworkGrid(self.G)
        self.t = 0
        self.schedule = RandomActivation(self)
        self.timestep = 0
        self.timestep_day = 0
        self.timestepN = 0
        self.base_demands_previous = {}
        self.snw = nx.watts_strogatz_graph(n = Micro_pop, p = 0.2, k = 6, seed = seed)
        self.snw_agents = {}
        self.nodes_endangered = All_terminal_nodes
        self.demand_test = []
        self.covid_exposed = start_inf #round(0.001*N) # number of starting infectious
        self.exposure_rate = 0.05 # infection rate per contact per day in households
        self.exposure_rate_large = 0.01 # infection rate per contact per day in workplaces
        self.e2i = (4.5,1.5) # mean and sd number of days before infection shows
        self.i2s = (1.1,0.9) # time after viral shedding before individual shows sypmtoms
        self.s2sev = (6.6,4.9) # time after symptoms start before individual develops potential severe covid
        self.sev2c = (1.5,2.0) # time after severe symptoms before critical status
        self.c2d = (10.7,4.8) # time between critical dianosis to death
        self.recTimeAsym = (8.0,2.0) # time for revovery for asymptomatic cases
        self.recTimeMild = (8.0,2.0) # mean and sd number of days for recovery: mild cases
        self.recTimeSev = (18.1,6.3)
        self.recTimeC = (18.1,6.3)
        self.daily_contacts = daily_contacts
        self.cumm_infectious = self.covid_exposed
        self.wfh_dag = bn.import_DAG('Input Files/data_driven_wfh.bif')
        self.bbn_params = bbn_params # pandas dataframe of bbn parameters
        self.lag_period = lag_period # number of days to wait before social distancing
        self.model_wfh = wfh
        self.res_pat_select = res_pat_select
        self.wfh_lag = wfh_lag # infection percent before work from home allowed
        self.wfh_thres = False # whether wfh lag has been reached

        """
        Save parameters to a DataFrame, param_out, to save at the end of the
        simulation. This helps with data organization.
        """
        self.param_out = pd.DataFrame(columns = ['Param', 'value1', 'value2'])
        self.param_out = self.param_out.append(pd.DataFrame([['covid_exposed', self.covid_exposed]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['household_rate', self.exposure_rate]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['workplace_rate', self.exposure_rate_large]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['infection_time', self.e2i[0], self.e2i[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['sypotimatic_time', self.i2s[0], self.i2s[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['severe_time', self.s2sev[0], self.s2sev[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['critical_time', self.sev2c[0], self.sev2c[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['death_time', self.c2d[0], self.c2d[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['asymp_recovery_time', self.recTimeAsym[0], self.recTimeAsym[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['mild_recovery_time', self.recTimeMild[0], self.recTimeMild[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['severe_recovery_time', self.recTimeSev[0], self.recTimeSev[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['critical_recovery_time', self.recTimeC[0], self.recTimeC[1]]],
                                                            columns = ['Param', 'value1', 'value2']))
        self.param_out = self.param_out.append(pd.DataFrame([['daily_contacts', self.daily_contacts]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['lag_period', self.lag_period]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['wfh', self.model_wfh]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['res pattern', self.res_pat_select]],
                                                            columns = ['Param', 'value1']))
        self.param_out = self.param_out.append(pd.DataFrame([['wfh_lag', self.wfh_lag]],
                                                            columns = ['Param', 'value1']))

        self.demand_matrix = pd.DataFrame(0, index = np.arange(0, 86400*days, 3600), columns = G.nodes)
        self.pressure_matrix = pd.DataFrame(0, index = np.arange(0, 86400*days, 3600), columns = G.nodes)
        self.age_matrix = pd.DataFrame(0, index = np.arange(0, 86400*days, 3600), columns = G.nodes)
        self.node_num = pd.DataFrame(0, index = np.arange(0, 86400*days, 3600), columns = G.nodes)

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

        self.status_tot = [0,self.num_agents-self.covid_exposed,0,self.covid_exposed,0,0,0,0,0,0,0,self.cumm_infectious, len(self.agents_wfh())]
        self.status_tot = np.divide(self.status_tot, self.num_agents)
        self.status_tot = pd.DataFrame([self.status_tot], columns = ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I', 'wfh'])

        self.create_node_list()
        self.create_agents()
        self.set_attributes()
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

        print('Time to initialize: ', init_stop - init_start)


    def node_list(self, list, nodes):
        list_out = []
        for node in nodes:
            for i in range(int(list[node])):
                list_out.append(node)
        return list_out

    def create_node_list(self):
        nodes_industr_2x = self.nodes_industr + self.nodes_industr + self.nodes_industr
        self.ind_loc_list = self.node_list(self.nodes_capacity, nodes_industr_2x)
        self.res_loc_list = self.node_list(self.nodes_capacity, self.nodes_resident)
        # self.rest_loc_list = node_list(self.nodes_capacity, self.nodes_cafe)
        # self.comm_loc_list = node_list(self.nodes_capacity, self.nodes_rest)


    def create_agents(self):
        ''' Creating lists of nodes where employers have decided not to allow
        working from home or jobs that are "essential". '''
        no_wfh_ind_nodes = self.random.choices(population=self.nodes_industr,
                                               k=int(len(self.nodes_industr)*0.5))
        # no_wfh_comm_nodes = self.random.choices(population=self.nodes_rest,
        #                                         k=int(len(self.nodes_rest)*0.2))
        # no_wfh_rest_nodes = self.random.choices(population=self.nodes_cafe,
        #                                         k=int(len(self.nodes_cafe)*0.2))
        total_no_wfh = no_wfh_ind_nodes# + no_wfh_comm_nodes + no_wfh_rest_nodes

        ind_agents = max(self.industr_distr_ph) * 3
        # rest_agents = max(self.comm_rest_distr_ph)
        # comm_agents = max(self.comm_distr_ph)
        # CREATING AGENTS
        ''' Needed to account for multifamily housing, so iterating through
        residential nodes and placing agents that way and then storing their
        housemates in the agent object. '''
        res_nodes = copy.deepcopy(self.nodes_resident)
        self.random.shuffle(res_nodes)
        ids = 0
        for node in res_nodes:
            curr_node = list()
            for spot in range(int(self.nodes_capacity[node])):
                a = ConsumerAgent(ids, self)
                self.schedule.add(a)
                if ind_agents != 0:
                    a.work_node = self.random.choice(self.ind_loc_list)
                    a.home_node = node
                    # a.home_node = self.random.choice(self.res_loc_list)
                    self.ind_loc_list.remove(a.work_node)
                    # self.res_loc_list.remove(a.home_node)
                    a.work_type = 'industrial'
                    ind_agents -= 1
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
                    a.can_wfh == False
                self.grid.place_agent(a, a.home_node)
                curr_node.append(a.unique_id)
                ids += 1

            if len(curr_node) > 6: # multifamily housing
                while len(curr_node) > 6:
                    home_size = self.random.choice(range(1,7))
                    curr_housemates = self.random.choices(curr_node, k=home_size)
                    curr_node = [a for a in curr_node if a not in curr_housemates]
                    for mate in curr_housemates:
                        agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                        agent.housemates = copy.deepcopy(curr_housemates) # this includes current agent

                for mate in curr_node:
                    agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                    agent.housemates = copy.deepcopy(curr_node)
            else:
                for agent in curr_node:
                    agent = [a for a in self.schedule.agents if a.unique_id == agent][0]
                    agent.housemates = copy.deepcopy(curr_node)


    def create_demand_houses(self):
        ''' Create houses using pysimdeum for stochastic demand simulation '''
        self.res_houses = list()
        for node in self.nodes_resident:
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
        ages = [1,2,3,4,5,6,7,8,9]
        age_weights = [0.25, 0.18, 0.15, 0.14, 0.12, 0.08, 0.05, 0.01, 0.01]

        '''
        Assign agents either susceptible or infected to COVID based on initial
        infected number.
        '''
        exposed_sample = self.random.sample([i for i, a in enumerate(self.schedule.agents)],
                                               self.covid_exposed)
        for i, agent in enumerate(self.schedule.agents):
            ''' Set age of each agent '''
            agent.age = self.random.choices(population = ages,
                                            weights = age_weights,
                                            k = 1)[0]

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
                    else:
                        agent.agent_params[param] = int(agent_set_params[param]) - 1
                except:
                    pass

    def create_comm_network(self):
        '''
        CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
        Assigning Agents randomly to nodes in SNW
        '''
        self.snw_agents_node = {}
        Nodes_in_snw = list(range(1, Micro_pop + 1))

        # Create dictionairy with dict[agents]= Node
        for agent in self.schedule.agents:
            node_to_agent = self.random.choice(Nodes_in_snw)
            self.snw_agents_node[agent] = node_to_agent
            Nodes_in_snw.remove(node_to_agent)

        # Create dictionairy with dict[Nodes]= agent
        self.snw_node_agents = {y: x for x, y in self.snw_agents_node.iteritems()}
        self.snw_node_agents = dict(zip(self.snw_agents_node.values(), self.snw_agents_node.keys()))

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

        self.stat_tot = np.divide(self.stat_tot, self.num_agents)
        step_status = pd.DataFrame([self.stat_tot], columns = ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I', 'wfh'])
        self.status_tot = pd.concat([self.status_tot, step_status])

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
                    if self.random.random() < self.exposure_rate_large:
                        agent.covid = 'exposed'
                    else:
                        pass
                elif node_type == 'residential':
                    if self.random.random() < self.exposure_rate:
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
            agent.agent_params["COVIDexp"] = 2
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
        ## Communication through TV and Radio
        # Percentage of people listening to radio:
        radio_reach = radio_distr[self.timestepN]/100
        tv_reach = TV_distr[self.timestepN]/100

        ## Communication through radio
        for i, a in enumerate(self.schedule.agents):
            if self.random.random() < radio_reach:
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

        ## Communication through TV
        for i, a in enumerate(self.schedule.agents):
            if self.random.random() < tv_reach:
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

    def move(self):
        """
        Move the correct number of agents to and from commercial nodes.
        """
        curr_comm_num = self.comm_distr_ph[self.timestepN]
        prev_comm_num = self.comm_distr_ph[self.timestepN - 1]
        delta_agents_comm = round(curr_comm_num - prev_comm_num)
        if delta_agents_comm > 0:
            Possible_Agents_to_move = [a for a in self.schedule.agents
                                       if a.pos in self.nodes_resident]
                                       #and a.work_type == 'commercial']

            nodes_comm = list()
            for node in self.nodes_rest:
                avail_spots = self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_comm.append(node)

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
                # print(location)
                # print(Agent_to_move.pos)
                # print(Agent_to_move.home_node)
                # print(Agent_to_move.work_node)
                # print(Agent_to_move)
                # if Agent_to_move in self.agents_moved:
                #     print('Agent was moved to industrial node.')
                # if len(self.grid.G.nodes[Agent_to_move.pos]['agent']) < 7:
                #     print(self.grid.G.nodes[Agent_to_move.pos]['agent'])
                #     print(self.grid.G.nodes[Agent_to_move.home_node]['agent'])
                #
                # if (Agent_to_move.work_type != None and
                #     Agent_to_move in self.grid.G.nodes[Agent_to_move.work_node]['agent']):
                #     print(f"Agent {Agent_to_move} is at work.")
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
        else:
            pass

        """
        Move the correct number of agents to and from cafe nodes.
        """
        curr_rest_num = self.comm_rest_distr_ph[self.timestepN]
        prev_rest_num = self.comm_rest_distr_ph[self.timestepN - 1]
        delta_agents_rest = round(curr_rest_num - prev_rest_num)
        if delta_agents_rest > 0:
            Possible_Agents_to_move = [a for a in self.schedule.agents
                                       if a.pos in self.nodes_resident
                                       and a.work_type == 'restaurant']

            nodes_cafe = list()
            for node in self.nodes_cafe:
                avail_spots = self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_cafe.append(node)

            for i in range(min(delta_agents_rest,len(Possible_Agents_to_move))):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                location = self.random.choice(nodes_cafe)
                # if (Agent_to_move.wfh == 1 and
                #     self.wfh_thres and
                #     Agent_to_move.can_wfh == True):
                #     pass
                # else:
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
        Agents_to_home = int(min(1092, len(Possible_Agents_to_move_home)))

        for i in range(Agents_to_home):
            Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
            self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
            Possible_Agents_to_move_home.remove(Agent_to_move)
            self.infect_agent(Agent_to_move, 'residential')

        # Agents from Residential nodes to Industrial
        Possible_Agents_to_move_to_work = [a for a in self.schedule.agents
                                           if a.pos in self.nodes_resident
                                           and a.work_type == 'industrial']

        Agents_to_work = 1092 # int(1092/2) if self.timestep != 0 else 1092
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
                if Agent_to_move.home_node == Agent_to_move.pos:
                    print(f"Agent {Agent_to_move} not moved")
                Possible_Agents_to_move_to_work.remove(Agent_to_move)
                self.infect_agent(Agent_to_move, 'workplace')

    # def move_wfh(self):
    #     """
    #     Move the correct number of agents to and from commercial nodes.
    #     """
    #     curr_comm_num = self.comm_distr_ph[self.timestepN]
    #     prev_comm_num = self.comm_distr_ph[self.timestepN - 1]
    #     delta_agents_comm = int(curr_comm_num - prev_comm_num)
    #     if delta_agents_comm > 0:
    #         Possible_Agents_to_move = [a for a in self.schedule.agents
    #                                    if a.pos in self.nodes_resident
    #                                    and a.work_type == 'commercial']
    #         # delta_agents_comm = round(delta_agents_comm * (1 - (self.stat_tot[3] * 2)))
    #         for i in range(delta_agents_comm):
    #             Agent_to_move = self.random.choice(Possible_Agents_to_move)
    #             if Agent_to_move.wfh == 0:
    #                 self.grid.move_agent(Agent_to_move, Agent_to_move.work_node)
    #                 Possible_Agents_to_move.remove(Agent_to_move)
    #                 self.infect_agent(Agent_to_move, 'workplace')
    #             else:
    #                 pass
    #
    #     elif delta_agents_comm < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
    #         Possible_Agents_to_move = self.commercial_agents()
    #         for i in range(min(abs(delta_agents_comm), len(Possible_Agents_to_move))):
    #             Agent_to_move = self.random.choice(Possible_Agents_to_move)
    #             self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
    #             Possible_Agents_to_move.remove(Agent_to_move)
    #             self.infect_agent(Agent_to_move, 'residential')
    #     else:
    #         pass
    #
    #     """
    #     Move the correct number of agents to and from rest nodes.
    #     """
    #     curr_rest_num = self.comm_rest_distr_ph[self.timestepN]
    #     prev_rest_num = self.comm_rest_distr_ph[self.timestepN - 1]
    #     delta_agents_rest = int(curr_rest_num - prev_rest_num)
    #     if delta_agents_rest > 0:
    #         Possible_Agents_to_move = [a for a in self.schedule.agents
    #                                    if a.pos in self.nodes_resident
    #                                    and a.work_type == 'restaurant']
    #         # delta_agents_rest = round(delta_agents_rest * (1 - (self.stat_tot[3] * 2)))
    #         for i in range(delta_agents_rest):
    #             Agent_to_move = self.random.choice(Possible_Agents_to_move)
    #             if Agent_to_move.wfh == 0:
    #                 self.grid.move_agent(Agent_to_move, Agent_to_move.work_node)
    #                 Possible_Agents_to_move.remove(Agent_to_move)
    #                 self.infect_agent(Agent_to_move, 'workplace')
    #             else:
    #                 pass
    #
    #     elif delta_agents_rest < 0:
    #         Possible_Agents_to_move = self.rest_agents()
    #         for i in range(min(abs(delta_agents_rest), len(Possible_Agents_to_move))):
    #             Agent_to_move = self.random.choice(Possible_Agents_to_move)
    #             self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
    #             Possible_Agents_to_move.remove(Agent_to_move)
    #             self.infect_agent(Agent_to_move, 'residential')
    #     else:
    #         pass
    #
    # def move_indust_wfh(self):
    #     """
    #     Test function for moving industrial agents during work from home scenarios.
    #     """
    #
    #     # Moving Agents from Industrial nodes back home to residential home nodes
    #     Possible_Agents_to_move_home = self.industry_agents()
    #     Agents_to_home = int(len(Possible_Agents_to_move_home) / 2)
    #
    #     t = self.timestepN
    #     for i in range(Agents_to_home):
    #         Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
    #         self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
    #         Possible_Agents_to_move_home.remove(Agent_to_move)
    #         self.infect_agent(Agent_to_move, 'residential')
    #
    #     # Agents from Residential nodes to Industrial
    #     Possible_Agents_to_work = [a for a in self.schedule.agents
    #                                        if a.pos in self.nodes_resident
    #                                        and a.work_type == 'industrial'
    #                                        and a.wfh == 0]
    #
    #     if self.timestep != 0:
    #         Agents_to_work = (int(1092/2) if len(Possible_Agents_to_work) >= int(1092/2) else len(Possible_Agents_to_work))
    #     else:
    #         Agents_to_work = int(1092)
    #
    #     for i in range(Agents_to_work):
    #         Agent_to_move = self.random.choice(Possible_Agents_to_work)
    #         if Agent_to_move.wfh == 0:
    #             self.grid.move_agent(Agent_to_move, Agent_to_move.work_node)
    #             Possible_Agents_to_work.remove(Agent_to_move)
    #             self.infect_agent(Agent_to_move, 'workplace')
    #         else:
    #             pass


    def set_patterns(self, node):
        house = [house for house in self.res_houses if house.id == node.name][0]
        consumption = house.simulate(num_patterns=10)
        tot_cons = consumption.sum(['enduse', 'user']).mean([ 'patterns'])
        hourly_cons = tot_cons.groupby('time.hour').mean()
        wn.add_pattern('hs_' + house.id,
                       np.array(np.divide(hourly_cons, hourly_cons.mean())))
        node.demand_timeseries_list[0].pattern_name = 'hs_' + house.id


    def demandsfunc(self):
        for node in self.grid.G.nodes:
            try:
                Capacity_node = self.nodes_capacity[node]
                node_1 = wn.get_node(node)
                # determine demand reduction
                demand_reduction_node = 0
                agents_at_node_list = self.grid.G.nodes[node]['agent']
                agents_at_node = len(agents_at_node_list)
                agents_wfh = len([a for a in agents_at_node_list if a.wfh == 1])
                # for agent in self.grid.G.nodes[node]['agent']: # determine demand reduction for every agent at that node
                #     if agent.compliance == 1:
                #         rf = self.random.randint(0.035 * 1000, 0.417 * 1000) / 1000  # Reduction factor for demands
                #         demand_reduction_node += node_1.demand_timeseries_list[0].base_value * rf /  agents_at_node     # Calculating demand reduction per agent per node
                #     else:
                #         continue

                # Save first base demand so later assign it back to Node after simulation
                self.base_demands_previous[node] = node_1.demand_timeseries_list[0].base_value
                # print(self.base_demands_previous[node])
                node_1.demand_timeseries_list[0].base_value = node_1.demand_timeseries_list[0].base_value * agents_at_node/ Capacity_node - demand_reduction_node
                # check how many agents are working from home at current node
                # if more than 50%, changes pattern
                if self.res_pat_select == 'lakewood':
                    perc_wfh = agents_wfh / agents_at_node
                    if perc_wfh > 0.5 and node in self.nodes_resident:
                        node_1.demand_timeseries_list[0].pattern_name = 'wk1'
                elif self.res_pat_select == 'pysimdeum':
                    if node in self.nodes_resident:
                        self.set_patterns(node_1)
                else:
                   pass
            except:
                pass


    def run_hydraulic(self):
        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        wn.options.time.duration += 3600

        # Assigning first base demand again to individual Nodes so WNTR doesnt add all BD up
        for node,base_demand in self.base_demands_previous.items():

            node_1 = wn.get_node(node)
            node_1.demand_timeseries_list[0].base_value = base_demand

        # SAVING CURRENT DEMAND TIMESTEP IN DEMAND MATRIX
        self.demand_matrix[self.timestep: self.timestep + 1] = results.node['demand'][self.timestepN: self.timestepN + 1]
        self.pressure_matrix[self.timestep: self.timestep + 1] = results.node['pressure'][self.timestepN: self.timestepN + 1]
        self.age_matrix[self.timestep: self.timestep + 1] = results.node['quality'][self.timestepN: self.timestepN + 1]


    def inform_status(self):
        info_stat_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].information == 1:
                info_stat_all += 1
            else:
                pass
        print('\tPeople informed: ' + str(info_stat_all))


    def compliance_status(self):
        compl_stat_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].compliance == 1:
                compl_stat_all += 1
            else:
                pass

        print('\tPeople complying: ' + str(compl_stat_all))


    def predict_wfh(self, agent):
        # agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
        # for agent in agents_not_wfh:
        agent.adj_covid_change = 0
        evidence_agent = agent.agent_params
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        query = bn.inference.fit(self.wfh_dag,
                                 variables = ['work_from_home'],
                                 evidence = evidence_agent,
                                 verbose = 0)
        if self.random.random() < query.df['p'][1]:
        # if self.random.random() < self.rp_wfh_probs[agent.agent_params['risk_perception_r']]:
            agent.wfh = 1


    def change_house_adj(self, agent):
        ''' Function to check whether agents in a given agents node have become
        infected with COVID '''
        # node = agent.home_node
        # agents_at_node = copy.deepcopy(self.grid.G.nodes[node]['agent'])
        # if len(agents_at_node) > 6:
        agents_in_house = copy.deepcopy(agent.housemates)
        # print(agent)
        # print(agents_in_house)
        agents_in_house.remove(agent.unique_id)
        for agent in agents_in_house:
            agent = [a for a in self.schedule.agents if a.unique_id == agent][0]
            agent.adj_covid_change == 1
            if agent.agent_params["COVIDeffect_4"] < 6:
                agent.agent_params["COVIDeffect_4"] += 0.1
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
            if agent.infectious_time == 1:
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


    def change_time_model(self):
        self.timestep += 1
        if self.timestep % 24 == 0:
            wn.options.time.duration = 0
            curr_dir = os.getcwd()
            files_in_dir = os.listdir(curr_dir)

            for file in files_in_dir:
                if file.endswith(".rpt") or file.endswith(".bin") or file.endswith(".inp"):
                    os.remove(os.path.join(curr_dir, file))
            self.timestep_day += 1
            agents_not_wfh = [a for a in self.schedule.agents if a.wfh == 0]
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

                if self.model_wfh:
                    if agent.adj_covid_change == 1 and agent.wfh == 0:
                        self.predict_wfh(agent)
            # self.check_social_dist()
            if self.stat_tot[3] > self.wfh_lag and not self.wfh_thres:
                self.wfh_thres = True
        else:
            pass
        self.timestepN = self.timestep - self.timestep_day * 24
        # print(self.timestep)

    def industry_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.nodes_industr]

    def commercial_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.nodes_rest]

    def rest_agents(self):
        return [a for a in self.schedule.agents if a.pos in self.nodes_cafe]

    def infect_agent(self, agent, next_loc):
        if agent.covid == 'infectious':
            self.contact(agent, next_loc)
        else:
            pass

    def agents_wfh(self):
        return [a for a in self.schedule.agents if a.wfh == 1]

    def print_func(self):
        print('Hour step: ' + str(self.timestepN))
        print('Time step: ' + str(self.timestep))
        print('Day step: ' + str(self.timestep_day))
        print('\n')
        print('\tStatus (%): ' + str(self.stat_tot))
        print('\tStatus (#): ' + str(np.multiply(self.stat_tot, self.num_agents)))
        print('\n')
        print('\tAgents at industrial nodes: ' + str(len(self.industry_agents())))
        print('\tAgents at commercial nodes: ' + str(len(self.commercial_agents())))
        print('\tAgents at restaurant nodes: ' + str(len(self.rest_agents())))
        print('\n')
        print('\tAgents working from home: ' + str(len(self.agents_wfh())))
        print('\n')

    def step(self):
        self.schedule.step()
        if self.timestep != 0:
            self.move()
            # self.move_wfh()
        self.check_agent_loc()
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
        # self.communication_utility()
        self.demandsfunc()
        self.run_hydraulic()
        self.change_time_model()
        # self.inform_status()
        # self.compliance_status()
        self.num_status()
        self.print_func()
