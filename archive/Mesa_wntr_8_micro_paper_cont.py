# Consumer Model WNTR- MESA
import warnings
warnings.simplefilter("ignore", UserWarning)
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from wntr_1 import *
from Char_micropolis import *
import networkx as nx
from mesa.space import NetworkGrid
import random
from agent_model import *
import math
import time
# import bnlearn as bn

inp_file = 'MICROPOLIS_v1_orig_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()
wn.options.time.duration = 0
wn.options.time.hydraulic_timestep = 3600
wn.options.time.pattern_timestep = 3600


class ConsumerModel(Model):
    """A Model with some number of Agents"""
    def __init__(self,
                 N,
                 num_nodes = len(G.nodes),
                 nodes_id = G.nodes,
                 nodes_capacity = Max_pop_pnode_resid,
                 nodes_resident = Nodes_resident,
                 nodes_industr = Nodes_industr,
                 nodes_cafe = Nodes_comm_cafe,
                 nodes_rest = Nodes_comm_rest,
                 init_pop = Init_pop_res_indust,
                 all_terminal_nodes = All_terminal_nodes,
                 sum_distr_ph = Sum_distr_ph,
                 Comm_distr_ph = Comm_distr_ph,
                 Comm_rest_distr_ph = Comm_rest_distr_ph,
                 Resident_distr_ph = Resident_distr_ph,
                 Nodes_comm_all = Nodes_comm_all,
                 industr_distr_ph = Industr_distr_ph,
                 seed = None,
                 start_inf = 5,
                 daily_contacts = 10):

        init_start = time.perf_counter()
        self.num_agents = N
        self.G = G
        self.num_nodes = num_nodes
        self.nodes_resident = nodes_resident
        self.nodes_industr = nodes_industr
        self.nodes_cafe = nodes_cafe # There is no node assigned to "dairy queen" so it was neglected
        self.nodes_rest = nodes_rest
        self.nodes_id = nodes_id
        self.init_pop = init_pop
        self.terminal_nodes = All_terminal_nodes
        self.nodes_capacity = nodes_capacity
        self.resid_distr_ph = Resident_distr_ph
        self.comm_rest_distr_ph = Comm_rest_distr_ph
        self.comm_distr_ph = Comm_distr_ph
        self.industr_distr_ph = industr_distr_ph
        self.sum_distr_ph = Sum_distr_ph
        self.nodes_comm_all = Nodes_comm_all
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
        self.results = None
        self.demand_test = []
        self.demand_matrix = pd.DataFrame(0, index = np.arange(0, 86400, 3600), columns = G.nodes)
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
        # self.wfh_dag = bn.import_DAG('model_summary_work_from_home.bif')

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

        # CREATING AGENTS
        for i in range(self.num_agents):
            a = ConsumerAgent(i, self)
            self.schedule.add(a)

        # INITIALIZATION OF AGENTS (in Char_micropolis.py the random assignment is conducted)
        # Iterating over dictionary, and assigning the right amount of agents[f] to the specific node
        f = 0
        for node, pop in self.init_pop.items():
            for i in range (pop):
                self.grid.place_agent(self.schedule.agents[f], node)
                f += 1

        # ASSIGNING HOME NODES TO INITIALIZED AGENTS, if at home, then just their current node,
        # If they are not at a residential node, then assigning homenode according to node capacities
        for i, agent in enumerate(self.schedule.agents):
            if self.schedule.agents[i].pos in self.nodes_resident:
                self.schedule.agents[i].home_node = self.schedule.agents[i].pos

            else:
                possible_home_nodes = [node for node in self.grid.G.nodes
                                           if node in self.nodes_resident
                                            and self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])>0]
                home_node = self.random.choice(possible_home_nodes)
                self.schedule.agents[i].home_node = home_node

        # Assign agents an age 1 = 0-19, 2 = 20-29, 3 = 30-39, 4 = 40-49,
        # 5 = 50-59, 6 = 60-69, 7 = 70-79, 8 = 80-89, 9 = 90+
        ages = [1,2,3,4,5,6,7,8,9]
        age_weights = [0.25, 0.18, 0.15, 0.14, 0.12, 0.08, 0.05, 0.01, 0.01]
        for i, agent in enumerate(self.schedule.agents):
            agent.age = self.random.choices(population = ages,
                                            weights = age_weights,
                                            k = 1)[0]

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

        # Assign agents either susceptible or infected to COVID
        exposed_sample = self.random.sample([i for i, a in enumerate(self.schedule.agents)],
                                               self.covid_exposed)
        for i, agent in enumerate(self.schedule.agents):
            if i in exposed_sample:
                agent.covid = 'infectious'
            else:
                agent.covid = 'susceptible'

        # Count the number of each compartment in SEIR model
        self.status_tot = [0,0,0,0,0,0,0,0,0,0,0,self.cumm_infectious]
        for i, agent in enumerate(self.schedule.agents):
            if agent.covid == 'susceptible':
                self.status_tot[1] += 1
            elif agent.covid == 'exposed':
                self.status_tot[2] += 1
            elif agent.covid == 'infectious':
                self.status_tot[3] += 1
            elif agent.covid == 'recovered':
                self.status_tot[4] += 1
            else:
                pass
        self.status_tot = np.divide(self.status_tot, self.num_agents)
        print('Percent infectious: ' + str(self.status_tot[3]))
        print('\n')

        self.status_tot = pd.DataFrame([self.status_tot], columns = ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I'])
        # Create dictionary of work from home probabilities using bnlearn.
        # self.wfh_probs = {}
        # for i in range(7):
        #     query = bn.inference.fit(self.wfh_dag,
        #                              variables = ['work_from_home'],
        #                              evidence = {'COVIDeffect_4':(i)},
        #                              verbose = 0)
        #     self.wfh_probs[i+1] = query.df['p'][1]
        #
        # print(self.wfh_probs)

        # CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
        # Assigning Agents randomly to nodes in SNW
        # self.snw_agents_node = {}
        # Nodes_in_snw = list(range(1, Micro_pop + 1))
        #
        # # Create dictionairy with dict[agents]= Node
        # for agent in self.schedule.agents:
        #     node_to_agent = self.random.choice(Nodes_in_snw)
        #     self.snw_agents_node[agent] = node_to_agent
        #     Nodes_in_snw.remove(node_to_agent)

        # Create dictionairy with dict[Nodes]= agent
        #self.snw_node_agents = {y: x for x, y in self.snw_agents_node.iteritems()}
        # self.snw_node_agents = dict(zip(self.snw_agents_node.values(), self.snw_agents_node.keys()))

        init_stop = time.perf_counter()

        print('Time to initialize: ', init_stop - init_start)

    def num_status(self):
        """
        Function to calculate the number of agents in each compartment (Susceptible,
        exposed, infectious, recovered, and dead), the number of either symptomatic
        and asymptomatic, and the number of agents that are in each severity
        class (mild, severe, and critical). This information is printed every
        hour step.
        """
        self.stat_tot = [self.timestep, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.cumm_infectious]
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
        step_status = pd.DataFrame([self.stat_tot], columns = ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild', 'Sev', 'Crit', 'sum_I'])
        self.status_tot = pd.concat([self.status_tot, step_status])

    def contact(self, agent_to_move, node_type):
        """
        Function to test whether a specific agent has exposed the other agents
        in their node to covid. This is currently called everytime an agent moves
        to a new location and NOT every hour.
        """
        if len(self.grid.G.nodes[agent_to_move.pos]['agent']) > 6:
            agents_at_node = self.grid.G.nodes[agent_to_move.pos]['agent']
            agents_to_infect = self.random.choices(population = agents_at_node,
                                                   k = self.daily_contacts)
        else:
            agents_to_infect = self.grid.G.nodes[agent_to_move.pos]['agent']
        for agent in agents_to_infect:
            # if agent.covid_affect <= 7 and node_type == 'residential':
            #     agent.covid_affect += 0.1
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
            if self.random.random() < self.wfh_probs[math.floor(agent.covid_affect)]:
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
        # First: Moving Agents from residential nodes to commercial nodes and back
        delta_agents_t = int(
            round((self.comm_distr_ph[self.timestepN] - self.comm_distr_ph[self.timestepN - 1])))
        if delta_agents_t > 0:
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                                       if a.pos in self.nodes_resident]
            possible_steps_comm_all = [node
                                       for node in self.grid.G.nodes
                                       if node in self.nodes_comm_all
                                       and self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])>0] #Making sure capacity is not exceeded
            for i in range(delta_agents_t):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                if Agent_to_move.wfh == 0:
                    new_position = self.random.choice(possible_steps_comm_all)
                    self.grid.move_agent(Agent_to_move, new_position)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'workplace')
                    else:
                        pass
                    # Delete node from List if capacity is exceeded
                    if self.nodes_capacity[new_position] - len(self.grid.G.nodes[new_position]['agent'])>0:
                        continue
                    else:
                        possible_steps_comm_all.remove(new_position)
                else:
                    pass

        elif delta_agents_t < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
            # Things to add: Let Agents return to Node where they came from. Introduce variable " self.homenode and self.worknode
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                              if a.pos in self.nodes_comm_all]
            for i in range(abs(delta_agents_t)):
                try:
                    Agent_to_move = self.random.choice(Possible_Agents_to_move)
                    self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'residential')
                    else:
                        pass
                except:    #For the case that there are no agents to distribute anymore, stop the loop
                    pass
        elif delta_agents_t == 0:
            pass


    # Moving Agents from and to work in Industrial Nodes. Every 8 hours half the Agents in industrial nodes
    # are being replaced with Agents from Residential nodes. At 1, 9 and 17:00
    def move_indust(self):
        # BV: changed times to 6, 14, and 22 because I think this is more representative
        # of a three shift schedule. Unsure if this changes anything with water
        # patterns, but I suspect it might.
        if self.timestepN == 6 or self.timestepN == 14 or self.timestepN == 22:

            # Moving Agents from Industrial nodes back home to residential home nodes
            Possible_Agents_to_move_home = [a for i, a in enumerate(self.schedule.agents)
                                            if a.pos in self.nodes_industr]
            # print('Agents at industrial nodes: ' + str(len(Possible_Agents_to_move_home)))
            # if len(Possible_Agents_to_move_home) > 1092/2:
            Agents_to_home = int(1092 / 2)
            # else:
            #     Agents_to_home = len(Possible_Agents_to_move_home)

            t = self.timestepN
            for i in range(Agents_to_home):
                Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move_home.remove(Agent_to_move)
                if Agent_to_move.covid == 'infectious':
                    self.contact(Agent_to_move, 'residential')
                else:
                    pass
            # Agents from Residential nodes to Industrial
            Possible_Agents_to_move_to_work = [a for i, a in enumerate(self.schedule.agents)
                          if a.pos in self.nodes_resident]

            possible_steps_industry = [node
                                   for node in self.grid.G.nodes
                                   if node in self.nodes_industr]

            Agents_to_work = int(1092/2)

            for i in range(Agents_to_work):
                Agent_to_move = self.random.choice(Possible_Agents_to_move_to_work)
                if Agent_to_move.wfh == 0:
                    new_position = self.random.choice(possible_steps_industry)  # Add boundary condition-> check whether capacity of the node is not exceeded
                    self.grid.move_agent(Agent_to_move, new_position)
                    Possible_Agents_to_move_to_work.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'workplace')
                    else:
                        pass
                else:
                    pass

        else:
            pass

    def move_wfh(self):
        # First: Moving Agents from residential nodes to commercial nodes and back
        delta_agents_t = int(
            round((self.comm_distr_ph[self.timestepN] - self.comm_distr_ph[self.timestepN - 1])))
        if delta_agents_t > 0:
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                                       if a.pos in self.nodes_resident]
            possible_steps_comm_all = [node
                                       for node in self.grid.G.nodes
                                       if node in self.nodes_comm_all
                                       and self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])>0] #Making sure capacity is not exceeded
            delta_agents_t = round(delta_agents_t * (1 - self.stat_tot[3]))
            for i in range(delta_agents_t):
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                if Agent_to_move.wfh == 0:
                    new_position = self.random.choice(possible_steps_comm_all)
                    self.grid.move_agent(Agent_to_move, new_position)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'workplace')
                    else:
                        pass
                    # Delete node from List if capacity is exceeded
                    if self.nodes_capacity[new_position] - len(self.grid.G.nodes[new_position]['agent'])>0:
                        continue
                    else:
                        possible_steps_comm_all.remove(new_position)
                else:
                    pass

        elif delta_agents_t < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
            # Things to add: Let Agents return to Node where they came from. Introduce variable " self.homenode and self.worknode
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                              if a.pos in self.nodes_comm_all]
            for i in range(abs(delta_agents_t)):
                try:
                    Agent_to_move = self.random.choice(Possible_Agents_to_move)
                    self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                    Possible_Agents_to_move.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'residential')
                    else:
                        pass
                except:    #For the case that there are no agents to distribute anymore, stop the loop
                    pass
        elif delta_agents_t == 0:
            pass

    def move_indust_wfh(self):
        """
        Test function for moving industrial agents during work from home scenarios.
        """
        # BV: changed times to 6, 14, and 22 because I think this is more representative
        # of a three shift schedule. Unsure if this changes anything with water
        # patterns, but I suspect it might.
        if self.timestepN == 6 or self.timestepN == 14 or self.timestepN == 22:

            # Moving Agents from Industrial nodes back home to residential home nodes
            Possible_Agents_to_move_home = [a for i, a in enumerate(self.schedule.agents)
                                            if a.pos in self.nodes_industr]
            # print('Agents at industrial nodes: ' + str(len(Possible_Agents_to_move_home)))
            # if len(Possible_Agents_to_move_home) > 1092/2:
            Agents_to_home = int(len(Possible_Agents_to_move_home) / 2)
            # else:
            #     Agents_to_home = len(Possible_Agents_to_move_home)

            t = self.timestepN
            for i in range(Agents_to_home):
                Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move_home.remove(Agent_to_move)
                if Agent_to_move.covid == 'infectious':
                    self.contact(Agent_to_move, 'residential')
                else:
                    pass
            # Agents from Residential nodes to Industrial
            Possible_Agents_to_move_to_work = [a for i, a in enumerate(self.schedule.agents)
                                               if a.pos in self.nodes_resident]

            possible_steps_industry = [node
                                       for node in self.grid.G.nodes
                                       if node in self.nodes_industr]

            Agents_to_work = int(1092/2 * (1 - self.stat_tot[3]))

            for i in range(Agents_to_work):
                Agent_to_move = self.random.choice(Possible_Agents_to_move_to_work)
                if Agent_to_move.wfh == 0:
                    new_position = self.random.choice(possible_steps_industry)  # Add boundary condition-> check whether capacity of the node is not exceeded
                    self.grid.move_agent(Agent_to_move, new_position)
                    Possible_Agents_to_move_to_work.remove(Agent_to_move)
                    if Agent_to_move.covid == 'infectious':
                        self.contact(Agent_to_move, 'workplace')
                    else:
                        pass
                else:
                    pass

        else:
            pass

    def demandsfunc(self):
        for node in self.grid.G.nodes:
            try:
                Capacity_node = self.nodes_capacity[node]
                node_1 = wn.get_node(node)
                # determine demand reduction
                demand_reduction_node = 0
                agents_at_node = len(self.grid.G.nodes[node]['agent'])
                for agent in self.grid.G.nodes[node]['agent']: # determine demand reduction for every agent at that node
                    if agent.compliance == 1:
                        rf = self.random.randint(0.035 * 1000, 0.417 * 1000) / 1000  # Reduction factor for demands
                        demand_reduction_node += node_1.demand_timeseries_list[0].base_value * rf /  agents_at_node     # Calculating demand reduction per agent per node
                    else:
                        continue

                # Save first base demand so later assign it back to Node after simulation
                self.base_demands_previous[node] = node_1.demand_timeseries_list[0].base_value
                # print(self.base_demands_previous[node])
                node_1.demand_timeseries_list[0].base_value = node_1.demand_timeseries_list[0].base_value * agents_at_node/ Capacity_node - demand_reduction_node

            except:
                pass


    def run_hydraulic(self):
        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(wn)
        self.results = sim.run_sim()
        wn.options.time.duration += 3600

        # Assigning first base demand again to individual Nodes so WNTR doesnt add all BD up
        for node,base_demand in self.base_demands_previous.items():

            node_1 = wn.get_node(node)
            node_1.demand_timeseries_list[0].base_value = base_demand

        # SAVING CURRENT DEMAND TIMESTEP IN DEMAND MATRIX
        self.demand_matrix[self.timestepN: self.timestepN + 1] = self.results.node['demand'][self.timestepN: self.timestepN + 1]


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

    def change_time_model(self):
        self.timestep += 1
        if self.timestep % 24 == 0:
            self.timestep_day += 1
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
            # self.check_social_dist()
        else:
            pass
        self.timestepN = self.timestep - self.timestep_day * 24
        # print(self.timestep)

    def count_industry(self):
        return len([a for a in self.schedule.agents if a.pos in self.nodes_industr])

    def count_commercial(self):
        return len([a for a in self.schedule.agents if a.pos in self.nodes_rest])

    def count_rest(self):
        return len([a for a in self.schedule.agents if a.pos in self.nodes_cafe])

    def print_func(self):
        print('Time step: ' + str(self.timestep))
        print('Day step: ' + str(self.timestep_day))
        print('\n')
        print('\tStatus (%): ' + str(self.stat_tot))
        print('\tStatus (#): ' + str(np.multiply(self.stat_tot, self.num_agents)))
        print('\n')
        print('\tAgents at industrial nodes: ' + str(self.count_industry()))
        print('\tAgents at commercial nodes: ' + str(self.count_commercial()))
        print('\tAgents at restaurant nodes: ' + str(self.count_rest()))
        print('\n')

    def step(self):
        self.schedule.step()
        self.move()
        # self.move_indust()
        # self.move_wfh()
        self.move_indust_wfh()
        self.check_status()
        # self.communication_utility()
        # self.demandsfunc()
        # self.run_hydraulic()
        self.change_time_model()
        # self.inform_status()
        # self.compliance_status()
        self.num_status()
        self.print_func()
