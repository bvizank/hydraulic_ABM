#Consumer Model WNTR- MESA
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

inp_file = 'MICROPOLIS_v1_orig_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()
wn.options.time.duration = 0
wn.options.time.hydraulic_timestep = 3600
wn.options.time.pattern_timestep =3600


class ConsumerModel(Model):
    """A Model with some number of Agents"""
    def __init__(self,N, num_nodes = len(G.nodes),nodes_id = G.nodes, nodes_capacity = Max_pop_pnode_resid,
                 nodes_resident = Nodes_resident, nodes_industr = Nodes_industr,
                 nodes_cafe = Nodes_comm_cafe, nodes_rest = Nodes_comm_rest, init_pop = Init_pop_res_indust,
                 all_terminal_nodes = All_terminal_nodes, sum_distr_ph = Sum_distr_ph, Comm_distr_ph = Comm_distr_ph,
                 Comm_rest_distr_ph = Comm_rest_distr_ph, Resident_distr_ph = Resident_distr_ph,
                 Nodes_comm_all = Nodes_comm_all, industr_distr_ph = Industr_distr_ph ):
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
        self.base_demands_previous = {}
        self.snw = nx.watts_strogatz_graph(n=Micro_pop, p=0.2, k=6, seed=None)
        self.snw_agents = {}
        self.nodes_endangered = All_terminal_nodes
        self.results = None
        self.demand_test = []
        self.demand_matrix = pd.DataFrame(0, index=np.arange(0,86400,3600), columns = G.nodes)
        self.cleared_nodes_it_1 =  Cleared_nodes_iteration_1
        self.cleared_nodes_it_2 = Cleared_nodes_iteration_2
        self.cleared_nodes_it_3 = Cleared_nodes_iteration_3
        self.cleared_nodes_it_4 = Cleared_nodes_iteration_4
        self.cleared_nodes_it_5 = Cleared_nodes_iteration_5
        self.cleared_nodes_it_6 = Cleared_nodes_iteration_6
        self.nodes_cleared = []

        # CREATING AGENTS

        for i in range(self.num_agents):
            a = ConsumerAgent(i, self)
            self.schedule.add(a)


        # INITIALIZATION OF AGENTS (in Char_micropolis.py the random assignment is conducted)
        # Iterating over dictionary, and assigning the right amount of agents[f] to the specific node
        f = 0
        for node, pop in self.init_pop.items():
            for i in range (pop):
                self.grid.place_agent(self.schedule.agents[f],node)
                f +=1

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

        #CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
         # Assigning Agents randomly to nodes in SNW
        self.snw_agents_node = {}
        Nodes_in_snw    = list(range(1,Micro_pop+1))

    # Create dictionairy with dict[agents]= Node
        for agent in self.schedule.agents:
            node_to_agent = self.random.choice(Nodes_in_snw)
            self.snw_agents_node[agent] = node_to_agent
            Nodes_in_snw.remove(node_to_agent)

    # Create dictionairy with dict[Nodes]= agent
        #self.snw_node_agents = {y: x for x, y in self.snw_agents_node.iteritems()}
        self.snw_node_agents = dict(zip(self.snw_agents_node.values(), self.snw_agents_node.keys()))

    def cleareance_nodes(self):


            if self.timestep == 2:
                self.nodes_cleared = self.cleared_nodes_it_1
            elif self.timestep == 4:
                self.nodes_cleared.extend(self.cleared_nodes_it_2)
            elif self.timestep == 6:
                self.nodes_cleared.extend(self.cleared_nodes_it_3)
            elif self.timestep == 8:
                self.nodes_cleared.extend(self.cleared_nodes_it_4)
            elif self.timestep == 10:
                self.nodes_cleared.extend(self.cleared_nodes_it_5)
            elif self.timestep == 12:
                self.nodes_cleared.extend(self.cleared_nodes_it_6)
            else:
                pass


        # Delete cleared nodes from list of endangered nodes

          #  for node in self.nodes_cleared:
              #  try:
               #     self.nodes_endangered.remove(node)
               # except:
               #     pass

       # else:
          #  pass

    def communication_utility(self):

    ## Communication through TV and Radio

    # Percentage of people listening to radio:
        radio_reach = radio_distr[self.timestep]/100
        tv_reach = TV_distr[self.timestep]/100

    ## Communication through radio
        for i, a in enumerate(self.schedule.agents):
            if random.random() < radio_reach:
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

    ## Communication through TV
        for i, a in enumerate(self.schedule.agents):
            if random.random() < tv_reach:
                a.information = 1
                a.informed_by = 'utility'
                a.informed_count_u += 1
            else:
                pass

    ## CLEARANCE
    ## Communication clearance through radio

        if self.timestep > 2:

            for i, a in enumerate(self.schedule.agents):
                if random.random() < radio_reach:
                    a.information_clearance = 1


                else:
                    pass

    ## Communication clearance through TV
            for i, a in enumerate(self.schedule.agents):
                if random.random() < tv_reach:
                    a.information_clearance =1

                else:
                    pass
        else:
            pass

    def move(self):
        # First: Moving Agents from residential nodes to commercial nodes and back

        delta_agents_t = int(
            round((self.comm_distr_ph[self.timestep] - self.comm_distr_ph[self.timestep - 1])))
        if delta_agents_t > 0:
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                              if a.pos in self.nodes_resident]
            possible_steps_comm_all = [node
                                       for node in self.grid.G.nodes
                                       if node in self.nodes_comm_all
                                       and self.nodes_capacity[node] - len(self.grid.G.nodes[node]['agent'])>0] #Making sure capacity is not exceeded
            for i in range(delta_agents_t):
                #print(i)
                Agent_to_move = self.random.choice(Possible_Agents_to_move)
                new_position = self.random.choice(
                    possible_steps_comm_all)
                self.grid.move_agent(Agent_to_move, new_position)
                Possible_Agents_to_move.remove(Agent_to_move)
                # Delete node from List if capacity is exceeded
                if self.nodes_capacity[new_position] - len(self.grid.G.nodes[new_position]['agent'])>0:
                    continue
                else:
                    possible_steps_comm_all.remove(new_position)

        elif delta_agents_t < 0: # It means, that agents are moving back to residential nodes from commmercial nodes
            Possible_Agents_to_move = [a for i, a in enumerate(self.schedule.agents)
                              if a.pos in self.nodes_comm_all]
            for i in range(abs(delta_agents_t)):
               # print(i)
                try:
                    Agent_to_move = self.random.choice(Possible_Agents_to_move)
                    self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                    Possible_Agents_to_move.remove(Agent_to_move)
                except:    #For the case that there are no agents to distribute anymore, stop the loop
                    pass
        elif delta_agents_t == 0:
            pass


        # Moving Agents from and to work in Industrial Nodes. Every 8 hours half the Agents in industrial nodes
        # are being replaced with Agents from Residential nodes. At 1, 9 and 17:00
    def move_indust(self):
        print(self.timestep)
        if self.timestep == 1 or self.timestep == 9 or self.timestep == 17:

          #Moving Agents from Industrial nodes back home to residential home nodes
            Possible_Agents_to_move_home = [a for i, a in enumerate(self.schedule.agents)
                                            if a.pos in self.nodes_industr]
            Agents_to_home = int(1092 / 2)
            t = self.timestep
            for i in range(Agents_to_home):
               #  print(i)
                Agent_to_move = self.random.choice(Possible_Agents_to_move_home)
                self.grid.move_agent(Agent_to_move, Agent_to_move.home_node)
                Possible_Agents_to_move_home.remove(Agent_to_move)

          #Agents from Residential nodes to Industrial
            Possible_Agents_to_move_to_work = [a for i, a in enumerate(self.schedule.agents)
                          if a.pos in self.nodes_resident]

            possible_steps_industry = [node
                                   for node in self.grid.G.nodes
                                   if node in self.nodes_industr ]

            Agents_to_work = int(1092/2)

            for i in range(Agents_to_work):
                Agent_to_move = self.random.choice(Possible_Agents_to_move_to_work)
                new_position = self.random.choice(possible_steps_industry)
                self.grid.move_agent(Agent_to_move, new_position)
                Possible_Agents_to_move_to_work.remove(Agent_to_move)

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

                if node in self.nodes_cleared:
                    for agent in self.grid.G.nodes[node]['agent']:  # determine demand reduction for every agent at that node
                        if agent.information_clearance == 1:
                            continue
                        elif agent.information_clearance == 0 and agent.compliance == 1:
                            rf = random.randint(0.035 * 1000, 0.417 * 1000) / 1000  # Reduction factor for demands
                            demand_reduction_node += node_1.demand_timeseries_list[0].base_value * rf / agents_at_node


                elif node not in self.nodes_cleared:
                    for agent in self.grid.G.nodes[node]['agent']: # determine demand reduction for every agent at that node
                        if agent.compliance == 1:
                            rf = random.randint(0.035 * 1000, 0.417 * 1000) / 1000  # Reduction factor for demands
                            demand_reduction_node +=  node_1.demand_timeseries_list[0].base_value * rf /  agents_at_node     # Calculating demand reduction per agent per node

                        else:
                            continue
                else:
                    pass

                # Save first base demand so later assign it back to Node after simulation
                self.base_demands_previous[node] = node_1.demand_timeseries_list[0].base_value
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
        self.demand_matrix[self.timestep: self.timestep +1] = self.results.node['demand'][self.timestep: self.timestep +1]


    def inform_status(self):
        info_stat_all = 0
        info_clear_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].information == 1:
                info_stat_all += 1

            if self.schedule.agents[i].information_clearance == 1:
                info_clear_all += 1

            else:
                pass
        print('People informed :' + str(info_stat_all))
        print('People informed about clearance:' + str(info_clear_all))
        return info_stat_all, info_clear_all


    def compliance_status(self):
        compl_stat_all = 0
        for i,a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].compliance == 1:
                compl_stat_all += 1
            else:
                pass

        print('People complying :' + str(compl_stat_all))
        return compl_stat_all



    def change_time_model(self):
        self.timestep += 1
        return self.timestep
        print(self.timestep)


    def step(self):
        self.schedule.step()
        self.move()
        self.move_indust()
        self.cleareance_nodes()
        self.communication_utility()
        self.demandsfunc()
        self.run_hydraulic()
        self.change_time_model()
        self.inform_status()
        self.compliance_status()




class ConsumerAgent(Agent):

    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.timestep = 0
        self.home_node = None
        self.demand =  0
        self.base_demand = 0
        self.information = 0
        self.information_clearance = 0
        self.informed_by = None
        self.compliance = 0
        self.informed_count_u = 0
        self.informed_count_p_f = 0




    def complying(self):
        if self.information == 1 and self.informed_count_u < 2 and self.informed_count_p_f < 3:
            if self.informed_by == 'utility' and random.random() < 0.5:
                self.compliance = 1
            elif self.informed_by == 'household' and random.random() < 0.3:
                self.compliance = 1
            elif self.informed_by == 'peer_swn' and random.random() < 0.3:
                self.compliance = 1
        else:
            pass



    def communcation(self):

        household_members = [a for i, a in enumerate(self.model.schedule.agents)
                             if a.home_node == self.home_node]
        if self.compliance == 1:
            f =0
            for a in household_members: #probability or timeframe until every agent informs their family?
                if f < 1:
                    a.information = 1
                    f += 1
                    a.informed_count_p_f +=1

                    if a.informed_by == 'utility':
                        continue
                    else:
                        a.informed_by = 'household'

                else:
                    break
            try:
                f = 0
                for key, value in self.model.snw.adj[self.model.snw_agents_node[self]].items(): # Warning people adjacent in SNW
                    if f < 1:
                        self.model.snw_node_agents[key].information = 1
                        f += 1
                        a.informed_count_p_f += 1
                        if self.model.snw_node_agents[key].informed_by == 'utility':
                            continue
                        else:
                            self.model.snw_node_agents[key].informed_by = 'peer_snw'

                    else:
                        break
            except KeyError:
                pass

        else:
            pass

    def communcation_clerance(self):

        household_members = [a for i, a in enumerate(self.model.schedule.agents)
                             if a.home_node == self.home_node]
        if self.compliance == 1 and self.information_clearance == 1:
            f = 0
            for a in household_members:
                if f < 1:
                    a.information_clearance = 1
                    f += 1

                else:
                    break
            try:
                f = 0
                for key, value in self.model.snw.adj[
                    self.model.snw_agents_node[self]].items():  # Warning people adjacent in SNW
                    if f < 1:
                        self.model.snw_node_agents[key].information_clearace = 1
                        f += 1

                    else:
                        break
            except KeyError:
                pass
        else:
            pass

    def change_time(self):
        self.timestep += 1
        return self.timestep
        #print(self.timestep)

    def step(self):
        self.complying()
        self.communcation()
        self.communcation_clerance()
        self.change_time()






