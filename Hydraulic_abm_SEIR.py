# Consumer Model WNTR- MESA
from parameters import Parameters
import warnings
import data as dt
import math
import time
import numpy as np
import pandas as pd
from copy import deepcopy as dcp
import wntr
from wntr.epanet.util import EN

import matplotlib.pyplot as plt

warnings.simplefilter("ignore", UserWarning)


class ConsumerModel(Parameters):
    """
    ABM model class. Contains all methods to run the ABM simulation

    to do:
        - Figure out a better way to handle the keyword arguments
        -

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
    """

    def __init__(
        self,
        N,  # number of people simulated
        city,
        days=90,
        id=0,  # id of simulation
        seed=None,
        **kwargs,
    ):
        super().__init__(N, city, days, id, seed, **kwargs)
        init_start = time.perf_counter()

        """ Setup and mapping of variables from various sources. For more information
        see utils.py and parameters.py """
        if city == "micropolis" or city == "mesopolis":
            self.setup_virtual(city)
        else:
            self.setup_real(city)

        # if we are running the simulation hourly, we need to have
        # demand patterns that are as long as the simulation, which
        # is what update_patterns does.
        if self.hyd_sim == "hourly":
            self.update_patterns()

        init_stop = time.perf_counter()

        if self.verbose == 1:
            print("Time to initialize: ", init_stop - init_start)

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
            len(self.agents_wfh()),
        ]

        for i, agent in enumerate(self.schedule.agents):
            if agent.covid == "susceptible":
                self.stat_tot[1] += 1
            elif agent.covid == "exposed":
                self.stat_tot[2] += 1
            elif agent.covid == "infectious":
                self.stat_tot[3] += 1
            elif agent.covid == "recovered":
                self.stat_tot[4] += 1
            elif agent.covid == "dead":
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
        agents_at_node = self.buildings[agent_to_move.building].agents_at_building()
        if node_type == "residential":
            agents_to_expose = [
                a
                for a in agent_to_move.household.agents_at_house()
                if a in agents_at_node
            ]
        elif node_type == "workplace":
            if len(agents_at_node) > self.daily_contacts:
                agents_to_expose = self.random.sample(
                    population=agents_at_node, k=self.daily_contacts
                )
            else:
                agents_to_expose = agents_at_node

        for agent_id in agents_to_expose:
            agent = self.agents_list[agent_id]
            if agent.covid == "susceptible":
                if node_type == "workplace":
                    if agent_to_move.ppe == 0:
                        if self.random.random() < self.exposure_rate_large:
                            agent.covid = "exposed"
                        else:
                            pass
                    else:
                        if (
                            self.random.random()
                            < self.exposure_rate_large * self.ppe_reduction
                        ):
                            agent.covid = "exposed"
                        else:
                            pass
                elif node_type == "residential":
                    if agent_to_move.ppe == 0:
                        if self.random.random() < self.exposure_rate:
                            agent.covid = "exposed"
                        else:
                            pass
                    else:
                        if (
                            self.random.random()
                            < self.exposure_rate * self.ppe_reduction
                        ):
                            agent.covid = "exposed"
                        else:
                            pass
                else:
                    print("Warning: node type: " + str(node_type) + " does not exist.")
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
            if agent.covid == "exposed":
                agent.exposed_time += 1
            elif agent.covid == "infectious":
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
            if (
                self.random.random()
                < self.wfh_probs[math.floor(agent.agent_params["COVIDeffect_4"])]
            ):
                agent.wfh = 1

    def communication_utility(self):
        # Communication through TV and Radio
        # Percentage of people listening to radio:
        radio_reach = dt.radio[self.timestepN] / 100
        tv_reach = dt.tv[self.timestepN] / 100

        # Communication through radio
        for i, agent in enumerate(self.schedule.agents):
            if self.random.random() < radio_reach:
                agent.agent_params["MediaExp_3"] = 0
                agent.information = 1
                agent.informed_by = "utility"
                agent.informed_count_u += 1
            else:
                pass

        # Communication through TV
        for i, agent in enumerate(self.schedule.agents):
            if self.random.random() < tv_reach:
                agent.agent_params["MediaExp_3"] = 0
                agent.information = 1
                agent.informed_by = "utility"
                agent.informed_count_u += 1
            else:
                pass

    def move_com(self):
        """
        Move the correct number of agents to and from commercial nodes.
        """
        curr_comm_num = self.com_dist[self.timestepN] * self.num_com_agents
        prev_comm_num = self.com_dist[self.timestepN - 1] * self.num_com_agents
        # print(curr_comm_num)
        # print(prev_comm_num)
        delta_agents_comm = int(curr_comm_num - prev_comm_num)
        # print(delta_agents_comm)
        if delta_agents_comm > 0:
            res_agent_list = self.residential_agents()

            # find all the commercial nodes with open spots and make a list
            nodes_comm = list()
            for node in np.append(self.com_nodes, self.gro_nodes):
                avail_spots = self.buildings[node].capacity - (
                    self.buildings[node].count_agents()
                )
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_comm.append(node)
            # print("Comm nodes: " + str(len(nodes_comm)))

            # move each agent that is slated to be moved
            agents_moved = list()
            for i in range(min(delta_agents_comm, len(res_agent_list))):
                agent_id = self.random.choice(res_agent_list)
                agent_to_move = self.agents_list[agent_id]
                location = self.random.choice(nodes_comm)
                if location in self.gro_nodes:
                    if agent_to_move.less_groceries == 1:
                        continue
                    agent_to_move.move(location)
                    res_agent_list = np.delete(
                        res_agent_list, np.where(res_agent_list == agent_id)[0][0]
                    )
                    nodes_comm.remove(location)
                    agents_moved.append(agent_to_move.unique_id)
                    self.infect_agent(agent_to_move, "workplace")
                else:
                    """The agents in this arm are considered workers"""
                    if agent_to_move.wfh == 1:
                        continue
                    agent_to_move.move(location)
                    res_agent_list = np.delete(
                        res_agent_list, np.where(res_agent_list == agent_id)[0][0]
                    )
                    nodes_comm.remove(location)
                    agents_moved.append(agent_to_move.unique_id)
                    self.infect_agent(agent_to_move, "workplace")

            self.res_agents[agents_moved] = 0
            self.com_agents[agents_moved] = 1

        elif delta_agents_comm < 0:
            # It means, that agents are moving back to residential nodes from commmercial nodes
            com_agent_list = self.commercial_agents()
            agents_moved = list()
            for i in range(min(abs(delta_agents_comm), len(com_agent_list))):
                agent_id = self.random.choice(com_agent_list)
                agent_to_move = self.agents_list[agent_id]
                agent_to_move.move(agent_to_move.home_node)
                com_agent_list = np.delete(
                    com_agent_list, np.where(com_agent_list == agent_id)[0][0]
                )
                agents_moved.append(agent_to_move.unique_id)
                self.infect_agent(agent_to_move, "residential")

            self.res_agents[agents_moved] = 1
            self.com_agents[agents_moved] = 0

        if self.com_dist[self.timestepN] == 0:
            com_agent_list = self.commercial_agents()
            # print(f"Moving {len(com_agent_list)} agents home.")
            agents_moved = list()
            for i in range(min(abs(delta_agents_comm), len(com_agent_list))):
                agent_id = self.random.choice(com_agent_list)
                agent_to_move = self.agents_list[agent_id]
                agent_to_move.move(agent_to_move.home_node)
                com_agent_list = np.delete(
                    com_agent_list, np.where(com_agent_list == agent_id)[0][0]
                )
                agents_moved.append(agent_to_move.unique_id)
                self.infect_agent(agent_to_move, "residential")

            self.res_agents[agents_moved] = 1
            self.com_agents[agents_moved] = 0

    def move_caf(self):
        """
        Move the correct number of agents to and from cafe nodes.
        """
        # curr_rest_num = self.cafe_dist[self.timestepN] * self.num_caf_agents
        # prev_rest_num = self.cafe_dist[self.timestepN - 1] * self.num_caf_agents
        delta_agents_rest = round(
            (self.cafe_dist[self.timestepN] - self.cafe_dist[self.timestepN - 1])
            * self.num_caf_agents
        )
        # print(f"Cafe agents to move: {delta_agents_rest}")
        # print(f"Current number of agents at cafe nodes: {len(self.cafe_agents())}")

        if delta_agents_rest > 0:
            res_agent_list = self.residential_agents()

            # find all of the cafe vacancies and make a list
            nodes_cafe = list()
            for node in self.caf_nodes:
                # print(f"Capacity: {self.buildings[node].capacity}")
                # print(f"Current agents: {self.buildings[node].agent_ids}")
                avail_spots = self.buildings[node].capacity - (
                    self.buildings[node].count_agents()
                )
                # print(avail_spots)
                if avail_spots > 0:
                    for i in range(int(avail_spots)):
                        nodes_cafe.append(node)
            # print("Num cafe nodes: " + str(len(nodes_cafe)))
            # print("Cafe nodes: ")
            # print(nodes_cafe)

            agents_moved = list()
            for i in range(
                min(delta_agents_rest, len(res_agent_list), len(nodes_cafe))
            ):
                agent_id = self.random.choice(res_agent_list)
                agent_to_move = self.agents_list[agent_id]
                location = self.random.choice(nodes_cafe)
                if agent_to_move.no_dine == 1:
                    continue
                agent_to_move.move(location)
                res_agent_list = np.delete(
                    res_agent_list, np.where(res_agent_list == agent_id)[0][0]
                )
                nodes_cafe.remove(location)
                agents_moved.append(agent_to_move.unique_id)
                self.infect_agent(agent_to_move, "workplace")

            self.res_agents[agents_moved] = 0
            self.caf_agents[agents_moved] = 1

        elif delta_agents_rest < 0:
            caf_agent_list = self.cafe_agents()
            agents_moved = list()
            for _ in range(min(abs(delta_agents_rest), len(caf_agent_list))):
                agent_id = self.random.choice(caf_agent_list)
                agent_to_move = self.agents_list[agent_id]
                agent_to_move.move(agent_to_move.home_node)
                caf_agent_list = np.delete(
                    caf_agent_list, np.where(caf_agent_list == agent_id)[0][0]
                )
                agents_moved.append(agent_to_move.unique_id)
                self.infect_agent(agent_to_move, "residential")

            self.res_agents[agents_moved] = 1
            self.caf_agents[agents_moved] = 0

        if self.cafe_dist[self.timestepN] == 0:
            # move all agents out of cafe nodes
            caf_agent_list = self.cafe_agents()
            # print(f"Moving {len(caf_agent_list)} agents back home.")
            agents_moved = list()
            for _ in range(len(caf_agent_list)):
                agent_id = self.random.choice(caf_agent_list)
                agent_to_move = self.agents_list[agent_id]
                agent_to_move.move(agent_to_move.home_node)
                caf_agent_list = np.delete(
                    caf_agent_list, np.where(caf_agent_list == agent_id)[0][0]
                )
                agents_moved.append(agent_to_move.unique_id)
                self.infect_agent(agent_to_move, "residential")

            self.res_agents[agents_moved] = 1
            self.caf_agents[agents_moved] = 0

    # Moving Agents from and to work in Industrial Nodes. Every 8 hours half the Agents in industrial nodes
    # are being replaced with Agents from Residential nodes. At 1, 9 and 17:00
    def move_ind(self):
        # Moving Agents from Industrial nodes back home to residential home nodes
        ind_agent_list = dcp(self.industry_agents())
        # print(ind_agent_list)
        agents_to_home_n = int(min(self.ind_agent_n / 2, len(ind_agent_list)))
        agents_to_work = (
            int(self.ind_agent_n / 2) if self.timestep != 0 else self.ind_agent_n
        )

        agents_to_home = self.rng.choice(ind_agent_list, agents_to_home_n, False)
        # print(agents_to_home)
        # print(len(ind_agent_list))

        # agents_moved = list()
        for i in agents_to_home:
            # agent_id = self.random.choice(ind_agent_list)
            # agent_to_move = self.agents_list[agent_id]
            # # print(f"{agent_id} is moving from {agent_to_move.building}")
            # agent_to_move.move(agent_to_move.home_node)
            # # print(f"{agent_id} moved to {agent_to_move.building} with type {self.buildings[agent_to_move.building].type}")
            # ind_agent_list = np.delete(
            #     ind_agent_list, np.where(ind_agent_list == agent_id)[0][0]
            # )

            # get the agent object
            agent_to_move = self.agents_list[i]
            # move the agent to their home node
            agent_to_move.move(agent_to_move.home_node)
            # add the agent_id to the agents_moved list
            # agents_moved.append(agent_to_move.unique_id)
            # expose the agent based on the new node it moved to
            self.infect_agent(agent_to_move, "residential")

        self.res_agents[agents_to_home] = 1
        self.ind_agents[agents_to_home] = 0

        # Agents from Residential nodes to Industrial
        # list of agents that are at their home node and have
        # a work node. The list are indices of agents

        """
        How agent's should move to industrial nodes

        1. Agents that are at resdiential nodes and have work nodes
           can move. That makes our list.
        2. Because we give work nodes to twice as many agents as is
           necessary, each industrial nodes has 2x it's capacity in
           agents that have it as a work node. This means we need to
           see whether the node that an agent is moving to is full,
           and if it is, get another agent to move.
        """
        res_agent_ind = dcp(self.res_agents * self.ind_work_nodes)
        # print(res_agent_ind)
        # print(res_agent_ind)
        # agents_to_work = self.rng.choice(res_agent_ind, agents_to_work_n, False)
        # res_agent_list = np.where(res_agent_ind > 0)[0]

        # for a in self.schedule.agents:
        #     print(a.building)
        # print(res_agent_list)

        """ Make a list of all of the empty spots at industrial nodes """
        nodes_ind = list()
        for node in self.ind_nodes:
            avail_spots = self.buildings[node].capacity - (
                self.buildings[node].count_agents()
            )
            if avail_spots > 0:
                # get all agents with a work node that is the current node
                node_agents = np.where(res_agent_ind == node)[0]

                # if the number of agents that can move is less than the avail spots
                # then truncate the avail spots to how many agents can move
                if len(node_agents) < avail_spots:
                    avail_spots = len(node_agents)
                elif len(node_agents) == 0:
                    continue

                for i in range(int(avail_spots)):
                    nodes_ind.append(node)

        agents_moved = list()
        for i in range(agents_to_work):
            # pick a work node from the pool of work nodes that have capacity
            work_node = self.random.choice(nodes_ind)
            # remove the current agent from the list of agent ind locations
            nodes_ind.remove(work_node)
            # pick an agent that is at that work node
            agent_ids = np.where(res_agent_ind == work_node)[0]
            # print(len(agent_ids))
            agent_id = self.rng.choice(agent_ids)
            res_agent_ind[agent_id] = 0
            agent_to_move = self.agents_list[agent_id]
            if agent_to_move.wfh == 1 and self.wfh_thres and agent_to_move.can_wfh:
                continue
            agents_moved.append(agent_to_move.unique_id)
            agent_to_move.move(agent_to_move.work_node)
            # res_agent_list = np.delete(
            #     res_agent_list, np.where(res_agent_list == agent_id)[0][0]
            # )
            self.infect_agent(agent_to_move, "workplace")

        self.res_agents[agents_moved] = 0
        self.ind_agents[agents_moved] = 1

    def collect_demands(self):
        """
        Collect the location of each agent.
        """
        for building_id, building in self.buildings.items():
            if building.households is None:
                building.agent_history.append(dcp(building.count_agents()))
            else:
                for house in building.households:
                    house.agent_history.append(dcp(house.count_agents()))
            # agents_at_node = len(building.agent_ids)
            # self.agent_matrix[building_id, self.timestep] = agents_at_node

    def calc_res_demands(self, building):
        out_pat = np.zeros(24)
        for house in building.households:
            """First get the demand pattern. If a majority of household
            is working from home, get the wfh pattern"""
            agents_at_node = house.agent_obs
            agents_wfh = len([a for a in agents_at_node if a.wfh == 1])
            if self.res_pat_select == "lakewood" and len(agents_at_node) != 0:
                perc_wfh = agents_wfh / len(agents_at_node)
                if perc_wfh > 0.5:
                    base_pat = self.wn.get_pattern("wk1").multipliers
                else:
                    base_pat = house.demand_pattern

            house_agents = np.array(
                dcp(house.agent_history[self.timestep - 23 : self.timestep + 1])
            )
            new_mult = house_agents / house.capacity

            """ demand for the next 24 hours, not including reduction or
            agent multiplier """
            curr_pat = base_pat * new_mult
            # print(f"curr_pat for res buildings: \n {curr_pat}")

            """ Calculate the demand reduction given the total demand """
            # total demand in L/day
            daily_demand = curr_pat.sum() * house.base_demand * 60 * 60

            reduction_val = 0

            # average agent multiplier
            avg_agent_multiplier = sum(new_mult) / len(new_mult)

            # add the demand from this household to the tap_demand
            if self.twa_process == "absolute":
                # reduction value needs to be offset by the agent reduction
                # which is the average agent multiplier
                house.tap_demand += (
                    daily_demand - house.reduction * avg_agent_multiplier
                )
                # iterate the bottle_demand as well
                house.bottle_demand += house.reduction * avg_agent_multiplier
                # increase the total reduction value for this node
                reduction_val += house.reduction * avg_agent_multiplier

                # set a minimum reduction_val as 50% of daily_demand
                # print(f"House base demand {house.base_demand * 60 * 60 * 24}")
                if reduction_val > daily_demand:
                    print(f"House base demand {house.base_demand * 60 * 60 * 24}")
                    print(f"House demand pattern {curr_pat}")
                    print(
                        f"Reduction value {reduction_val} exceeds daily demand {daily_demand}"
                    )
                    reduction_val = daily_demand
            elif self.twa_process == "percentage":
                # daily demand already has information about the number
                # of agents at this house from new_mult
                house.tap_demand += daily_demand * house.change
                # iterate the bottle_demand as well
                house.bottle_demand += daily_demand * (1 - house.change)
                # increase the total reduction value for this node
                reduction_val += house.bottle_demand

            # this is the demand we want for this node. It includes the
            # reduction for agents at the node and for bw use.
            desired_demand = daily_demand - reduction_val

            if daily_demand == 0:
                new_demand_multiplier = 0
            else:
                new_demand_multiplier = desired_demand / daily_demand

            new_mult = new_mult * new_demand_multiplier

            out_pat += base_pat * new_mult

        return out_pat / len(building.households)

    # this how set demand at each node, given what was collected in function above:
    def change_demands(self):
        """
        Add the current days demand pattern to each nodes demand pattern.
        This appends a new 24 hour pattern based on the demands at each node.
        """

        for i, node in enumerate(self.nodes_w_demand):
            node_pat = np.zeros(24)
            if node not in self.wdn_nodes:
                continue
            for building_id in self.wdn_nodes[node]:
                building = self.buildings[building_id]
                building_agents = np.array(
                    dcp(building.agent_history[self.timestep - 23 : self.timestep + 1])
                )
                new_mult = building_agents / building.capacity

                # if the building is a commercial or industrial building,
                # multiply the demand pattern by the agent multiplier
                if building.type in ["com", "ind", "caf", "gro"]:
                    curr_pat = building.demand_pattern * new_mult
                else:
                    """
                    Ideally we would have the new_mult, i.e., the percentage
                    full the building is, for each household. We don't currently
                    have that available, so we are using new_mult for the
                    building. What this means is for mfh building types, the
                    new_mult will be a little off from what it should actually
                    be.
                    """
                    curr_pat = self.calc_res_demands(building)
                    # curr_pat = self.calc_res_demands(building, new_mult)

            # Add the current buildings pattern to the past building's
            # for this node
            node_pat += curr_pat

            # average the node pat based on the number of bulidings
            node_pat /= len(self.wdn_nodes[node])

            wn_node_pat = self.wn.get_pattern("node_" + node)

            # add the last 24 hours of multipliers to the exisiting pattern
            if self.timestep_day == 1:
                wn_node_pat.multipliers = node_pat
            else:
                wn_node_pat.multipliers = np.concatenate(
                    (wn_node_pat.multipliers, node_pat)
                )

    def run_hyd_hour(self):
        """
        Run the hydraulic simulation for one hour at a time.
        """
        # update demands for each node in the network based on the number of
        # agents at the node
        step_agents = list()
        for i, node in enumerate(self.nodes_w_demand):
            # we need to iterate through all nodes with demand to be able to
            # run the hydraulic simualation.
            if node in self.nodes_capacity:
                # curr_node = self.wn.get_node(node)
                capacity_node = self.nodes_capacity[node]
                agents_at_node_list = self.grid.G.nodes[node]["agent"]
                agents_at_node = len(agents_at_node_list)
                step_agents.append(agents_at_node)
                if capacity_node != 0:
                    self.sim._en.ENsetnodevalue(
                        self.node_index[node],
                        EN.BASEDEMAND,
                        self.base_demands[node] * agents_at_node / capacity_node,
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
        """
        Run the hydraulic simulation for a month. This method handles all funcs
        necessary to run the monthly simulation, including collecting the demand
        values for each node at each hour, making the new demand patterns each
        day, and running the simulation every month.
        """
        # first we need to collect the nodal demands at each hour. That means
        # we run self.collect_demands() each time this method is called.
        self.collect_demands()
        # if the timestep is a day then we need to update the demand patterns
        if (self.timestep + 1) % 24 == 0 and self.timestep != 0:
            if self.verbose == 1:
                print("Starting demand changes")
            self.change_demands()
            self.collect_house_demands()
            if self.verbose == 1:
                print("Done with demand changes")
        # if the timestep is the beginning of a week then we want to run the sim
        # also run the sim at the end of the simulation
        if (
            ((self.timestep + 1) / 24) % 30 == 0
            and self.timestep != 0
            or (self.timestep + 1) / 24 == self.days
        ):
            # first set the demand patterns for each node
            for node in self.nodes_w_demand:
                # if node in self.nodes_capacity:
                node_pattern = self.wn.get_pattern("node_" + node)
                # print(node_pattern.multipliers)
                self.sim._en.ENsetpattern(
                    self.sim._en.ENgetpatternindex("node_" + node),
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
                for node, building in self.buildings.items():
                    if building.households is not None:
                        for house in building.households:
                            node_age = self.sim._results.node["quality"].loc[
                                :, house.node
                            ]
                            house.update_household(node_age.iloc[-1] / 3600)

                """ collect household level data """
                self.collect_household_data()

            # if we aren't allowing bottled water buying then we still need to
            # calculate the cost of tap water
            if not self.warmup and not self.bw:
                step_tw_cost = list()
                step_tw_demand = list()
                # for each house, calculate the demand and cost of tap water
                for node, building in self.buildings.items():
                    if building.households is not None:
                        for house in building.households:
                            house.calc_tap_cost()
                            step_tw_demand.append(dcp(house.tap_demand))
                            house.tap_demand = 0
                            step_tw_cost.append(dcp(house.tap_cost))
                self.tw_cost[self.timestep] = step_tw_cost
                self.tw_demand[self.timestep] = step_tw_demand

    def run_hydraulic(self):
        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim("id" + str(self.id))

        demand = results.node["demand"] * 1000000
        demand = demand.astype("int")
        self.demand_matrix = demand
        self.pressure_matrix = results.node["pressure"]
        self.age_matrix = results.node["quality"]
        flow = results.link["flowrate"] * 1000000
        flow = flow.astype("int")
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
            print("\tPeople informed: " + str(info_stat_all))

    def compliance_status(self):
        compl_stat_all = 0
        for i, a in enumerate(self.schedule.agents):
            if self.schedule.agents[i].compliance == 1:
                compl_stat_all += 1
            else:
                pass

        if self.verbose == 1:
            print("\tPeople complying: " + str(compl_stat_all))

    def check_agent_change(self):
        """Function to check each agent for a change in household COVID"""
        for agent in self.schedule.agents:
            # assumes the person talks with household members immediately after
            # finding out about COVID infection
            if agent.infectious_time == 0:
                pass
            elif agent.infectious_time == 1 or agent.infectious_time % (24 * 5) == 0:
                if agent.home_node != "TN1372":
                    agent.change_house_adj()

    def check_agent_loc(self):
        self.wrong_node = 0
        for i, agent in enumerate(self.schedule.agents):
            if agent.pos == agent.work_node:
                if agent not in self.grid.G.nodes[agent.work_node]["agent"]:
                    self.wrong_node += 1
                    print(f"Agent {agent} is not at its work node.")
            elif agent.pos == agent.home_node:
                if agent not in self.grid.G.nodes[agent.home_node]["agent"]:
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
        """
        Lengthen the 5 built-in patterns to match the number of days in the
        simulation
        """
        for i in range(1, 6, 1):
            curr_pat = self.wn.get_pattern(str(i))
            curr_mult = dcp(curr_pat.multipliers)
            for j in range(self.days):
                curr_pat.multipliers = np.concatenate((curr_pat.multipliers, curr_mult))

    def plot_water_age(self, age):
        """
        Saves a plot of the water age.

        Parameters
        ----------
        age : list
            age values
        """
        age = age.mean(axis=1).rolling(24).mean()
        age.plot()
        plt.savefig(
            "water_age_" + str(self.timestep) + ".png",
            format="png",
            bbox_inches="tight",
        )
        plt.close()

    def check_water_age(self):
        """
        Check the difference in water age between the first and last timestep
        in the current hydraulic results
        """
        # get the most recent results
        curr_results = self.sim.get_results()
        # parse last and last water age lists
        # mean_age = curr_results.node['quality'].loc[:, self.age_nodes].mean(axis=1) / 3600
        mean_age = curr_results.node["quality"].mean(axis=1) / 3600
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
        if self.verbose > 0:
            print(self.water_age_slope)

    def collect_agent_data(self):
        """BBN input containers"""
        step_cov_pers = list()
        step_cov_ff = list()
        step_media = list()

        """ BBN output containers """
        step_wfh = list()
        step_dine = list()
        step_groc = list()
        step_ppe = list()

        for agent in self.schedule.agents:
            step_cov_pers.append(dcp(agent.agent_params["COVIDexp"]))
            step_cov_ff.append(dcp(agent.agent_params["COVIDeffect_4"]))
            step_media.append(dcp(agent.agent_params["MediaExp_3"]))

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

    def collect_house_demands(self):
        step_bw_demand = list()
        step_tw_demand = list()
        for node, building in self.buildings.items():
            if building.households is not None:
                for house in building.households:
                    step_bw_demand.append(dcp(house.bottle_demand))
                    step_tw_demand.append(dcp(house.tap_demand))

        self.bw_demand[self.timestep] = step_bw_demand
        self.tw_demand[self.timestep] = step_tw_demand

    def collect_household_data(self):
        """Income output containers"""
        step_bw_cost = list()
        step_tw_cost = list()
        step_hygiene = list()
        step_drink = list()
        step_cook = list()

        for node, building in self.buildings.items():
            if building.households is not None:
                for house in building.households:
                    step_bw_cost.append(dcp(house.bottle_cost))
                    step_tw_cost.append(dcp(house.tap_cost))
                    hygiene = 1 if "hygiene" in house.bottle else 0
                    drink = 1 if "drink" in house.bottle else 0
                    cook = 1 if "cook" in house.bottle else 0
                    step_hygiene.append(hygiene)
                    step_drink.append(drink)
                    step_cook.append(cook)

        self.bw_cost[self.timestep] = step_bw_cost
        self.tw_cost[self.timestep] = step_tw_cost

        self.hygiene[self.timestep] = step_hygiene
        self.drink[self.timestep] = step_drink
        self.cook[self.timestep] = step_cook

    def calc_equity_metrics(self, income, cow):
        """
        Calculate the equity metrics for the given time period.

        See:
        https://nicholasinstitute.duke.edu/water-affordability/affordability/about_dashboard.html#metrics

        Parameters
        ----------
        income : np.array
            income for each household

        cow : np.array
            cost of water for each household for the given timestep

        """
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
        """Increment day time step"""
        self.timestep_day += 1

        if not self.warmup:
            """Check status of agents with COVID exposure and infections"""
            for i, agent in enumerate(self.schedule.agents):
                if agent.covid == "exposed":
                    agent.check_infectious()
                elif agent.covid == "infectious":
                    agent.check_symptomatic()
                    if agent.inf_severity >= 1:
                        agent.check_severity()
                    else:
                        pass
                    agent.check_recovered()
                    agent.check_death()

                # this means that agents only check wfh/no_dine/grocery, a max of 5 times throughout the 90 days
                if agent.adj_covid_change == 1:
                    if agent.wfh == 0 and "wfh" in self.bbn_models:
                        agent.predict_wfh()

                    if agent.no_dine == 0 and "dine" in self.bbn_models:
                        agent.predict_dine_less()

                    if agent.less_groceries == 0 and "grocery" in self.bbn_models:
                        agent.predict_grocery()

                    if agent.ppe == 0 and "ppe" in self.bbn_models:
                        agent.predict_ppe()

            # collect the bbn and sv data for each agent
            self.collect_agent_data()

            """ Set the wfh threshold if lag time has been reached """
            if self.stat_tot[3] > self.wfh_lag and not self.wfh_thres:
                self.wfh_thres = True

    def eos_tasks(self):
        """End of simulation tasks"""
        # collect the demands from each node each hour
        self.collect_demands()
        if (self.timestep + 1) / 24 == self.days:
            # setup demand patterns so they extend self.days
            self.update_patterns()
            self.wn.options.time.duration = 3600 * 24 * self.days
            self.wn.options.quality.parameter = "AGE"
            # run the hydraulic simulation and collect results
            self.run_hydraulic()
        if (self.timestep + 1) % 24 == 0:
            # update the demand patterns with the current days demand
            self.change_demands()

    def industry_agents(self):
        return np.where(self.ind_agents == 1)[0]
        # return [a for a in self.schedule.agents if a.building in self.ind_nodes]

    def commercial_agents(self):
        return np.where(self.com_agents == 1)[0]
        # return [a for a in self.schedule.agents if a.building in self.com_nodes]

    def cafe_agents(self):
        return np.where(self.caf_agents == 1)[0]
        # return [a for a in self.schedule.agents if a.building in self.caf_nodes]

    def residential_agents(self):
        return np.where(self.res_agents == 1)[0]
        # return [a for a in self.schedule.agents if a.building == a.home_node]

    def infect_agent(self, agent, next_loc):
        if agent.covid == "infectious" and not self.warmup:
            self.contact(agent, next_loc)

    def agents_wfh(self):
        return [a for a in self.schedule.agents if a.wfh == 1]

    def print_func(self):
        p1 = ["t", "S", "E", "I", "R", "D"]
        p2 = ["Symp", "Asymp", "Mild", "Sev", "Crit", "sum_I", "wfh"]
        print("Hour step: " + str(self.timestepN))
        print("Time step: " + str(self.timestep))
        print("Day step: " + str(self.timestep_day))
        print("\n")
        # out_1 = ''
        # out_2 = ''
        # counter = 0
        # for i, item in enumerate(self.stat_tot):
        #     if i < 6:
        #         out_1 = out_1 + ' ' + p1[i] + ': ' + '{:.2f}'.format(item)
        #     else:
        #         out_2 = out_2 + ' ' + p2[counter] + ': ' + '{:.2f}'.format(item)
        #         counter += 1

        # print('\t', out_1)
        # print('\t', out_2)
        # print('\tStatus (%): ', ['{:.2f}'.format(i) for i in self.stat_tot])
        # print('\tStatus (#): ', ['{:.3f}'.format(i) * self.num_agents for i in self.stat_tot])
        print("\n")
        print("\tAgents at industrial nodes: " + str(len(self.industry_agents())))
        print("\tAgents at commercial nodes: " + str(len(self.commercial_agents())))
        print("\tAgents at restaurant nodes: " + str(len(self.cafe_agents())))
        print("\n")
        print("\tAgents at home: " + str(len(self.residential_agents())))
        # print('\n')
        # print('\tAgents with close COVID: ' + str(self.check_covid_change()))

    def step(self):
        self.schedule.step()
        if self.timestep != 0:
            if self.verbose == 1:
                print("Starting com move")
            self.move_com()
            if self.verbose == 1:
                print("Done with com move")
                print("Starting cafe move")
            self.move_caf()
            if self.verbose == 1:
                print("Done with cafe move")
                print("Starting ind move")
        # self.check_agent_loc()
        # self.check_covid_change()
        # BV: changed times to 6, 14, and 22 because I think this is more representative
        # of a three shift schedule. Unsure if this changes anything with water
        # patterns, but I suspect it might.
        if (
            self.timestep == 0
            or self.timestepN == 5
            or self.timestepN == 13
            or self.timestepN == 21
        ):
            self.move_ind()

        # make sure that all agents that were moved to buildings
        # are transferred to their respective household objects
        for node, building in self.buildings.items():
            if building.households is not None:
                building.agents_to_household()

        # COVID related methods are not run during warmup
        if not self.warmup:
            self.check_status()
            self.check_agent_change()
            self.communication_utility()
            # self.collect_agent_data()

        # daily updating is done during warmup
        if (self.timestep + 1) % 24 == 0 and self.timestep != 0:
            self.daily_tasks()

        if self.hyd_sim == "eos":
            self.eos_tasks()
        elif self.hyd_sim == "hourly":
            self.run_hyd_hour()
        elif self.hyd_sim == "monthly":
            if self.verbose == 1:
                print("Starting hydraulic step")
            self.run_hyd_monthly()
            if self.verbose == 1:
                print("Done with hydraulic step")
        else:
            NotImplementedError(f"Hydraulic simulation {self.hyd_sim} not set up.")

        # self.inform_status()
        # self.compliance_status()
        if not self.warmup:
            self.num_status()

        if self.verbose == 1:
            self.print_func()

        self.timestepN = self.timestep + 1 - self.timestep_day * 24
        self.timestep += 1

        if self.water_age_slope < self.tol:
            self.warmup = False
