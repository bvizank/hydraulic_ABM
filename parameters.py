from mesa import Model
import pandas as pd
import numpy as np
import utils as ut
import city_info as ci
from agent_model import Household, Building
from hydraulic import EpanetSimulator_Stepwise
import data as dt
import networkx as nx
import bnlearn as bn
import wntr
import os
import csv
import math
from copy import deepcopy as dcp
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import matplotlib.pyplot as plt


class Parameters(Model):
    """
    Create the list of parameters that will be used for the given
    simulation.
    """

    def __init__(self, N, city, days, id, seed, **kwargs):
        super().__init__()
        """ Set the default parameters """
        self.output_loc = None
        self.days = days
        self.id = id
        self.num_agents = N
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.network = city
        self.t = 0
        self.schedule = RandomActivation(self)
        self.timestep = 0
        self.timestep_day = 0
        self.timestepN = 0
        # self.base_demands_previous = {}
        self.snw_agents = {}
        # self.nodes_endangered = All_terminal_nodes
        self.demand_test = []

        """ COVID-19 exposure pars """
        self.exposure_rate = 0.05  # infection rate per contact per day in households
        self.exposure_rate_large = (
            0.01  # infection rate per contact per day in workplaces
        )
        self.e2i = (4.5, 1.5)  # mean and sd number of days before infection shows
        self.i2s = (
            1.1,
            0.9,
        )  # time after viral shedding before individual shows sypmtoms
        self.s2sev = (
            6.6,
            4.9,
        )  # time after symptoms start before individual develops potential severe covid
        self.sev2c = (1.5, 2.0)  # time after severe symptoms before critical status
        self.c2d = (10.7, 4.8)  # time between critical dianosis to death
        self.recTimeAsym = (8.0, 2.0)  # time for revovery for asymptomatic cases
        self.recTimeMild = (
            8.0,
            2.0,
        )  # mean and sd number of days for recovery: mild cases
        self.recTimeSev = (18.1, 6.3)
        self.recTimeC = (18.1, 6.3)

        self.covid_exposed = int(0.001 * self.num_agents)
        self.cumm_infectious = self.covid_exposed
        self.daily_contacts = 10
        self.verbose = 1

        self.res_pat_select = "lakewood"
        self.wfh_lag = 0
        self.no_wfh_perc = 0.5
        self.wfh_thres = False  # whether wfh lag has been reached

        self.bbn_models = ["wfh", "dine", "grocery", "ppe"]

        """ Import the four DAGs for the BBNs """
        self.wfh_dag = bn.import_DAG(
            "Input Files/data_driven_models/work_from_home.bif", verbose=0
        )
        self.dine_less_dag = bn.import_DAG(
            "Input Files/pmt_models/dine_out_less_pmt-6.bif", verbose=0
        )
        self.grocery_dag = bn.import_DAG(
            "Input Files/pmt_models/shop_groceries_less_pmt-6.bif", verbose=0
        )
        self.ppe_dag = bn.import_DAG(
            "Input Files/data_driven_models/mask.bif", verbose=0
        )

        """
        This value comes from this paper:
        https://www.cdc.gov/mmwr/volumes/71/wr/mm7106e1.htm#T3_down

        Odds that you will get covid if you wear a mask is
        66% less likely than without a mask, therefore, the new
        exposure rate is 34% of the original.
        """
        self.ppe_reduction = 0.34

        """
        hyd_sim represents the way the hydraulic simulation is to take place
        options are 'eos' (end of simulation) and 'hourly' with the default 'eos'
        """
        self.hyd_sim = "eos"
        """
        The warmup input dictates whether a warmup period is run to reach steady
        state water age values. Default is true
        """
        self.warmup = True
        """
        Warmup tolerance is the threshold for when warmup period is complete
        """
        self.tol = 0.001
        """
        bw dictates whether bottled water buying is modeled. Defaults to True
        """
        self.bw = True
        """
        twa_mods are the modulators to the twa thresholds passed to households
        """
        self.twa_mods = [130, 140, 150]
        """
        self.ind_min_demand is the minimum industrial demand as a percentage
        """
        self.ind_min_demand = 0
        """
        dist_income is whether income is industrial distance based
        """
        self.dist_income = True
        """
        twa_process dictates whether the twas are represented as absolute
        reductions or percentage reductions

        two options are absolute and percentage
        """
        self.twa_process = "absolute"

        """ weights of number of people per household from ACS S2501 """
        self.weights = [41.8, 28.5, 12.3, 17.4 / 3, 17.4 / 3, 17.4 / 3]

        """ skeletonized network """
        self.skeleton = False

        """ Data collectors """
        # self.agent_matrix = dict()

        """ Initialize the COVID state variable collectors """
        self.cov_pers = dict()
        self.cov_ff = dict()
        self.media_exp = dict()

        """ Initialize the PM adoption collectors """
        self.wfh_dec = dict()
        self.dine_dec = dict()
        self.groc_dec = dict()
        self.ppe_dec = dict()

        """ Initialize household income and COW collectors """
        self.bw_cost = dict()
        self.tw_cost = dict()
        self.bw_demand = dict()
        self.traditional = dict()
        self.burden = dict()

        self.hygiene = dict()
        self.drink = dict()
        self.cook = dict()

        """ Initialize status collectors for COVID-19 """
        status_tot = [
            0,
            self.num_agents - self.covid_exposed,
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
            len(self.agents_wfh()),
        ]
        status_tot = [i / self.num_agents for i in status_tot]
        self.status_tot = {0: status_tot}

        """ Update the above parameters with the supplied kwargs """
        self.update_pars(**kwargs)

    def update_pars(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                if key == "bbn_models" and "all" in value:
                    value = ["wfh", "dine", "grocery", "ppe"]
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid attribute")

    def save_pars(self, output_loc):
        """
        Save parameters to a DataFrame, param_out, to save at the end of the
        simulation. This helps with data organization.
        """
        with open(output_loc + "/datasheet.csv", "w", newline="") as csvfile:
            params = csv.writer(csvfile, delimiter=",")
            params.writerow(["Param", "Value"])
            for key, value in self.__dict__.items():
                params.writerow([key, value])

    def capacity_helper(self, x, ind, com, res):
        if x["type"] == "res":
            if res:
                return self.random.choices(range(1, 7), weights=self.weights, k=1)[0]
            else:
                return x["capacity"]
        if x["type"] == "com":
            # we want each commercial building to have a capacity of at least
            # 2
            return int(max(math.ceil(x["area"] / com), 2))
            # return self.random.gammavariate(1.4, 8)
        if x["type"] == "ind":
            return int(max(math.ceil(x["area"] / ind), 2))

    def building_helper(self, x):
        building = Building(
            x.name,
            x["capacity"],
            x["wdn_node"],
            x["type"],
            x["area"],
            x["parusedsc2"],
            self,
        )
        return building

    def household_helper(self, x):
        house = Household(
            x.name,
            x["total_res"] - x["capacity"],
            x["total_res"],
            x["wdn_node"],
            None,
            self.twa_mods,
            self,
            x["capacity"],
            x["bg"],
            x["area"],
        )
        return house

    def assign_capacity(self, ind_denom, com_denom, res=False):
        # assign each building a capacity based on type
        # self.node_buildings["capacity"] = self.node_buildings.apply(
        #     self.capacity_helper, axis=1
        # )

        self.node_buildings["capacity"] = self.node_buildings.apply(
            self.capacity_helper, args=(ind_denom, com_denom, res), axis=1
        )

        # make capacity an integer
        self.node_buildings["capacity"] = (
            self.node_buildings["capacity"].fillna(0).astype(int)
        )

        # initialize the number of agents based on the capacity at res buildings
        self.num_agents = self.node_buildings.groupby("type")["capacity"].sum()["res"]
        if self.verbose > 0:
            print(f"Total number of agents: {self.num_agents}")

        """ Set the number of work agents and the locations that need workers """
        self.num_ind_agents = self.node_buildings.groupby("type")["capacity"].sum()[
            "ind"
        ]
        if self.verbose > 0:
            print(f"Capacity of all industrial nodes: {self.num_ind_agents}")

        # calculate the number of commercial spots
        self.num_com_agents = self.node_buildings.groupby("type")["capacity"].sum()[
            "com"
        ]
        if self.verbose > 0:
            print(f"Capacity of all commercial nodes: {self.num_com_agents}")

        # initialize the number of agents to be moved each industrial move step
        self.ind_agent_n = int(dcp(self.num_ind_agents) / 2)

        # print(f"Distribution based industrial spots: {self.ind_agent_n}")

    def iterate_capacity(self):
        # factor to convert building area to capacity
        ind_denom = 1000
        com_denom = 1000
        self.assign_capacity(ind_denom, com_denom, res=True)

        ind_cap_difference = self.num_ind_agents - self.num_agents * 0.25

        while abs(ind_cap_difference) > 100:
            if ind_cap_difference > 0:
                ind_denom += 10
                self.assign_capacity(ind_denom, com_denom)
            else:
                ind_denom -= 10
                self.assign_capacity(ind_denom, com_denom)
            ind_cap_difference = self.num_ind_agents - self.num_agents * 0.25

        # the self.num_agents * 0.3 term is the number of agents that don't
        # move to a commercial or work node during a given day. This would be
        # people like elderly, kids under a certain age, and unemplyed people.
        com_cap_difference = self.num_com_agents - (
            self.num_agents - self.num_ind_agents - self.num_agents * 0.2
        )

        while abs(com_cap_difference) > 100:
            if com_cap_difference > 0:
                com_denom += 10
                self.assign_capacity(ind_denom, com_denom)
            else:
                com_denom -= 10
                self.assign_capacity(ind_denom, com_denom)
            com_cap_difference = self.num_com_agents - (
                self.num_agents - self.num_ind_agents - self.num_agents * 0.2
            )

    def setup_real(self, city, wn_name=None):
        """
        Import the required data for a real city with a synthetic WDN

        1. Import the water network model using WNTR
        2. Import the demand patterns for residential, cafe, commercial,
           and industrial nodes.
        3. Import the relative distribution of agents amongst node types.
        4. Assign each building a node in the WDN.
        5. Instantiate a household for each residential building.

        Parameters:
        -----------
            city :: str
                The name of the city. There is logic to weed out names
                without data.
        """
        # set skeleton to true because most real networks will be skeletonized
        # could add argument to setup_real to make this adjustable
        self.skeleton = True

        city_dir = os.path.join("Input Files/cities", city)

        """ Import the water network model using WNTR """
        if wn_name:
            inp_file = os.path.join(city_dir, wn_name + ".inp")
            if not os.path.exists(inp_file):
                raise ValueError(f"File {wn_name + '.inp'} does not exist.")
        else:
            inp_file = os.path.join(city_dir, city + ".inp")
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.nodes_w_demand = [j for j, _ in self.wn.junctions()]

        """ Import the demand patterns """
        self.demand_patterns = pd.read_csv(
            os.path.join(city_dir, "patterns.csv"), delimiter=","
        )

        """ Import the distributions of agents """
        self.node_distributions = pd.read_csv(
            os.path.join(city_dir, "hourly_population.csv"), delimiter=","
        )

        """ Import the income distributions for each block group """
        self.income_dist = pd.read_csv(
            os.path.join(city_dir, "income_bg.csv"),
            delimiter=",",
            index_col=0,
            dtype="int64",
        )

        """ import demographic data using pandas """
        self.demo = pd.read_csv(
            os.path.join(city_dir, "demographics_bg.csv"),
            delimiter=",",
            index_col=0,
            dtype="float64",
        )

        self.income_dist.columns = [int(i) for i in self.income_dist.columns]

        """ Assign each building a node in the WDN """
        self.node_buildings = ci.make_building_list(self.wn, city, city_dir)

        """ Assign the nodes with capacity values """
        self.iterate_capacity()
        # plt.hist(self.node_buildings.query('type == "com"').loc[:, "capacity"])
        # plt.show()

        """ Get a running tally of residential agents """
        self.node_buildings["total_res"] = self.node_buildings["capacity"][
            self.node_buildings["type"] == "res"
        ].cumsum()

        self.covid_exposed = int(0.001 * self.num_agents)
        # self.ind_agent_n = int(self.num_agents * max(self.node_distributions['ind']))
        self.com_dist = self.node_distributions["com"].tolist()
        self.cafe_dist = self.node_distributions["caf"].tolist()
        print(self.com_dist)

        # convert floats from cumsum to ints
        self.node_buildings["total_res"] = (
            self.node_buildings["total_res"].fillna(0).astype(int)
        )

        self.work_loc_list = self.node_list(
            self.node_buildings[self.node_buildings["type"] == "ind"]["capacity"],
            (
                self.node_buildings[
                    self.node_buildings["type"] == "ind"
                ].index.to_list()
                + self.node_buildings[
                    self.node_buildings["type"] == "ind"
                ].index.to_list()
            ),
        )

        if self.verbose > 0:
            print(f"Total number of agents with a work node: {len(self.work_loc_list)}")

        # define lists with each node type
        self.nav_nodes = []

        self.ind_nodes = self.node_buildings[
            self.node_buildings["type"] == "ind"
        ].index.to_numpy()

        self.res_nodes = self.node_buildings[
            self.node_buildings["type"] == "res"
        ].index.to_numpy()

        self.com_nodes = self.node_buildings[
            self.node_buildings["type"] == "com"
        ].index.to_numpy()

        restaurants = ["RESTAURANT LOUNGE", "FAST FOOD RESTAURAN"]
        grocers = ["MARKET"]

        print(
            self.node_buildings["parusedsc2"]
            .str.split("|")
            .str.get(0)
            .isin(restaurants)
        )

        self.caf_nodes = self.node_buildings[
            (
                self.node_buildings["parusedsc2"]
                .str.split("|")
                .str.get(0)
                .isin(restaurants)
            )
            & (self.node_buildings["type"] == "com")
        ].index.to_numpy()

        self.gro_nodes = self.node_buildings[
            (self.node_buildings["parusedsc2"].str.split("|").str.get(0).isin(grocers))
            & (self.node_buildings["type"] == "com")
        ].index.to_numpy()

        self.num_caf_agents = self.node_buildings["capacity"][
            self.node_buildings.index.isin(self.caf_nodes)
        ].sum()

        # the number of com agents does not include the caf agents
        self.num_com_agents -= self.num_caf_agents

        print("Cafe node capacity:")
        print(self.num_caf_agents)

        print("Grocery node capacity:")
        print(
            self.node_buildings["capacity"][
                self.node_buildings.index.isin(self.gro_nodes)
            ].sum()
        )

        self.com_nodes = np.setdiff1d(self.com_nodes, self.caf_nodes)
        # self.com_nodes = np.setdiff1d(self.com_nodes, self.gro_nodes)
        # select the number of cafe nodes
        # inds = self.random.sample(
        #     range(len(self.com_nodes)), int(len(self.com_nodes) * 0.05)
        # )
        # self.caf_nodes = self.com_nodes[inds]
        # self.com_nodes = np.delete(self.com_nodes, inds)

        # select the number of grocery nodes
        # inds = self.random.sample(
        #     range(len(self.com_nodes)), int(len(self.com_nodes) * 0.01)
        # )
        # self.gro_nodes = self.com_nodes[inds]
        # self.com_nodes = np.delete(self.com_nodes, inds)

        # define industrial nodes that do not allow work from home
        self.total_no_wfh = self.random.choices(
            population=self.ind_nodes, k=int(len(self.ind_nodes) * self.no_wfh_perc)
        )

        # print(self.work_loc_list)

        # init tracking arrays for each node type
        if self.verbose > 0:
            print(self.num_agents)
        self.ind_agents = np.zeros(self.num_agents, dtype=np.int64)
        self.res_agents = np.zeros(self.num_agents, dtype=np.int64)
        self.com_agents = np.zeros(self.num_agents, dtype=np.int64)
        self.caf_agents = np.zeros(self.num_agents, dtype=np.int64)

        # init dictionary of agents
        self.agents_list = dict()

        # make dictionary of building objects
        self.buildings = (
            self.node_buildings[~self.node_buildings["type"].isin(["res"])].apply(
                self.building_helper, axis=1
            )
        ).to_dict()

        # print(self.node_buildings.index)

        # initialize income values for all of the households in the sim
        self.income_list = dict()
        for i, row in self.income_dist.iterrows():
            grp_size = len(self.node_buildings.query('type == "res" and bg == @i'))

            if self.verbose > 0:
                print(f"Group size for bg {i}: {grp_size}")
            if grp_size > 0:
                self.income_list[i] = ut.income_list(
                    data=row, n_house=grp_size * 1.1, model=self
                )

        # print([len(v) for i, v in self.income_list.items()])

        # make dictionary of household objects
        self.households = (
            self.node_buildings[self.node_buildings["type"].isin(["res"])].apply(
                self.household_helper, axis=1
            )
        ).to_dict()

        # now it includes all of the households.
        self.buildings.update(self.households)

        building_group = self.node_buildings.groupby("wdn_node")
        self.wdn_nodes = building_group.indices

        # set the base demand for each node based on the buildings
        for name, node in self.wn.junctions():
            demand = 0
            """ Set demand to zero if there are no bulidings that are
            assigned to the current node """
            if name not in self.wdn_nodes:
                continue
            for building_id in self.wdn_nodes[name]:
                building = self.buildings[building_id]
                # print(f"Building {building_id} with type {building.type} has a demand of {building.base_demand}")
                demand += building.base_demand
            node.demand_timeseries_list[0].base_value = demand

        # make array indicating which agents have industrial worktypes
        self.ind_work_nodes = np.zeros(self.num_agents, dtype=np.int64)
        inds = list()
        nodes = list()
        for a, o in self.agents_list.items():
            if o.work_type == "industrial":
                inds.append(a)
                nodes.append(o.work_node)

        self.ind_work_nodes[inds] = nodes

        if self.verbose == 0.5:
            print("Setting agent attributes ...............")
        self.set_attributes()
        self.dag_nodes()
        self.create_comm_network()

        self.save_wn(city)
        self.init_hydraulic(virtual=False)

        self.init_income()

        """ Save the list of buildings and nodes """
        self.node_buildings.to_pickle("buildings.pkl")

        # print(self.node_buildings.groupby('wdn_node')['demand'].mean())
        # print(self.node_buildings.groupby('wdn_node')['pattern'].mean())
        # print(self.node_buildings.columns)

    def setup_virtual(self, network):
        if network == "micropolis":
            inp_file = "Input Files/micropolis/MICROPOLIS_v1_inc_rest_consumers.inp"
            data = pd.read_excel(r"Input Files/micropolis/Micropolis_pop_at_node.xlsx")
        elif network == "mesopolis":
            inp_file = "Input Files/mesopolis/Mesopolis.inp"
            data = pd.read_excel(r"Input Files/mesopolis/Mesopolis_pop_at_node.xlsx")
        pattern_list, self.wn = ut.init_wntr(inp_file)

        # input the number of agents required at each node type at each time
        node_id = data["Node"].tolist()
        maxpop_node = data["Max Population"].tolist()
        if network == "mesopolis":
            house_num = data["HOUSE"].tolist()
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
            self.air_nodes = ut.find_nodes("air", pattern_list, network)
            self.com_nodes = ut.find_nodes("com", pattern_list, network)
            self.res_nodes = ut.find_nodes("res", pattern_list, network)
            self.ind_nodes = ut.find_nodes("ind", pattern_list, network)
            self.nav_nodes = ut.find_nodes("nav", pattern_list, network)
            self.cafe_nodes = ut.find_nodes("cafe", pattern_list, network)

        self.terminal_nodes = (
            self.cafe_nodes
            + self.res_nodes
            + self.ind_nodes
            + self.com_nodes
            + self.air_nodes
            + self.nav_nodes
        )

        # finish setup process by loading distributions of agents at each node type,
        # media data, and the distance between residential nodes and closest ind.
        # node.
        pop_dict = ut.load_distributions(network)
        self.res_dist = pop_dict["res"]  # residential capacities at each hour
        self.com_dist = pop_dict["com"]  # commercial capacities at each hour
        self.ind_dist = pop_dict["ind"]  # industrial capacities at each hour
        self.sum_dist = pop_dict["sum"]  # sum of capacities
        self.cafe_dist = pop_dict["cafe"]  # restaurant capacities at each hour
        if network == "micropolis":
            # self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
            self.nav_dist = [0]  # placeholder for agent assignment
            self.ind_agent_n = max(self.ind_dist)
        if network == "mesopolis":
            self.air_dist = pop_dict["air"]
            self.nav_dist = pop_dict["nav"]
            self.ind_agent_n = max(self.ind_dist) + max(self.nav_dist)

        # calculate the distance to nearest industrial node
        self.ind_node_dist, na = ut.calc_industry_distance(
            self.wn, node_dict["ind"], nodes=node_dict["res"]
        )

        self.setup_grid()

        """ Set up the rest of the agent information """
        self.create_node_list()
        if self.verbose == 0.5:
            print("Creating agents ..............")
        self.create_agents(virtual=True)
        if self.verbose == 0.5:
            print("Setting agent attributes ...............")
        self.set_attributes()
        self.dag_nodes()
        self.create_comm_network()

        self.init_hydraulic()

    def setup_grid(self):
        """Set up the grid for agent movement"""
        self.G = self.wn.get_graph()
        self.grid = NetworkGrid(self.G)
        self.num_nodes = len(self.G.nodes)

    def setup_loc_matrices(self):
        """Setup the matrices that keep track of each agent at each building"""
        # find the building ids that correspond to each building type
        self.res_buildings = self.node_buildings[
            self.node_buildings["type"] == "res"
        ].index
        self.com_buildings = self.node_buildings[
            self.node_buildings["type"] == "com"
        ].index
        self.ind_buildings = self.node_buildings[
            self.node_buildings["type"] == "ind"
        ].index

        # make the location matrices
        self.res_loc = np.zeros((len(self.res_buildings), self.num_agents))
        self.com_loc = np.zeros((len(self.com_buildings), self.num_agents))
        self.ind_loc = np.zeros((len(self.ind_buildings), self.num_agents))

    def dag_nodes(self):
        self.wfh_nodes = dcp(self.wfh_dag["adjmat"].columns)
        self.dine_nodes = dcp(self.dine_less_dag["adjmat"].columns)
        self.grocery_nodes = dcp(self.grocery_dag["adjmat"].columns)
        self.ppe_nodes = dcp(self.ppe_dag["adjmat"].columns)

    def base_demand_list(self):
        self.base_demands = dict()
        self.base_pattern = dict()
        self.node_index = dict()
        # self.demand_multiplier = dict()
        for node in self.nodes_w_demand:
            node_1 = self.wn.get_node(node)
            self.base_demands[node] = node_1.demand_timeseries_list[0].base_value

            """ Make a pattern for each node for hydraulic simulation """
            if self.hyd_sim == "eos" or self.hyd_sim == "monthly":
                curr_pattern = dcp(node_1.demand_timeseries_list[0].pattern)
                self.wn.add_pattern("node_" + node, curr_pattern.multipliers)
                # set the demand pattern for the node to the new pattern
                node_1.demand_timeseries_list[0].pattern_name = "node_" + node
            # elif self.hyd_sim == 'hourly' or isinstance(self.hyd_sim, int):
            #     self.node_index[node] = dcp(self.sim._en.ENgetnodeindex(node))
            self.base_pattern[node] = (
                dcp(node_1.demand_timeseries_list[0].pattern_name),
                dcp(self.wn.get_pattern(node_1.demand_timeseries_list[0].pattern_name)),
            )

    def set_age(self):
        """Set initial water age"""
        init_age = pd.read_pickle("hot_start_clinton.pkl")
        for name in self.nodes_w_demand:
            if name not in self.wdn_nodes:
                continue
            curr_node = self.wn.get_node(name)
            curr_node.initial_quality = float(init_age.loc[[name]].values[0])

    def node_list(self, list, nodes):
        list_out = []
        for node in nodes:
            for i in range(int(list[node])):
                list_out.append(node)
        return list_out

    def create_node_list(self):
        nodes_industr_2x = self.ind_nodes + self.ind_nodes
        nodes_nav_2x = self.nav_nodes + self.nav_nodes
        self.work_loc_list = self.node_list(
            self.nodes_capacity, nodes_industr_2x + nodes_nav_2x
        )
        self.res_loc_list = self.node_list(self.nodes_capacity, self.res_nodes)
        # self.rest_loc_list = node_list(self.nodes_capacity, self.cafe_nodes)
        # self.comm_loc_list = node_list(self.nodes_capacity, self.com_nodes)

    def create_agents(self, virtual=True):
        """Creating lists of nodes where employers have decided not to allow
        working from home or jobs that are "essential"."""

        if virtual:
            """Potentially change this to include navy nodes"""
            self.total_no_wfh = self.random.choices(
                population=self.ind_nodes, k=int(len(self.ind_nodes) * self.no_wfh_perc)
            )

            # create dictionary of households
            self.households = dict()
            self.household_n = dict()

            self.work_agents = (max(self.ind_dist) + max(self.nav_dist)) * 2

            # CREATING AGENTS
            res_nodes = dcp(self.res_nodes)
            self.random.shuffle(res_nodes)
            ids = 0
            max_node_dist = max(self.ind_node_dist.values())

            """
            Place all the agents in the residential nodes, each filled up to
            its capacity

            The self.households and self.households_n are filled in the household
            methods (micro_ and meso_)
            """
            for node in res_nodes:
                # distance to closest industrial node relative to max distance
                # essentially a normalized distance
                node_dist = self.ind_node_dist[node] / max_node_dist
                # for spot in range(int(self.nodes_capacity[node])):
                if self.network == "micropolis":
                    ids = self.micro_household(ids, node, node_dist)
                elif self.network == "mesopolis":
                    ids = self.meso_household(ids, node)

        self.init_income()

        if self.network == "mesopolis":
            """
            Need to initialize the rest of the agents because there are only
            139,654 residential spots. Currently we are assuming that the
            remaining 7062 agents are at the airport, but this is a bad
            assumption because the max capacity of the airport is 429.
            """
            for pop in range(self.num_agents - ids):
                a = ConsumerAgent(ids, self)
                self.schedule.add(a)
                a.home_node = "TN1372"
                ids += 1

    def micro_household(self, init_id, curr_node, node_dist):
        """Assigns each agent's housemates based on the number
        of agents at the current node.

        If the residential node is larger than 6, then we need to
        artificially make households of 6 or less."""
        node_cap = dcp(self.nodes_capacity[curr_node])
        node_cap_static = node_cap
        if node_cap == 0:
            return init_id

        curr_ids = init_id

        house_list = list()

        """
        Need to account for multifamily housing, so iterating through
        residential nodes and placing agents that way and then storing their
        housemates in the agent object.

        Iterate through the agents at the current res node
        and add them to households of 1 to 6 agents
        """
        while node_cap > 6:
            # pick a random size for current household
            home_size = self.random.choice(range(1, 7))
            prev_id = curr_ids
            curr_ids += home_size
            # make the household and append it to a list of households
            house_list.append(
                Household(prev_id, curr_ids, curr_node, node_dist, self.twa_mods, self)
            )
            node_cap -= home_size
        else:
            # if the node size is 6 or fewer, we only need to do this once
            house_list.append(
                Household(
                    curr_ids,
                    node_cap_static + init_id,
                    curr_node,
                    node_dist,
                    self.twa_mods,
                    self,
                )
            )

        self.households[curr_node] = house_list
        self.household_n[curr_node] = len(house_list)

        return init_id + node_cap

    def meso_household(self, curr_node):
        """Assign each agent's housemates based on the current node.

        The current plan is to fill all residential node spots and then
        the rest of the agents are travellers at the airport."""
        # need to rectify the difference between the number of
        # residential spots (139,654) and the total population
        # (146,716)
        """ Iterate through the agents at the current res node
        and add them to households of 1 to 6 agents """
        while len(curr_node) > 6:
            # pick a random size for current household
            # a uniform distribution between 1 and 7 will average to 4
            home_size = int(self.random.uniform(1, 8))
            # pick the agents that will occupy this household
            curr_housemates = self.random.choices(curr_node, k=home_size)
            # remake the curr_node variable without the agents just chosen
            curr_node = [a for a in curr_node if a not in curr_housemates]
            """ Assign the current list of housemates to each agent
            so they each know who their housemates are """

            for mate in curr_housemates:
                agent = self.schedule._agents[mate]
                # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
                agent.housemates = dcp(curr_housemates)  # this includes current agent

        for mate in curr_node:
            agent = self.schedule._agents[mate]
            # agent = [a for a in self.schedule.agents if a.unique_id == mate][0]
            agent.housemates = dcp(curr_node)

    def init_income(self):
        # collect income and income level from each household that was just created
        self.income = [h.income for n, h in self.households.items()]
        self.income_level = [h.income_level for n, h in self.households.items()]
        self.hh_size = [len(h.agent_ids) for n, h in self.households.items()]
        self.income_comb = pd.DataFrame(
            data={
                "income": self.income,
                "level": self.income_level,
                "hh_size": self.hh_size,
                "id": list(self.households.keys()),
            },
            index=[h.node for n, h in self.households.items()],
        )

    def set_attributes(self):
        """
        Assign agents an age 1 = 0-19, 2 = 20-29, 3 = 30-39, 4 = 40-49,
        5 = 50-59, 6 = 60-69, 7 = 70-79, 8 = 80-89, 9 = 90+
        """
        ages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        age_weights = [0.25, 0.18, 0.15, 0.14, 0.12, 0.08, 0.05, 0.01, 0.01]

        """
        Assign agents either susceptible or infected to COVID based on initial
        infected number.
        """
        exposed_sample = self.random.sample(
            [i for i, a in enumerate(self.schedule.agents)], self.covid_exposed
        )
        for i, agent in enumerate(self.schedule.agents):
            """Set age of each agent"""
            agent.age = self.random.choices(population=ages, weights=age_weights, k=1)[
                0
            ]

            """ Set covid infection for each agent """
            if i in exposed_sample:
                agent.covid = "infectious"
            else:
                agent.covid = "susceptible"

            """ Set bbn parameters for each agent """
            agent_set_params = dt.bbn_params[
                self.random.randint(0, dt.bbn_params.shape[0] - 1), :
            ]
            # agent_set_params = dt.bbn_params.sample()
            # agent.agent_params['risk_perception_r'] = int(agent_set_params['risk_perception_r']) - 1
            for i, param in enumerate(dt.bbn_param_list):
                agent.agent_params[param] = agent_set_params[i] - 1

    def create_comm_network(
        self,
    ):  # a random set of agents from household (all agents live at the same node)
        """
        CREATING COMMUNICATION NETWORK WITH SWN = SMALL WORLD NETWORK
        Assigning Agents randomly to nodes in SWN
        """
        self.swn = nx.watts_strogatz_graph(
            n=self.num_agents, p=0.2, k=6, seed=self.seed
        )
        for agent in self.schedule.agents:
            agent.friends = [x for x in self.swn.neighbors(agent.unique_id)]

    def save_wn(self, city):
        """
        Save the current wn as input file with the peak demand for each node
        """
        for name in self.nodes_w_demand:
            if name not in self.wdn_nodes:
                continue
            if name == "1555":
                print("Found a reservoir")
            demand = 0
            pattern = np.zeros(24)
            for building_id in self.wdn_nodes[name]:
                building = self.buildings[building_id]
                demand += building.base_demand
                pattern += building.demand_pattern

            pattern /= len(self.wdn_nodes[name])
            node = self.wn.get_node(name)

            """ NEED TO CONVERT THE LPS FROM THE BUILDINGS TO CMS WHICH IS
            THE INPUT FOR WNTR """
            node.demand_timeseries_list[0].base_value = demand * np.max(pattern) / 1000

        wntr.network.write_inpfile(
            self.wn, city + ".inp", units=self.wn.options.hydraulic.inpfile_units
        )

    def init_hydraulic(self, virtual=True):
        """Initialize the hydraulic information collectors"""
        # add wfh patterns
        for i in range(dt.wfh_patterns.shape[1]):
            self.wn.add_pattern("wk" + str(i + 1), dt.wfh_patterns[:, i])
        # these are nodes with demands (there are also nodes without demand):
        # self.nodes_w_demand = [
        #     node for node in self.grid.G.nodes
        #     if hasattr(self.wn.get_node(node), 'demand_timeseries_list')
        # ]

        # self.agent_matrix = np.zeros(
        #     (len(self.buildings), self.days * 24), dtype=np.int64
        # )

        if self.hyd_sim == "eos":
            self.daily_demand = np.empty((24, len(self.nodes_w_demand)))
            self.demand_matrix = pd.DataFrame(
                0, index=np.arange(0, 86400 * self.days, 3600), columns=self.G.nodes
            )
            self.pressure_matrix = pd.DataFrame(
                0, index=np.arange(0, 86400 * self.days, 3600), columns=self.G.nodes
            )
            self.age_matrix = pd.DataFrame(
                0, index=np.arange(0, 86400 * self.days, 3600), columns=self.G.nodes
            )
            self.flow_matrix = pd.DataFrame(
                0,
                index=np.arange(0, 86400 * self.days, 3600),
                columns=[name for name, link in self.wn.links()],
            )
        elif self.hyd_sim == "hourly":
            self.current_demand = pd.Series(0, index=self.G.nodes)
        elif self.hyd_sim == "monthly":
            # self.daily_demand = np.empty((24, len(self.nodes_w_demand)))
            self.current_age = None

        # initialization methods
        self.base_demand_list()

        # set age with previous steady state values
        self.set_age()

        # set epanet options
        if self.hyd_sim in ["hourly", "monthly"]:
            if self.verbose > 0:
                print(
                    "Set the pattern and hydraulic timestep values and quality parameter"
                )
            self.wn.options.time.pattern_timestep = 3600
            self.wn.options.time.hydraulic_timestep = 3600
            # self.wn.options.time.quality_timestep = 900
            self.wn.options.quality.parameter = "AGE"

            if self.verbose > 0:
                print(self.wn.options.quality.parameter)

                print("Initialize the EPANET simulator")
            self.sim = EpanetSimulator_Stepwise(
                self.wn, file_prefix="temp" + str(self.id)
            )
            self.sim.initialize(file_prefix="temp" + str(self.id))
            # self.current_demand = self.sim._results.node['demand']

        # set the base demand based on the buildings at each node
        if not virtual:
            if self.verbose > 0:
                print("Setting the base demand values based on each building.....")
            for name in self.nodes_w_demand:
                if name not in self.wdn_nodes:
                    continue
                demand = 0
                for building_id in self.wdn_nodes[name]:
                    building = self.buildings[building_id]
                    demand += building.base_demand

                node = self.wn.get_node(name)
                """ NEED TO CONVERT THE LPS FROM THE BUILDINGS TO CMS WHICH IS
                THE INPUT FOR WNTR """
                node.demand_timeseries_list[0].base_value = demand / 1000

            wntr.network.write_inpfile(
                self.wn, "init_wn.inp", units=self.wn.options.hydraulic.inpfile_units
            )

            if self.verbose > 0:
                print(self.wn.options.hydraulic.__dict__)

        # water age slope to determine the warmup period end
        self.water_age_slope = 1
