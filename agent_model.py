from mesa import Agent
import math
from copy import deepcopy as dcp
import bnlearn as bn
import data as dt


class ConsumerAgent(Agent):

    def __init__(self, unique_id, household, model):
        super().__init__(unique_id, model)
        self.household = household
        self.timestep = 0
        self.home_node = None
        self.work_node = None
        self.work_type = None
        self.demand = 0
        self.base_demand = 0
        self.information = 0
        self.informed_by = None
        self.compliance = 0
        self.informed_count_u = 0
        self.informed_count_p_f = 0
        self.age = 0
        self.covid = None
        # counters for different covid stages
        self.exposed_time = 0
        self.infectious_time = 0
        self.symp_time = 0
        self.sev_time = 0
        self.crit_time = 0
        self.symptomatic = None  # 0 is asymptomatic, 1 symptomatic
        self.inf_severity = 0    # 0: asymptomatic, 1: mild, 2: severe, 3: critical
        self.adj_covid_change = 0     # 0: no change in housemates having covid, 1: recently a housemate became infectious
        self.wfh = 0  # working from home decision status
        self.no_dine = 0  # not dining out
        self.less_groceries = 0  # shopping for groceries less
        self.ppe = 0  # wearing ppe
        self.can_wfh = True   # bool based on workplace decision to allow work from home
        self.agent_params = {}  # BBN parameters for predicting work from home
        # thresholds for covid stage. Represent the time an agent should spend
        # in each stage.
        # time after first exposure to infectious
        self.exp_time = 24 * self.sample_ln(self.model.e2i[0], self.model.e2i[1])

        # time between first infectious and symptomatic
        self.inf_time = 24 * self.sample_ln(self.model.i2s[0], self.model.i2s[1])

        # time between symptomatic and severe
        self.sev_time = 24 * self.sample_ln(self.model.s2sev[0], self.model.s2sev[1])

        # time between severe and critical
        self.crit_time = 24 * self.sample_ln(self.model.sev2c[0], self.model.sev2c[1])

        # recovery times for each covid stage
        self.asymp_time = 24 * self.sample_ln(
            self.model.recTimeAsym[0],
            self.model.recTimeAsym[1]
        )
        self.mild_time = 24 * self.sample_ln(
            self.model.recTimeMild[0],
            self.model.recTimeMild[1]
        )
        self.sevRec_time = 24 * self.sample_ln(
            self.model.recTimeSev[0],
            self.model.recTimeSev[1]
        )
        self.critRec_time = 24 * self.sample_ln(
            self.model.recTimeC[0],
            self.model.recTimeC[1]
        )
        self.death_time = 24 * self.sample_ln(
           self.model.c2d[0],
           self.model.c2d[1]
        )

    def sample_ln(self, mux, sigmax):
        '''
        Return a log-normal sample with the given mu and sigma.
        '''

        mu = math.log(mux**2 / math.sqrt(sigmax**2 + mux**2))
        sigma = math.sqrt(math.log(sigmax**2/mux**2 + 1))
    
        return self.random.lognormvariate(mu, sigma)

    def complying(self):
        if self.information == 1 and self.informed_count_u < 2 and self.informed_count_p_f < 3:
            if self.informed_by == 'utility' and self.random.random() < 0.5:
                self.compliance = 1
            elif self.informed_by == 'household' and self.random.random() < 0.3:
                self.compliance = 1
            elif self.informed_by == 'peer_swn' and self.random.random() < 0.3:
                self.compliance = 1
        else:
            pass

    def communcation(self):
        household_members = [a for i, a in enumerate(self.model.schedule.agents)
                             if a.home_node == self.home_node]
        if self.compliance == 1:
            f = 0
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

    def check_infectious(self):
        """
        Function to check if agent has been exposed for sufficient time to be
        infectious. Agent is not symptomatic yet, but can still transmit disease.

        Function is typically run from loop over all agents at each DAY step.
        """
        if self.exposed_time >= self.exp_time:
            self.covid = 'infectious'
            self.model.cumm_infectious += 1
            self.exposed_time = 0
            self.agent_params["COVIDexp"] = 1
        else:
            pass

    def check_symptomatic(self):
        """
        Function to check if agent has been infectious for sufficient time to be
        either symptomatic or asymptomatic. Once asymptomatic, there is a set time
        until recovered, otherwise, moves to mild, severe, or critical depending
        on age.

        Function is typically run from loop over all agents at each DAY step.
        """
        if self.symptomatic is None and self.infectious_time >= self.inf_time:
            if self.random.random() < dt.susDict[self.age][0]:
                self.symptomatic = 1
                self.inf_severity = 1
            else:
                self.symptomatic = 0
        else:
            pass

    def check_severity(self):
        """
        Function to check the agents infection severity. Severity is based on the
        time the agent has been symptomatic and their age.

        Function is typically run from loop over all agents at each DAY step.
        """
        sev_prob = dt.susDict[self.age][1]
        crit_prob = dt.susDict[self.age][2]
        if (self.inf_severity == 1 and
                self.symp_time >= self.sev_time and
                self.random.random() < sev_prob):
            self.inf_severity = 2
        elif (self.inf_severity == 2 and
                self.sev_time >= self.crit_time and
                self.random.random() < crit_prob):
            self.inf_severity = 3
        else:
            pass

    def check_recovered(self):
        """
        Function to check if agent has been infectious for sufficient time to be
        recorvered. Depends on whether the agent was asymptomatic or symptomatic.

        Function is typically run from loop over all agents at each DAY step.
        """
        if self.symptomatic == 0 and self.infectious_time >= self.asymp_time:
            self.covid = 'recovered'
            self.infectious_time = 0
            self.symptomatic = None
        else:
            pass

        if self.inf_severity == 1 and self.symp_time >= self.mild_time:
            self.covid = 'recovered'
            self.infectious_time = 0
            self.symptomatic = None
        elif self.inf_severity == 2 and self.sev_time >= self.sevRec_time:
            self.covid = 'recovered'
            self.infectious_time = 0
            self.symptomatic = None
        elif self.inf_severity == 3 and self.crit_time >= self.critRec_time:
            self.covid = 'recovered'
            self.infectious_time = 0
            self.symptomatic = None
        else:
            pass

    def check_death(self):
        """
        Function to check if agent has been critical for sufficient time to be
        potentially dead. Depends on agents age. Removes agent from grid and
        sets agents position to None.

        Function is typically run from loop over all agents at each DAY step.
        """
        death_prob = dt.susDict[self.age][3]
        if (self.inf_severity == 3 and
                self.crit_time >= self.death_time and
                self.random.random() < death_prob):
            self.covid = 'dead'
            self.symptomatic = None
            self.model.grid.remove_agent(self)
        else:
            pass

    def predict_wfh(self):
        self.adj_covid_change = 0
        evidence_agent = dcp(self.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.model.wfh_nodes):
            if item != 'work_from_home':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.model.wfh_dag,
                                 variables=['work_from_home'],
                                 evidence=evidence,
                                 verbose=0)
        if self.random.random() < query.df['p'][1]:
            self.wfh = 1

    def predict_dine_less(self):
        self.adj_covid_change = 0
        evidence_agent = dcp(self.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.model.dine_nodes):
            if item != 'dine_out_less':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.model.dine_less_dag,
                                 variables=['dine_out_less'],
                                 evidence=evidence,
                                 verbose=0)
        if self.random.random() < query.df['p'][1]:
            self.no_dine = 1

    def predict_grocery(self):
        self.adj_covid_change = 0
        evidence_agent = dcp(self.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.model.grocery_nodes):
            if item != 'shop_groceries_less':
                evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.model.grocery_dag,
                                 variables=['shop_groceries_less'],
                                 evidence=evidence,
                                 verbose=0)
        if self.random.random() < query.df['p'][1]:
            self.less_groceries = 1

    def predict_ppe(self):
        self.adj_covid_change = 0
        evidence_agent = dcp(self.agent_params)
        evidence_agent['COVIDeffect_4'] = math.floor(evidence_agent['COVIDeffect_4'])
        evidence = dict()
        for i, item in enumerate(self.model.ppe_nodes):
            if item != 'mask':
                if evidence_agent[item] < 10 and evidence_agent[item] >= 0:
                    evidence[item] = evidence_agent[item]

        query = bn.inference.fit(self.model.ppe_dag,
                                 variables=['mask'],
                                 evidence=evidence,
                                 verbose=0)
        if self.random.random() < query.df['p'][1]:
            self.ppe = 1

    def change_house_adj(self):
        ''' Function to check whether agents in a given agents node have become
        infected with COVID '''
        agents_in_house = self.household.agent_ids
        agents_friends = [n for n in self.model.swn.neighbors(self.unique_id)]
        agents_in_network = agents_in_house + agents_friends
        agents_in_network.remove(self.unique_id)
        for id in agents_in_network:
            adj_agent = self.model.schedule._agents[id]
            adj_agent.adj_covid_change = 1
            if adj_agent.agent_params["COVIDeffect_4"] < 6:
                adj_agent.agent_params["COVIDeffect_4"] += 1
            else:
                pass

    def change_time(self):
        self.timestep += 1
        return self.timestep

    def step(self):
        # self.complying()
        # self.communcation()
        self.change_time()


class Household:
    '''
    Container for households. Contains a collection of agent objects

    Parameters
    ----------
    start_id : int
        starting index for agent ids

    end_id : int
        ending index for agent ids

    node : string
        string corresponding to the node in the water network that is this
        households home node

    node_dist : float
        this nodes relative distance to the nearest industrial node

    twa_mods : list
        list of twa modulators for [drink, cook, hygiene]

    model : ConsumerModel
        model object where agents are added
    '''

    def __init__(self, start_id, end_id, node, node_dist, twa_mods, model):
        self.agent_ids = list()  # list of agent that are in the household
        self.agent_obs = list()  # list of agent objects that are in the household
        self.tap = ['drink', 'cook', 'hygiene']  # the actions using tap water
        self.bottle = []  # actions using bottled water
        self.tap_demand = 0  # the tap water demand
        self.bottle_demand = 0  # the bottled water demand

        # https://www.cityofclintonnc.com/DocumentCenter/View/759/FY23-24-fee-schedule?bidId=
        self.base_rate_water = 15.55  # dollars per month; this is from the city of clinton, nc
        self.cons_rate_water = 0.000844022  # dollars per L; clinton, nc
        self.base_rate_sewer = 16.21  # dollars per month; clinton, nc
        self.cons_rate_sewer = 0.00081577  # dollars per L; clinton, nc
        self.bottle_cost_pl = 0.325  # dollars per L
        self.min_wage = 7.25  # minimum wage for NC

        self.tap_cost = 0  # the total cost of tap water
        self.bottle_cost = 0  # the total cost of bottled water
        self.reduction = 0  # the demand reduction for the last 168 hours
        self.change = 1
        self.model = model
        self.node = node
        self.ind_dist = node_dist

        self.agent_hours = 0  # the total number of hours agents spend at the home node

        for i in range(start_id, end_id):
            a = ConsumerAgent(i, self, model)
            model.schedule.add(a)
            if model.work_agents != 0:
                a.work_node = model.random.choice(model.work_loc_list)
                a.home_node = node
                model.work_loc_list.remove(a.work_node)
                if a.work_node in model.nav_nodes:
                    a.work_type = 'navy'
                elif a.work_node in model.ind_nodes:
                    a.work_type = 'industrial'
                model.work_agents -= 1
            else:
                a.home_node = node

            if a.work_node in model.total_no_wfh:
                a.can_wfh = False
            model.grid.place_agent(a, a.home_node)
            self.agent_obs.append(a)
            self.agent_ids.append(a.unique_id)

        # get the base demand for this node
        wn_node = model.wn.get_node(node)
        self.base_demand = wn_node.demand_timeseries_list[0].base_value

        # pick an income for the household based on the relative distance
        # to the nearest industrial node

        # the scaling factor represents the median income at the distance this
        # node is away from industrial
        if self.model.dist_income:
            scaling_factor = (28250.19550039 + 28711.81795579 * node_dist) / 38880
        else:
            scaling_factor = 1
        # pick an income from the gamma distribution trained with clinton
        # income ranges
        mean = 61628.09180717512
        var = 5671494492.817419

        a = mean**2/var
        b = var/mean
        # variance in income/dist data 14569.890867054484
        self.income = model.random.gammavariate(a, b) * scaling_factor
        # self.income = model.random.gammavariate(
        #     dt.size_income[int(len(self.agent_ids))][0],
        #     dt.size_income[int(len(self.agent_ids))][1]
        # )

        # if the income is below minimum wage, increase to minimum wage
        # 2080 is the number of hours per year worked if working 40hr/wk
        # if self.income < self.win_wage * 2080:
        #     self.income = self.min_wage * 2080

        # set the income level: low, medium, high
        # low income is set using HUD thresholds by household size
        # https://www.huduser.gov/portal/datasets/il.html
        high_income = 150000
        if self.income < dt.ex_low_income[int(len(self.agent_ids))]:
            self.income_level = 0
        if (self.income < dt.low_income[int(len(self.agent_ids))]
           and self.income > dt.ex_low_income[int(len(self.agent_ids))]):
            self.income_level = 1
        if (self.income >= dt.low_income[int(len(self.agent_ids))]
           and self.income < high_income):
            self.income_level = 2
        if self.income >= high_income:
            self.income_level = 3

        # pick water age thresholds for TWA behaviors
        self.twa_thresholds = {
            'drink':   model.random.betavariate(3, 1) * twa_mods[0] + 24,
            'cook':    model.random.betavariate(3, 1) * twa_mods[1] + 24,
            'hygiene': model.random.betavariate(3, 1) * twa_mods[2] + 24
        }

        # 
        self.demand_reduction = {
            'drink':   5.3,
            'cook':    10.6,
            'hygiene': 10.6
        }

    def count_agents(self):
        '''
        Count the number of agents that are currently at this household
        '''
        for agent in self.agent_obs:
            if agent.pos == self.node:
                self.agent_hours += 1

    def update_household(self, age):
        '''
        Perform updating methods. Update behaviors, calculate demand
        and calculate bottled water use.
        '''
        # print(age)

        # update the behaviors that this household is adopting based on the
        # water age
        self.update_behaviors(age)

        # calculate the demand for tap and bottled water for that last 30 days
        # this uses the reduction value calcualted at the last call to
        # update_household
        # self.calc_demand()

        # calculate the cost of the water for the last 30 days
        self.calc_cost()

        # calculate the demand change needed based on the behaviors adopted
        # at this call to update_household
        self.calc_demand_reduction()

    def update_behaviors(self, age):
        '''
        Update the behavior lists tap and bottle based on the water age

        ** NOTE **
        None of the behaviors are re-added to the self.tap
        variable. This means that once the water age at the node
        exceeds the threshold that household will always adopt that
        behavior. I think that makes sense as once we perceive something
        as unsafe we are unlikely to go back.
        '''
        if age > self.twa_thresholds['hygiene'] and 'hygiene' in self.tap:
            self.tap.remove('hygiene')
            self.bottle.append('hygiene')
        if age > self.twa_thresholds['drink'] and 'drink' in self.tap:
            self.tap.remove('drink')
            self.bottle.append('drink')
        if age > self.twa_thresholds['cook'] and 'cook' in self.tap:
            self.tap.remove('cook')
            self.bottle.append('cook')

    def calc_demand(self):
        '''
        Calculate the actual demand for the hydraulic interval
        '''
        # collect the last 30 days of hydraulic data
        timestep = self.model.timestep
        hyd_step = (30 * 24)

        # get the demand pattern from the model and subset for the last
        # hydraulic interval
        # demand_pattern = self.model.wn.get_pattern('node_'+self.node)
        # demand_pattern = dcp(demand_pattern.multipliers[timestep-hyd_step:timestep])

        # need to get the demand of just this household if it is in a
        # multifamily housing node
        # multiplier = len(self.agent_ids) / self.model.nodes_capacity[self.node]

        # Calculate the actual demand for that period
        # total_demand = demand_pattern.sum() * self.base_demand * multiplier

        ''' get the demand from the last 30 days, this is the tap water demand '''
        ''' TODO: Need to figure out multi-family housing '''
        print(self.model.sim._results.node['demand'].loc[:, self.node])
        total_demand = self.model.sim._results.node['demand'].loc[
            timestep-hyd_step:timestep,
            self.node
        ].sum()

        # set the households tap demand and bottled water demand
        # need to convert from gallons to liters
        # self.sum_demand = total_demand * 1000000  # liters
        # self.demand = self.sum_demand - self.reduction  # liters

        ''' this assumes that the current value for self.reduction is the value
        from the last 30 days. This is true because we have not run
        self.calc_demand_change() and is only true if we actually reduce the
        demand by the reduction amount '''
        self.tap_demand = total_demand * 1000000
        self.bottle_demand = self.reduction
        # self.bottled = self.reduction  # liters

        # need to return the percentage change in demand after bottled water
        # to calculate future months demand
        # self.change = self.demand / self.sum_demand

        # if self.reduction != 0:
        #     print(f"Number of agents: {len(self.agent_ids)}")
        #     print(f"Node: {self.node}")
        #     print(f"Base demand: {self.base_demand}")
        #     print(f"Demand pattern: {demand_pattern}")
        #     print(f"Monthly demand: {self.sum_demand}")
        #     print(f"Monthly reduction: {self.reduction}")

    def calc_tap_cost(self, structure='simple'):
        '''
        Helper to calculate cost of tap water
        '''
        if structure == 'simple':
            '''
            Calculate the cost of the tap water use. Any use above
            300 cu. ft. household pays consumption rate (which is per
            100 cu. ft.)
            '''
            cons_threshold = 300 * 28.3168  # 300 cu. ft. to L
            water = (
                self.base_rate_water +
                (self.tap_demand - cons_threshold if self.tap_demand > cons_threshold else 0) *
                self.cons_rate_water
            )

            '''
            Calculate sewer cost. All use is charged a base rate and a per
            100 cu. ft. consumption rate
            '''
            sewer = (
                self.base_rate_sewer +
                self.tap_demand * self.cons_rate_sewer
            )
            self.tap_cost += water + sewer

    def calc_cost(self):
        '''
        Calculate the cost of water for the household. Total cost is the cost
        of tap water plus the cost of bottled water.
        '''
        # calculate cost of tap water
        self.calc_tap_cost()

        # reset the monthly tap demand
        self.tap_demand = 0

        # calculate cost of bottled water
        self.bottle_cost += self.bottle_demand * self.bottle_cost_pl

        # reset the monthly bottled water demand
        self.bottle_demand = 0

        # calculate total cost
        self.cow = self.tap_cost + self.bottle_cost

        # if bottle > 0.0:
        #     print(self.node)
        #     print(tap)
        #     print(self.tap_cost)
        #     print(bottle)
        #     print(self.bottle_cost)
        #     print(self.cow)

    def calc_demand_reduction(self):
        '''
        Calculates the demand change for the hour based on the behaviors
        '''
        # avg_agents = self.agent_hours / 30 / 24
        if self.model.twa_process == 'absolute':
            self.reduction = 0
            if 'cook' not in self.tap:
                ''''
                add the amount of water used for cooking to the reduction

                this value is 11.5 L/c/d multiplied by the average number of agents
                that were at this node for the past month. Then multiply by 30 to
                get the monthly use in L.
                '''
                self.reduction += 11.5 * len(self.agent_ids)  # L/day

            if 'drink' not in self.tap:
                # change -= self.demand_reduction['drink'] / 100
                # one_month_demand = 0
                # for _ in range(30):
                one_day_demand = self.model.random.lognormvariate(
                    0.6273590016655857, math.sqrt(0.13157635778871926)
                )

                # set minimum of 0.25 L/c/d
                if one_day_demand < 0.25:
                    one_day_demand = 0.25

                self.reduction += one_day_demand * len(self.agent_ids)  # L/day

            if 'hygiene' not in self.tap:
                ''' update the reduction value with the amount we expect agents
                to reduce their demand when buying bottled water for hygiene
                purposes '''
                events_per_day = 2
                one_day_demand = self.model.random.triangular(0.25, 1.5, 0.5)

                self.reduction += events_per_day * one_day_demand * len(self.agent_ids)

        elif self.model.twa_process == 'percentage':
            self.change = 1
            if 'hygiene' not in self.tap:
                self.change -= self.demand_reduction['hygiene'] / 100
            if 'cook' not in self.tap:
                self.change -= self.demand_reduction['cook'] / 100
            if 'drink' not in self.tap:
                self.change -= self.demand_reduction['drink'] / 100

        # need to reset the agent_hours each month
        # self.agent_hours = 0

