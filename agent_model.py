from mesa import Agent

import random

class ConsumerAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.timestep = 0
        self.home_node = None
        self.work_node = None
        self.work_type = None
        self.demand =  0
        self.base_demand = 0
        self.information = 0
        self.informed_by = None
        self.compliance = 0
        self.informed_count_u = 0
        self.informed_count_p_f = 0
        self.age = 0
        self.covid = None
        self.exp_time = 0
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
        self.housemates = list()

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

    def change_time(self):
        self.timestep += 1
        return self.timestep
        #print(self.timestep)

    def step(self):
        # self.complying()
        # self.communcation()
        self.change_time()
