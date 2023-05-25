import numpy as np
import pandas as pd
from parameters import agent_pars
from utils import choose, sample, node_list
from base import BasePop
from copy import deepcopy as dc


class Population(BasePop):
    '''
    Class for holding dictionary of lists that contains the agent parameters.
    '''

    def __init__(self, sim, pars):
        '''
        Set the agent parameters. Each dictionary entry is a numpy list of
        the agent attribute given by the key.
        '''

        self.pars = pars
        self.model = sim
        self.nodes = dict()
        # allow for addition of keys to pars
        self._lock = False

        for key in agent_pars:
            if key == 'uid':
                self[key] = np.arange(pars['pop_size'], dtype=np.int32)
            elif key == 'home_node' or \
                 key == 'work_node':
                self[key] = np.full(pars['pop_size'], np.nan, dtype="<U6")
            elif key == 'housemates':
                self[key] = np.full(pars['pop_size'], np.nan)
            elif 'time' in key or \
                 key == 'covid' or \
                 key == 'symp_status' or \
                 key == 'work_ind' or \
                 key == 'work_nav':
                self[key] = np.full(pars['pop_size'], 0, dtype=np.int32)
            else:
                self[key] = np.full(pars['pop_size'], np.nan, dtype=np.float32)

        ''' Add to object dict the nodes as keys and list of bools
        for values representing the location of each agent '''
        for key in self.model.all_nodes:
            self[key] = np.zeros(self.pars['pop_size'])

        ''' Initialize the cafe nodes that agents can travel to '''
        self['cafe_nodes_nam'] = node_list(self.model.nodes_capacity,
                                           self.model.cafe_nodes)
        self['cafe_nodes_bin'] = np.ones(len(self['cafe_nodes_nam']),
                                         dtype=np.int32)
        ''' Initialize the com nodes that agents can travel to '''
        self['com_nodes_nam'] = node_list(self.model.nodes_capacity,
                                          self.model.com_nodes)
        self['com_nodes_bin'] = np.ones(len(self['com_nodes_nam']),
                                        dtype=np.int32)

        self.set_covid_attrs()
        self.set_bbn_attrs()
        self.set_move_attrs()

    def set_covid_attrs(self):
        '''
        Set the COVID related parameters, e.g., exp_time, inf_time, etc.
        Parameters have already been initialized, and need to set.
        '''

        ''' Set the e2i, i2s, s2sev, sev2c, s2d. These are personal
        parameters that are compared to the time each agent is  '''
        self['e2i'] = sample(dist='lognormal', par1=4.5, par2=1.5,
                             size=self.pars['pop_size'])
        self['i2s'] = sample(dist='lognormal', par1=1.1, par2=0.9,
                             size=self.pars['pop_size'])
        self['s2sev'] = sample(dist='lognormal', par1=6.6, par2=4.9,
                               size=self.pars['pop_size'])
        self['sev2c'] = sample(dist='lognormal', par1=1.5, par2=2.0,
                               size=self.pars['pop_size'])
        self['c2d'] = sample(dist='lognormal', par1=10.7, par2=4.8,
                             size=self.pars['pop_size'])

        ''' Set the recovery time values for each agent and state  '''
        self['asym_rec'] = sample(dist='lognormal', par1=8.0, par2=2.0,
                                  size=self.pars['pop_size'])
        self['mild_rec'] = sample(dist='lognormal', par1=8.0, par2=2.0,
                                  size=self.pars['pop_size'])
        self['sev_rec'] = sample(dist='lognormal', par1=18.1, par2=6.3,
                                 size=self.pars['pop_size'])
        self['crit_rec'] = sample(dist='lognormal', par1=18.1, par2=6.3,
                                  size=self.pars['pop_size'])

        ''' Set the initial number of infectious '''
        inds = choose(self.pars['pop_size'], self.pars['int_infectious'])
        self['covid'][inds] = 2

    def set_bbn_attrs(self):
        '''
        Set the BBN parameters for each agent.
        '''

        all_bbn = pd.read_csv(r'Input Files/all_bbn_data.csv')
        bbn_list = all_bbn.columns.to_numpy()  # list of bbn parameter names
        # create dictionary with each bbn param name as key and a numeric
        # index as the value. For looking up bbn params in the numpy array.
        self['bbn_dict'] = dict((key, i) for i, key in enumerate(bbn_list))

        # need to convert pandas dataframe to numpy to store in pop dictionary
        all_bbn = all_bbn.to_numpy()
        inds = choose(len(all_bbn), self.pars['pop_size'])
        self['bbn_params'] = all_bbn[inds]

    def set_move_attrs(self):
        '''
        Set the location related attributes for each agent.
        '''

        ''' Set residential nodes by making a list of available res nodes, then
        picking indices '''
        res_nodes = node_list(self.model.nodes_capacity, self.model.res_nodes)
        # print(type(res_nodes))
        res_inds = choose(len(res_nodes), self.pars['pop_size'])
        self['home_node'] = res_nodes[res_inds]

        ''' Set industrial work nodes by making a list of available works nodes,
        then picking indices '''
        ind_node_list = self.model.ind_nodes + self.model.ind_nodes
        nav_node_list = self.model.nav_nodes + self.model.nav_nodes
        work_nodes = node_list(self.model.nodes_capacity, ind_node_list + nav_node_list)
        work_agents = (max(self.model.ind_dist) + max(self.model.nav_dist)) * 2
        work_inds = choose(len(work_nodes), work_agents)
        ag_inds = choose(self.pars['pop_size'], work_agents)
        self['work_node'][ag_inds] = work_nodes[work_inds]

        ''' Set the work type for each agent that has an industrial or navy
        work node '''
        inds = self.count_node('work_node', self.model.ind_nodes)
        self['work_ind'][inds] = 1
        if self.model['city'] == 'mesopolis':
            inds = self.count_node('work_node', self.model.nav_nodes)
            self['work_nav'][inds] = 1

        ''' Set all agent's current node to their home node '''
        for i, node in enumerate(self['home_node']):
            self[node][i] = 1

        ''' Set households '''
        self.assign_mates()

    def assign_mates(self):
        ''' Assign housemates based on the number of agents at the given
        node '''
        for node in self.model.res_nodes:
            # inds of the agents with the current node as home node
            # print(node)
            # print(self['home_node'])
            inds = np.where(self['home_node'] == node)[0]
            if len(inds) > 6:
                while len(inds) > 6:
                    # pick a household size between 1 and 6.
                    house_size = np.random.randint(1, 6, size=1)[0]
                    # choose the agent inds that will be in that house
                    # these are the indices of the original inds var
                    curr_house_inds = choose(len(inds), house_size)
                    # assign those inds to those agents
                    self['housemates'][inds[curr_house_inds]] = inds[curr_house_inds]
                    inds = np.delete(inds, curr_house_inds, axis=0)
            else:
                # if the household already has 6 or less agents, assign those
                # indexes to each agent.
                self['housemates'][inds] = inds

    def move_agents(self, ind2res=None, res2ind=None,
                    caf2res=None, res2caf=None,
                    com2res=None, res2com=None,
                    nav2res=None, res2nav=None):
        '''
        Main method to move agents around the network.
        '''

        ''' First move agents to and from industrial nodes '''
        ''' Need to remove agents from the agents2ind if they are working
        from home '''
        if ind2res is not None:
            ''' Move agents from industrial to residential '''
            agents2res, node_dict = self.count_node(self.model.ind_nodes)
            inds = choose(len(agents2res), ind2res)
            homes = self['home_node'][agents2res[inds]]
            for home in homes:
                self[home][agents2res[inds]] = 1
            for node, inds in node_dict.items():
                
            # self['curr_node'][agents2res[inds]] = self['home_node'][agents2res[inds]]

        if res2ind is not None:
            agents2ind = self.count_node_if('curr_node', self.model.res_nodes,
                                            'work_ind', 1)
            print(len(agents2ind))
            inds = choose(len(agents2ind), res2ind)
            self['curr_node'][agents2ind[inds]] = self['work_node'][agents2ind[inds]]

        ''' Next move agents to and from cafe nodes '''
        if caf2res is not None:
            agents2res = self.count_node('curr_node', self.model.cafe_nodes)
            inds = choose(len(agents2res), caf2res)
            homes = self['home_node'][agents2res[inds]]
            for home in homes:
                self[home][agents2res[inds]] = 1

            ''' Update the cafe_nodes_bin list to ensure nodes are now open '''
            nodes = self['curr_node'][agents2res[inds]]
            nodes_u = self.node_in_cap(self['cafe_nodes_nam'], nodes)
            self['cafe_nodes_bin'][nodes_u] = 1
            self['curr_node'][agents2res[inds]] = self['home_node'][agents2res[inds]]

        if res2caf is not None:
            # first find the agents that are at home that could move
            agents2caf = self.count_node('curr_node', self.model.res_nodes)
            # next find the available cafe node spots
            nodes = self.true('cafe_nodes_bin')
            # choose the agents that will move
            inds_ag = choose(len(agents2caf), res2caf)
            # choose the nodes they will go to
            inds_caf = choose(len(nodes), res2caf)
            # set their nodes
            self['curr_node'][agents2caf[inds_ag]] = self['cafe_nodes_nam'][nodes[inds_caf]]
            # finally set nodes in cafe_nodes_bin to 0
            self['cafe_nodes_bin'][nodes[inds_caf]] = 0

        if com2res is not None:
            ''' Next move agents to and from com nodes '''
            agents2res = self.count_node('curr_node', self.model.com_nodes)
            inds = choose(len(agents2res), com2res)

            ''' Update the com_nodes_bin list to ensure nodes are now open '''
            nodes = self['curr_node'][agents2res[inds]]
            nodes_u = self.node_in_cap(self['com_nodes_nam'], nodes)
            self['com_nodes_bin'][nodes_u] = 1
            self['curr_node'][agents2res[inds]] = self['home_node'][agents2res[inds]]

        if res2com is not None:
            # first find the agents that are at home that could move
            agents2com = self.count_node('curr_node', self.model.res_nodes)
            # next find the available cafe node spots
            nodes = self.true('com_nodes_bin')
            # choose the agents that will move
            inds_ag = choose(len(agents2com), res2com)
            # choose the nodes they will go to
            inds_com = choose(len(nodes), res2com)
            # set their nodes
            self['curr_node'][agents2com[inds_ag]] = self['com_nodes_nam'][nodes[inds_com]]
            # finally set nodes in cafe_nodes_bin to 0
            self['com_nodes_bin'][nodes[inds_com]] = 0

        ''' If navy nodes exist, move similarly to industrial nodes '''
        if nav2res is not None:
            agents2res = self.count_node('curr_node',
                                         self.model.nav_nodes)
            inds = choose(len(agents2res), nav2res)
            self['curr_node'][agents2res[inds]] = self['home_node'][agents2res[inds]]

        if res2nav is not None:
            agents2nav = self.count_node_if('curr_node', self.model.res_nodes,
                                            'work_nav', 1)
            inds = choose(len(agents2nav), res2nav)
            self['curr_node'][agents2nav[inds]] = self['work_nodes'][agents2nav[inds]]

    def infect(self, inds):
        '''
        Infect people and determine their eventual outcomes.

            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds     (array): array of people to infect

        Returns:
            count (int): number of people infected
        '''

        if len(inds) == 0:
            return 0

        # Remove duplicates
        inds, unique = np.unique(inds, return_index=True)
        if source is not None:
            source = source[unique]

        # Keep only susceptibles
        keep = self.susceptible[inds] # Unique indices in inds and source that are also susceptible
        inds = inds[keep]
        if source is not None:
            source = source[keep]

        if self.pars['use_waning']:
            cvi.check_immunity(self, variant, sus=False, inds=inds)

        # Deal with variant parameters
        variant_keys = ['rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob']
        infect_pars = {k:self.pars[k] for k in variant_keys}
        variant_label = self.pars['variant_map'][variant]
        if variant:
            for k in variant_keys:
                infect_pars[k] *= self.pars['variant_pars'][variant_label][k]

        n_infections = len(inds)
        durpars      = self.pars['dur']

        # Update states, variant info, and flows
        self.susceptible[inds]    = False
        self.naive[inds]          = False
        self.recovered[inds]      = False
        self.diagnosed[inds]      = False
        self.exposed[inds]        = True
        self.n_infections[inds]  += 1
        self.exposed_variant[inds] = variant
        self.exposed_by_variant[variant, inds] = True
        self.flows['new_infections']   += len(inds)
        self.flows['new_reinfections'] += len(cvu.defined(self.date_recovered[inds])) # Record reinfections
        self.flows_variant['new_infections_by_variant'][variant] += len(inds)

        # Record transmissions
        for i, target in enumerate(inds):
            entry = dict(source=source[i] if source is not None else None, target=target, date=self.t, layer=layer, variant=variant_label)
            self.infection_log.append(entry)

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds] = cvu.sample(**durpars['exp2inf'], size=n_infections)
        self.date_exposed[inds]   = self.t
        self.date_infectious[inds] = self.dur_exp2inf[inds] + self.t

        # Reset all other dates
        for key in ['date_symptomatic', 'date_severe', 'date_critical', 'date_diagnosed', 'date_recovered']:
            self[key][inds] = np.nan

        # Use prognosis probabilities to determine what happens to them
        symp_probs = infect_pars['rel_symp_prob']*self.symp_prob[inds]*(1-self.symp_imm[variant, inds]) # Calculate their actual probability of being symptomatic
        is_symp = cvu.binomial_arr(symp_probs) # Determine if they develop symptoms
        symp_inds = inds[is_symp]
        asymp_inds = inds[~is_symp] # Asymptomatic
        self.flows_variant['new_symptomatic_by_variant'][variant] += len(symp_inds)

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        dur_asym2rec = cvu.sample(**durpars['asym2rec'], size=len(asymp_inds))
        self.date_recovered[asymp_inds] = self.date_infectious[asymp_inds] + dur_asym2rec  # Date they recover
        self.dur_disease[asymp_inds] = self.dur_exp2inf[asymp_inds] + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        n_symp_inds = len(symp_inds)
        self.dur_inf2sym[symp_inds] = cvu.sample(**durpars['inf2sym'], size=n_symp_inds) # Store how long this person took to develop symptoms
        self.date_symptomatic[symp_inds] = self.date_infectious[symp_inds] + self.dur_inf2sym[symp_inds] # Date they become symptomatic
        sev_probs = infect_pars['rel_severe_prob'] * self.severe_prob[symp_inds]*(1-self.sev_imm[variant, symp_inds]) # Probability of these people being severe
        is_sev = cvu.binomial_arr(sev_probs) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe
        self.flows_variant['new_severe_by_variant'][variant] += len(sev_inds)

        # CASE 2.1: Mild symptoms, no hospitalization required and no probability of death
        dur_mild2rec = cvu.sample(**durpars['mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**durpars['sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        crit_probs = infect_pars['rel_crit_prob'] * self.crit_prob[sev_inds] * (self.pars['no_hosp_factor'] if hosp_max else 1.) # Probability of these people becoming critical - higher if no beds available
        is_crit = cvu.binomial_arr(crit_probs)  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = cvu.sample(**durpars['sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**durpars['sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        death_probs = infect_pars['rel_death_prob'] * self.death_prob[crit_inds] * (self.pars['no_icu_factor'] if icu_max else 1.)# Probability they'll die
        is_dead = cvu.binomial_arr(death_probs)  # Death outcome
        dead_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # CASE 2.2.2.1: Did not die
        dur_crit2rec = cvu.sample(**durpars['crit2rec'], size=len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = self.dur_exp2inf[alive_inds] + self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec  # Store how long this person had COVID-19

        # CASE 2.2.2.2: Did die
        dur_crit2die = cvu.sample(**durpars['crit2die'], size=len(dead_inds))
        self.date_dead[dead_inds] = self.date_critical[dead_inds] + dur_crit2die # Date of death
        self.dur_disease[dead_inds] = self.dur_exp2inf[dead_inds] + self.dur_inf2sym[dead_inds] + self.dur_sym2sev[dead_inds] + self.dur_sev2crit[dead_inds] + dur_crit2die   # Store how long this person had COVID-19
        self.date_recovered[dead_inds] = np.nan # If they did die, remove them from recovered

        # Handle immunity aspects
        if self.pars['use_waning']:
            self.prior_symptoms[asymp_inds] = self.pars['rel_imm_symp']['asymp']
            self.prior_symptoms[mild_inds] = self.pars['rel_imm_symp']['mild']
            self.prior_symptoms[sev_inds] = self.pars['rel_imm_symp']['severe']
            cvi.update_peak_nab(self, inds, nab_pars=self.pars, natural=True)

        return n_infections # For incrementing counters
