import numpy as np
import pandas as pd
import parameters as param
import utils as util
from base import BasePop


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

        ''' Set agent parameters including locations/age '''
        for key in param.agent_pars:
            if key == 'uid':
                self[key] = np.arange(pars['pop_size'], dtype=np.int32)
            elif key == 'home_node' or \
                 key == 'work_node':
                self[key] = np.full(pars['pop_size'], np.nan, dtype="<U6")
            elif key == 'housemates':
                self[key] = np.full(pars['pop_size'], np.nan)
            elif key == 'work_ind' or \
                 key == 'work_nav' or \
                 key == 'age':
                self[key] = np.zeros(pars['pop_size'], dtype=np.int32)
            elif key == 'susceptible':
                self[key] = np.ones(pars['pop_size'], dtype=np.int32)
            else:
                self[key] = np.full(pars['pop_size'], np.nan, dtype=np.float32)

        ''' Set COVID-19 state variables '''
        for key in param.state_pars + param.prob_pars:
            if key == 'susceptible':
                self[key] = np.ones(pars['pop_size'], dtype=np.int32)
            else:
                self[key] = np.zeros(pars['pop_size'], dtype=np.int32)

        ''' Set duration and date variables '''
        for key in param.dur_pars + param.date_pars:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=np.float32)

        ''' Add to object dict the nodes as keys and list of bools
        for values representing the location of each agent '''
        for key in self.model.all_nodes:
            self[key] = np.zeros(self.pars['pop_size'], dtype=np.int32)

        ''' Initialize the cafe nodes that agents can travel to '''
        self['cafe_nodes_nam'] = util.node_list(self.model.nodes_capacity,
                                                self.model.cafe_nodes)
        self['cafe_nodes_bin'] = np.ones(len(self['cafe_nodes_nam']),
                                         dtype=np.int32)
        ''' Initialize the com nodes that agents can travel to '''
        self['com_nodes_nam'] = util.node_list(self.model.nodes_capacity,
                                               self.model.com_nodes)
        self['com_nodes_bin'] = np.ones(len(self['com_nodes_nam']),
                                        dtype=np.int32)

        self.set_covid_attrs()
        self.set_bbn_attrs()
        self.set_move_attrs()

    def set_covid_attrs(self):
        '''
        Set the COVID related parameters.
        Parameters have already been initialized and need to set.
        '''
        def find_cutoff(age_cutoffs, age):
            '''
            Find which age bin each person belongs to -- e.g. with standard
            age bins 0, 10, 20, etc., ages [5, 12, 4, 58] would be mapped to
            indices [0, 1, 0, 5]. Age bins are not guaranteed to be uniform
            width, which is why this can't be done as an array operation.
            '''
            return np.nonzero(age_cutoffs <= age)[0][-1]  # Index of the age bin to use

        progs = pars['prognoses'] # Shorten the name
        inds = np.fromiter((find_cutoff(progs['age_cutoffs'], this_age) for this_age in self.age), dtype=cvd.default_int, count=len(self)) # Convert ages to indices
        self.symp_prob[:] = susDict['symp_probs'][inds] # Probability of developing symptoms
        self.severe_prob[:] = progs['severe_probs'][inds]*progs['comorbidities'][inds] # Severe disease probability is modified by comorbidities
        self.crit_prob[:] = progs['crit_probs'][inds] # Probability of developing critical disease
        self.death_prob[:] = progs['death_probs'][inds] # Probability of death
        self.rel_sus[:] = progs['sus_ORs'][inds]  # Default susceptibilities
        self.rel_trans[:] = progs['trans_ORs'][inds] * cvu.sample(**self.pars['beta_dist'], size=len(inds))  # Default transmissibilities, with viral load drawn from a distribution

        ''' Set the initial number of infectious '''
        inds = util.choose(self.pars['pop_size'], self.pars['int_infectious'])
        self['infected'][inds] = 1

        ''' Set age '''
        self['age'] = np.random.choice(param.ages, self.par['pop_size'],
                                       prob=param.age_weights, replace=False)

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
        inds = util.choose(len(all_bbn), self.pars['pop_size'])
        self['bbn_params'] = all_bbn[inds]

    def set_move_attrs(self):
        '''
        Set the location related attributes for each agent.
        '''

        ''' Set residential nodes by making a list of available res nodes, then
        picking indices '''
        res_nodes = util.node_list(self.model.nodes_capacity, self.model.res_nodes)
        res_inds = util.choose(len(res_nodes), self.pars['pop_size'])
        self['home_node'] = res_nodes[res_inds]

        ''' Set industrial work nodes by making a list of available works nodes,
        then picking indices '''
        ind_node_list = self.model.ind_nodes + self.model.ind_nodes
        ind_nodes = util.node_list(self.model.nodes_capacity, ind_node_list)
        ind_agents = (max(self.model.ind_dist)) * 2
        ind_inds = util.choose(len(ind_nodes), ind_agents)
        ag_inds = util.choose(self.pars['pop_size'], len(ind_inds))
        self['work_node'][ag_inds] = ind_nodes[ind_inds]

        ''' Mark the agents that work at industrial nodes '''
        inds = self.defined_str('work_node')
        self['work_ind'][inds] = 1

        if self.pars['city'] == 'mesopolis':
            nav_node_list = self.model.nav_nodes + self.model.nav_nodes
            nav_nodes = util.node_list(self.model.nodes_capacity, nav_node_list)
            nav_agents = (max(self.model.nav_dist)) * 2
            nav_inds = util.choose(len(nav_nodes), nav_agents)
            ag_left = self.undefined('work_node')
            ag_inds = util.choose(len(ag_left), len(nav_inds))
            self['work_node'][ag_inds] = ag_left[nav_inds]

            ''' Mark the agents that work at navy nodes '''
            ''' Does not work. Need a list of indices that are only navy
            nodes. This currently counts all work nodes '''
            inds = self.defined('work_node')
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
            inds = np.where(self['home_node'] == node)[0]
            if len(inds) > 6:
                while len(inds) > 6:
                    # pick a household size between 1 and 6.
                    house_size = np.random.randint(1, 6, size=1)[0]
                    # choose the agent inds that will be in that house
                    # these are the indices of the original inds var
                    curr_house_inds = util.choose(len(inds), house_size)
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
            ''' Move agents from industrial to home homes '''
            agents2res, node_dict = self.count_node(self.model.ind_nodes)
            inds = util.choose(len(agents2res), ind2res)
            homes = self['home_node'][agents2res[inds]]
            uids = self['uid'][agents2res[inds]]
            nodes = np.array([self.find_node(i, self.model.ind_nodes) for i in uids])
            self.set_nodes(homes, nodes, uids, agents2res, ind2res, inds,
                           True)
            self['move_res'][agents2res[inds]] = 1

        ''' Move agents to industrial nodes from residential '''
        if res2ind is not None:
            agents2ind, node_dict = self.count_node_if(self.model.res_nodes,
                                                       'work_ind')
            inds = util.choose(len(agents2ind), res2ind)
            homes = self['home_node'][agents2ind[inds]]
            nodes = self['work_node'][agents2ind[inds]]
            uids = self['uid'][agents2ind[inds]]
            self.set_nodes(homes, nodes, uids, agents2ind, res2ind, inds,
                           False)
            self['move_nonres'][agents2ind[inds]] = 1

        ''' Next move agents to and from cafe nodes '''
        if caf2res is not None:
            ''' Move agents home from cafe nodes '''
            agents2res, node_dict = self.count_node(self.model.cafe_nodes)
            inds = util.choose(len(agents2res), caf2res)
            homes = self['home_node'][agents2res[inds]]
            uids = self['uid'][agents2res[inds]]
            nodes = np.array([self.find_node(i, self.model.cafe_nodes) for i in uids])
            self.set_nodes(homes, nodes, uids, agents2res, caf2res, inds,
                           True, 'caf')
            self['move_res'][agents2res[inds]] = 1

        if res2caf is not None:
            # first find the agents that are at home that could move
            agents2caf, node_dict = self.count_node(self.model.res_nodes)
            # next find the available cafe node spots
            caf_nodes = self.true('cafe_nodes_bin')
            # choose the agents that will move
            inds_ag = util.choose(len(agents2caf), res2caf)
            # choose the nodes they will go to
            inds_caf = util.choose(len(caf_nodes), res2caf)
            # set their nodes
            homes = self['home_node'][agents2caf[inds_ag]]
            nodes = self['cafe_nodes_nam'][caf_nodes[inds_caf]]
            uids = self['uid'][agents2caf[inds_ag]]
            self.set_nodes(homes, nodes, uids, agents2caf, res2caf, inds_ag,
                           False)
            # finally set nodes in cafe_nodes_bin to 0
            inds_caf = self.node_in_cap(self['cafe_nodes_nam'], homes)
            self['cafe_nodes_bin'][inds_caf] = 0
            self['move_nonres'][agents2caf[inds_ag]] = 1

        if com2res is not None:
            ''' Move agents home from commercial nodes '''
            agents2res, node_dict = self.count_node(self.model.com_nodes)
            inds = util.choose(len(agents2res), com2res)
            homes = self['home_node'][agents2res[inds]]
            uids = self['uid'][agents2res[inds]]
            nodes = np.array([self.find_node(i, self.model.com_nodes) for i in uids])
            self.set_nodes(homes, nodes, uids, agents2res, com2res, inds,
                           True, 'com')
            self['move_res'][agents2res[inds]] = 1

        if res2com is not None:
            # first find the agents that are at home that could move
            agents2com, node_dict = self.count_node(self.model.res_nodes)
            # next find the available cafe node spots
            com_nodes = self.true('com_nodes_bin')
            # choose the agents that will move
            inds_ag = util.choose(len(agents2com), res2com)
            # choose the nodes they will go to
            inds_com = util.choose(len(com_nodes), res2com)
            # set their nodes
            homes = self['home_node'][agents2com[inds_ag]]
            nodes = self['com_nodes_nam'][com_nodes[inds_com]]
            uids = self['uid'][agents2com[inds_ag]]
            self.set_nodes(homes, nodes, uids, agents2com, res2com, inds_ag,
                           False)
            # finally set nodes in cafe_nodes_bin to 0
            inds_com = self.node_in_cap(self['com_nodes_nam'], homes)
            self['com_nodes_bin'][inds_com] = 0
            self['move_nonres'][agents2com[inds_ag]] = 1

        ''' If navy nodes exist, move similarly to industrial nodes '''
        if nav2res is not None:
            agents2res, node_dict = self.count_node(self.model.nav_nodes)
            inds = util.choose(len(agents2res), nav2res)
            homes = self['home_node'][agents2res[inds]]
            uids = self['uid'][agents2res[inds]]
            nodes = np.array([self.find_node(i, self.model.nav_nodes) for i in uids])
            self.set_nodes(homes, nodes, uids, agents2res, nav2res, inds,
                           True)
            self['move_res'][agents2res[inds]] = 1

        if res2nav is not None:
            agents2nav, node_dict = self.count_node_if(self.model.res_nodes,
                                                       'work_nav')
            inds = util.choose(len(agents2nav), res2nav)
            homes = self['home_node'][agents2nav[inds]]
            nodes = self['work_node'][agents2nav[inds]]
            uids = self['uid'][agents2nav[inds]]
            self.set_nodes(homes, nodes, uids, agents2nav, res2nav, inds,
                           False)
            self['move_nonres'][agents2nav[inds]] = 1

    def set_nodes(self, homes, nodes, uids, agents2des, num2move, inds, dest,
                  origin=None):
        ''' First decide if agents are moving to residential or to
        non-residential. If agents are moving to residential, dest should
        be True, otherwise, False. '''
        if dest:
            res = 1
            non_res = 0
        else:
            res = 0
            non_res = 1

        ''' Get the agents that are at the specified nodes and change their
        location from the node to their home node '''
        for i, home in enumerate(homes):
            self[home][uids[i]] = res

        ''' Set the agents non_res node '''
        for i, node in enumerate(nodes):
            self[node][uids[i]] = non_res

        ''' Update the com/caf_nodes_bin list to ensure nodes are now open '''
        if origin == 'com':
            nodes_u = self.node_in_cap(self['com_nodes_nam'], nodes)
            self['com_nodes_bin'][nodes_u] = 1
        elif origin == 'cafe':
            nodes_u = self.node_in_cap(self['caf_nodes_nam'], nodes)
            self['caf_nodes_bin'][nodes_u] = 1
        elif origin is None:
            pass

    def reset_move(self):
        ''' Need to reset the move_res and move_nonres every step after
        infections '''
        self['move_res'][:] = 0
        self['move_nonres'][:] = 0

    def contact(self):
        '''
        Simulate contact between agents at the same node. Agents only contact
        if they just moved to that node. Still need to check if they were close
        to an infected agent...
        '''

        self.infect(self.count_node_if(self.model.res_nodes, 'move_res')[0],
                    self.pars['res_inf_rate'])
        self.infect(self.count_node_if(self.model.nonres_nodes, 'move_nonres')[0],
                    self.pars['nonres_inf_rate'])


    def update_states_pre(self, t):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.is_exp = self.true('exposed') # For storing the interim values since used in every subsequent calculation

        # Perform updates
        self.init_flows()
        self.flows['new_infectious']    += self.check_infectious() # For people who are exposed and not infectious, check if they begin being infectious
        self.flows['new_symptomatic']   += self.check_symptomatic()
        self.flows['new_severe']        += self.check_severe()
        self.flows['new_critical']      += self.check_critical()
        self.flows['new_recoveries']    += self.check_recovery()
        new_deaths, new_known_deaths    = self.check_death()
        self.flows['new_deaths']        += new_deaths
        self.flows['new_known_deaths']  += new_known_deaths

        return


    def infect(self, inds, inf_rate):
        '''
        Infect people and determine their eventual outcomes.

            * Every infected person can infect other people, regardless of
              whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild
              vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic
              person recovers
            * Critical cases either recover or die

        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds     (array): array of people to infect
            inf_rate (float): infection rate

        Returns:
            count (int): number of people infected
        '''

        if len(inds) == 0:
            return 0

        # Remove duplicates
        inds = np.unique(inds)

        # check if agent actually had a contact
        inds = util.n_binomial(inf_rate, len(inds))

        # Keep only susceptibles
        keep = self.susceptible[inds]  # Unique indices in inds and source that are also susceptible
        inds = inds[keep]

        n_infections = len(inds)

        # Update states, variant info, and flows
        self.susceptible[inds] = False
        self.recovered[inds] = False
        self.exposed[inds] = True

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds] = util.sample(self.pars['exp2inf'], n_infections)
        self.date_exposed[inds] = self.model.timestep
        self.date_infectious[inds] = self.dur_exp2inf[inds] + self.model.timestep

        # Reset all other dates
        for key in ['date_symptomatic', 'date_severe', 'date_critical', 'date_recovered']:
            self[key][inds] = np.nan

        # Use prognosis probabilities to determine what happens to them
        symp_probs = self.symp_prob[inds] # Calculate their actual probability of being symptomatic
        is_symp = util.binomial_arr(symp_probs) # Determine if they develop symptoms
        symp_inds = inds[is_symp]
        asymp_inds = inds[~is_symp] # Asymptomatic
        # self.flows_variant['new_symptomatic_by_variant'][variant] += len(symp_inds)

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        dur_asym2rec = util.sample(self.pars['asym2rec'], len(asymp_inds))
        self.date_recovered[asymp_inds] = self.date_infectious[asymp_inds] + dur_asym2rec  # Date they recover
        self.dur_disease[asymp_inds] = self.dur_exp2inf[asymp_inds] + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        n_symp_inds = len(symp_inds)
        self.dur_inf2sym[symp_inds] = util.sample(self.pars['inf2sym'], n_symp_inds) # Store how long this person took to develop symptoms
        self.date_symptomatic[symp_inds] = self.date_infectious[symp_inds] + self.dur_inf2sym[symp_inds] # Date they become symptomatic
        sev_probs = self.severe_prob[symp_inds]  # Probability of these people being severe
        is_sev = util.binomial_arr(sev_probs) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe
        # self.flows_variant['new_severe_by_variant'][variant] += len(sev_inds)

        # CASE 2.1: Mild symptoms, no hospitalization required and no probability of death
        dur_mild2rec = util.sample(self.pars['mild2rec'], len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = (self.dur_exp2inf[mild_inds] +
                                       self.dur_inf2sym[mild_inds] +
                                       dur_mild2rec)  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = util.sample(self.pars['sym2sev'], len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = (self.date_symptomatic[sev_inds] +
                                      self.dur_sym2sev[sev_inds])  # Date symptoms become severe
        crit_probs = self.crit_prob[sev_inds]  # Probability of these people becoming critical - higher if no beds available
        is_crit = util.binomial_arr(crit_probs)  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = util.sample(self.pars['sev2rec'], len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = (self.dur_exp2inf[non_crit_inds] +
                                           self.dur_inf2sym[non_crit_inds] +
                                           self.dur_sym2sev[non_crit_inds] +
                                           dur_sev2rec)  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = util.sample(self.pars['sev2crit'], len(crit_inds))
        self.date_critical[crit_inds] = (self.date_severe[crit_inds] +
                                         self.dur_sev2crit[crit_inds])  # Date they become critical
        death_probs = self.death_prob[crit_inds]  # Probability they'll die
        is_dead = util.binomial_arr(death_probs)  # Death outcome
        dead_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # CASE 2.2.2.1: Did not die
        dur_crit2rec = util.sample(self.pars['crit2rec'], len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = (self.dur_exp2inf[alive_inds] +
                                        self.dur_inf2sym[alive_inds] +
                                        self.dur_sym2sev[alive_inds] +
                                        self.dur_sev2crit[alive_inds] +
                                        dur_crit2rec)  # Store how long this person had COVID-19

        # CASE 2.2.2.2: Did die
        dur_crit2die = util.sample(self.pars['crit2die'], len(dead_inds))
        self.date_dead[dead_inds] = self.date_critical[dead_inds] + dur_crit2die # Date of death
        self.dur_disease[dead_inds] = (self.dur_exp2inf[dead_inds] +
                                       self.dur_inf2sym[dead_inds] +
                                       self.dur_sym2sev[dead_inds] +
                                       self.dur_sev2crit[dead_inds] +
                                       dur_crit2die)   # Store how long this person had COVID-19
        self.date_recovered[dead_inds] = np.nan # If they did die, remove them from recovered

        return n_infections  # For incrementing counters

    #%% Methods for updating state

    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet
        the date criterion '''
        if filter_inds is None:
            not_current = self.false(current)
        else:
            not_current = self.ifalsei(current, filter_inds)
        has_date = self.idefinedi(date, not_current)
        inds = self.itrue(self.t >= date[has_date], has_date)
        return inds

    def check_infectious(self):
        ''' Check if they become infectious '''
        inds = self.check_inds(self.infectious, self.date_infectious, filter_inds=self.is_exp)
        self.infectious[inds] = True
        # self.infectious_variant[inds] = self.exposed_variant[inds]
        # for variant in range(self.pars['n_variants']):
        #     this_variant_inds = cvu.itrue(self.infectious_variant[inds] == variant, inds)
        #     n_this_variant_inds = len(this_variant_inds)
        #     self.flows_variant['new_infectious_by_variant'][variant] += n_this_variant_inds
        #     self.infectious_by_variant[variant, this_variant_inds] = True
        return len(inds)


    def check_symptomatic(self):
        ''' Check for new progressions to symptomatic '''
        inds = self.check_inds(self.symptomatic, self.date_symptomatic, filter_inds=self.is_exp)
        self.symptomatic[inds] = True
        return len(inds)


    def check_severe(self):
        ''' Check for new progressions to severe '''
        inds = self.check_inds(self.severe, self.date_severe, filter_inds=self.is_exp)
        self.severe[inds] = True
        return len(inds)


    def check_critical(self):
        ''' Check for new progressions to critical '''
        inds = self.check_inds(self.critical, self.date_critical, filter_inds=self.is_exp)
        self.critical[inds] = True
        return len(inds)


    def check_recovery(self, inds=None, filter_inds='is_exp'):
        '''
        Check for recovery.

        More complex than other functions to allow for recovery to be manually imposed
        for a specified set of indices.
        '''

        # Handle more flexible options for setting indices
        if filter_inds == 'is_exp':
            filter_inds = self.is_exp
        if inds is None:
            inds = self.check_inds(self.recovered, self.date_recovered, filter_inds=filter_inds)

        # Now reset all disease states
        self.exposed[inds]          = False
        self.infectious[inds]       = False
        self.symptomatic[inds]      = False
        self.severe[inds]           = False
        self.critical[inds]         = False
        self.recovered[inds]        = True
        self.recovered_variant[inds] = self.exposed_variant[inds]
        self.infectious_variant[inds] = np.nan
        self.exposed_variant[inds]    = np.nan
        self.exposed_by_variant[:, inds] = False
        self.infectious_by_variant[:, inds] = False


        # Handle immunity aspects
        if self.pars['use_waning']:

            # Reset additional states
            self.susceptible[inds] = True
            self.diagnosed[inds]   = False # Reset their diagnosis state because they might be reinfected

        return len(inds)


    def check_death(self):
        ''' Check whether or not this person died on this timestep  '''
        inds = self.check_inds(self.dead, self.date_dead, filter_inds=self.is_exp)
        self.dead[inds]             = True
        diag_inds = inds[self.diagnosed[inds]] # Check whether the person was diagnosed before dying
        self.known_dead[diag_inds]  = True
        self.susceptible[inds]      = False
        self.exposed[inds]          = False
        self.infectious[inds]       = False
        self.symptomatic[inds]      = False
        self.severe[inds]           = False
        self.critical[inds]         = False
        self.known_contact[inds]    = False
        self.quarantined[inds]      = False
        self.recovered[inds]        = False
        self.infectious_variant[inds] = np.nan
        self.exposed_variant[inds]    = np.nan
        self.recovered_variant[inds]  = np.nan
        return len(inds), len(diag_inds)

