import numpy as np
import pandas as pd
from parameters import agent_pars
from utils import choose, sample


class Population():
    '''
    Class for holding dictionary of lists that contains the agent parameters.
    '''

    def __init__(self, pars):
        '''
        Set the agent parameters. Each dictionary entry is a numpy list of
        the agent attribute given by the key.
        '''

        self.pars = pars

        for key in agent_pars:
            if key == 'uid':
                self[key] = np.arange(pars['pop_size'], dtype=np.int32)
            elif key == 'home_node' or key == 'work_node' or key == 'work_type':
                self[key] = np.full(pars['pop_size'], np.nan)
            elif key == 'housemates':
                self[key] = np.full((6, pars['pop_size']), 0, dtype=np.int32)
            elif 'time' in key or \
                 key == 'covid' or \
                 key == 'symp_status':
                self[key] = np.full(pars['pop_size'], 0, dtype=np.int32)
            else:
                self[key] = np.full(pars['pop_size'], np.nan, dtyp=np.float32)

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
        self.e2i = sample(dist='lognormal', par1=4.5, par2=1.5)
        self.i2s = sample(dist='lognormal', par1=1.1, par2=0.9)
        self.s2sev = sample(dist='lognormal', par1=6.6, par2=4.9)
        self.sev2c = sample(dist='lognormal', par1=1.5, par2=2.0)
        self.c2d = sample(dist='lognormal', par1=10.7, par2=4.8)

        ''' Set the recovery time values for each agent and state  '''
        self.asym_rec = sample(dist='lognormal', par1=8.0, par2=2.0)
        self.mild_rec = sample(dist='lognormal', par1=8.0, par2=2.0)
        self.sev_rec = sample(dist='lognormal', par1=18.1, par2=6.3)
        self.crit_rec = sample(dist='lognormal', par1=18.1, part2=6.3)

        ''' Set the initial number of infectious '''
        inds = choose(self.pars['pop_size'], self.pars['int_infectious'])
        self.covid[inds] = 2

    def set_bbn_attrs(self):
        '''
        Set the BBN parameters for each agent.
        '''

        all_bbn = pd.read_csv(r'Input Files/all_bbn_data.csv')
        bbn_list = all_bbn.columns.to_numpy()  # list of bbn parameter names
        # create dictionary with each bbn param name as key and a numeric
        # index as the value. For looking up bbn params in the numpy array.
        self.bbn_dict = dict((key, i) for i, key in enumerate(bbn_list))

        # need to convert pandas dataframe to numpy to store in pop dictionary
        all_bbn = all_bbn.to_numpy()
        inds = choose(len(all_bbn), self.pars['pop_size'])
        self['bbn_params'] = all_bbn[inds]

    def set_move_attrs(self):
        '''
        Set the location related attributes for each agent.
        '''

        init_wntr()
        self['home_node']
        self['work_node']
        self['work_type']
        self['curr_node']
