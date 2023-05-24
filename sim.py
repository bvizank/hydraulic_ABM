''' Main Sim object '''
from base import BaseSim
from parameters import set_defaults
from population import Population
from utils import initialize
import time
import copy
import numpy as np


__all__ = ['Sim']


class Sim(BaseSim):
    '''
    Main simulation object that handles the movement of agents, the tranmission
    of COVID-19, and the agent decision making.
    '''

    def __init__(self, **kwargs):

        self.pars = set_defaults(**kwargs)
        super().__init__(self.pars)

        start = time.perf_counter()
        # setup the simulation
        self.setup()
        end = time.perf_counter()
        print(end-start)

    def setup(self):
        '''
        Initial setup for the ABM. Initializes agents with parameters and
        sets a schedule of activation.
        '''

        ''' Setup function initializes wntr and loads various datasets '''
        setup_out = initialize(self['city'])

        ''' Parse the output of setup. See the setup function in utils.py
        for more details on the specific outputs. '''
        self.res_nodes = setup_out[0]['res']
        self.ind_nodes = setup_out[0]['ind']
        self.com_nodes = setup_out[0]['com']
        self.cafe_nodes = setup_out[0]['cafe']  # There is no node assigned to "dairy queen" so it was neglected
        self.all_nodes = self.res_nodes + self.ind_nodes + self.com_nodes + \
                         self.cafe_nodes
        if self['city'] == 'micropolis':
            self.nav_nodes = []  # placeholder for agent assignment
        if self['city'] == 'mesopolis':
            self.air_nodes = setup_out[0]['air']
            self.nav_nodes = setup_out[0]['nav']
            self.all_nodes = self.all_nodes + self.air_nodes + self.nav_nodes

        self.nodes_capacity = setup_out[1]
        self.house_num = setup_out[2]

        self.res_dist = setup_out[3]['res']  # residential capacities at each hour
        self.com_dist = setup_out[3]['com']  # commercial capacities at each hour
        self.ind_dist = setup_out[3]['ind']  # industrial capacities at each hour
        self.sum_dist = setup_out[3]['sum']  # sum of capacities
        self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
        if self['city'] == 'micropolis':
            # self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
            self.nav_dist = [0]  # placeholder for agent assignment
        if self['city'] == 'mesopolis':
            self.air_dist = setup_out[3]['air']
            self.nav_dist = setup_out[3]['nav']

        self.sleep = setup_out[4]['sleep']
        self.radio = copy.deepcopy(setup_out[4]['radio'])
        self.tv = copy.deepcopy(setup_out[4]['tv'])
        self.bbn_params = setup_out[5]  # pandas dataframe of bbn parameters
        wfh_patterns = setup_out[6]
        self.terminal_nodes = setup_out[7]
        self.wn = setup_out[8]

        ''' Population object has attributes according to the agent_pars list
        in parameters.py. '''
        self.pop = Population(self, self.pars)

        ''' Move the initial number of industrial agents '''
        self.pop.move_agents(res2ind=np.amax(self.ind_dist))

    def step(self):
        '''
        Complete one time step of the simulation.
        '''

        ''' Move the correct number of industrial agents '''
        if self.timestep % 24 == 6 or \
           self.timestep % 24 == 14 or \
           self.timestep % 25 == 22:
            self.pop.move_agents(ind2res=self.ind_dist[self.timestep % 24]/2,
                                 res2ind=self.ind_dist[self.timestep % 24]/2)

        ''' Move the correct number of cafe agents '''
        if self.cafe_dist[self.timestep] > 0:
            self.pop.move_agents(res2caf=self.cafe_dist[self.timestep % 24])
        else:
            self.pop.move_agents(caf2res=self.cafe_dist[self.timestep % 24])

        ''' Move the correct number of com agents '''
        if self.cafe_dist[self.timestep % 24] > 0:
            self.pop.move_agents(res2com=self.com_dist[self.timestep % 24])
        else:
            self.pop.move_agents(com2res=self.com_dist[self.timestep % 24])

        ''' Need to add air and nav movements '''

    def run(self, steps):
        '''
        Run the full simulation.
        '''

        start_model = time.perf_counter()
        self.timestep = 0
        for s in range(steps):
            print(f"Step {s}..............")
            self.step()
            self.timestep += 1

        end_model = time.perf_counter()
        print(end_model - start_model)