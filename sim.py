''' Main Sim object '''
from base import BaseSim
from parameters import set_defaults
from population import Population
from utils import initialize


__all__ = ['Sim']


class Sim(BaseSim):
    '''
    Main simulation object that handles the movement of agents, the tranmission
    of COVID-19, and the agent decision making.
    '''

    def __init__(self, **kwargs):

        self.pars = set_defaults()
        super().__init__(self.pars)

        # setup the simulation
        self.setup()

    def setup(self):
        '''
        Initial setup for the ABM. Initializes agents with parameters and
        sets a schedule of activation.
        '''

        ''' Setup function initializes wntr and loads various datasets '''
        setup_out = initialize(self.city)

        ''' Parse the output of setup. See the setup function in utils.py
        for more details on the specific outputs. '''
        self.res_nodes = setup_out[0]['res']
        self.ind_nodes = setup_out[0]['ind']
        self.com_nodes = setup_out[0]['com']
        self.cafe_nodes = setup_out[0]['cafe']  # There is no node assigned to "dairy queen" so it was neglected
        if self.city == 'micropolis':
            self.nav_nodes = []  # placeholder for agent assignment
        if self.city == 'mesopolis':
            self.air_nodes = setup_out[0]['air']
            self.nav_nodes = setup_out[0]['nav']

        self.nodes_capacity = setup_out[1]
        self.house_num = setup_out[2]

        self.res_dist = setup_out[3]['res']  # residential capacities at each hour
        self.com_dist = setup_out[3]['com']  # commercial capacities at each hour
        self.ind_dist = setup_out[3]['ind']  # industrial capacities at each hour
        self.sum_dist = setup_out[3]['sum']  # sum of capacities
        self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
        if self.city == 'micropolis':
            # self.cafe_dist = setup_out[3]['cafe']  # restaurant capacities at each hour
            self.nav_dist = [0]  # placeholder for agent assignment
        if self.city == 'mesopolis':
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

    def step(self):
        '''
        Complete one time step of the simulation.
        '''