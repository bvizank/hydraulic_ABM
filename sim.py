''' Main Sim object '''
from base import BaseSim
from parameters import set_defaults
from population import Population


__all__ = ['Sim']


class Sim(BaseSim):
    '''
    Main simulation object that handles the movement of agents, the tranmission
    of COVID-19, and the agent decision making. Setup is handled in setup.py.
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

        ''' Population object has attributes according to the agent_pars list
        in parameters.py. '''
        self.pop = Population(self.pars)