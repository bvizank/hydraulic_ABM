''' Parameter classes for envrionment, agent, and model parameters '''
from base import BaseParams


__all__ = ['EnvParams', 'AgentParams', 'ModParams']

class EnvParams(BaseParams):
    '''
    Parameters related to the environment.
    '''


class AgentParams(BaseParams):
    '''
    Parameters related to the agents.
    '''

    def __init__(self):
        self.parameters = [
            'uid',
            'home_node',
            'work_node',
            'work_type',
            'demand',
            'base_demand',
            'age',
            'covid',
            'exp_time',
            'inf_time',
            'sym_time',
            'sev_time',
            'crit_time',
            'symp_status',
            'inf_sev',
            'ff_cov_change',
            'wfh',
            'dine',
            'groc',
            'ppe',
            'can_wfh',
            'bbn_params',
            'housemates'
        ]


class ModParams(BaseParams):
    '''
    Parameters related to the model.
    '''

    def __init__(self):
        self.parameters = [
            'days',
            'id',
            'num_agents'
        ]
