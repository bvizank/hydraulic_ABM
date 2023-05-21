''' Parameter values '''


__all__ = ['agent_pars', 'set_defaults']

agent_pars = [
    'uid',
    'home_node',
    'work_node',
    'work_type',
    'curr_node',
    'demand',
    'base_demand',
    'age',
    'covid',
    'exp_time',
    'inf_time',
    'sym_time',
    'sev_time',
    'crit_time',
    'e2i',
    'i2s',
    's2sev',
    'sev2c',
    'c2d',
    'asym_rec',
    'mild_rec',
    'sev_rec',
    'crit_rec',
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


def set_defaults(**kwargs):
    pars = {
        'days': 90,
        'id': 0,
        'city': 'micropolis',
        'inf_perc': 0.01
    }

    pars.update(kwargs)

    # Set the number of initial number of infectious
    if 'pop_size' in pars:
        pars['int_infectious'] = pars['pop_size'] * pars['inf_perc']
    else:
        errormsg = f'No population size defined.'
        raise ValueError(errormsg)


    return pars
