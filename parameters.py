''' Parameter values '''


__all__ = ['agent_pars', 'set_defaults']

agent_pars = [
    'uid',
    'home_node',
    'work_node',
    'work_ind',
    'work_nav',
    'move_res',
    'move_nonres',
    'demand',
    'base_demand',
    'age',
    'ff_cov_change',
    'wfh',
    'dine',
    'groc',
    'ppe',
    'can_wfh',
    'bbn_params',
    'housemates',
]


prob_pars = [
    'symp_prob',
    'severe_prob',
    'crit_prob',
    'death_prob',
]


state_pars = [
    'susceptible',
    'exposed',
    'infected',
    'recovered',
    'dead',
    'symp',
    'mild',
    'severe',
    'critical',
]


dur_pars = [
    'dur_exp2inf',
    'dur_inf2sym',
    'dur_sym2sev',
    'dur_sev2crit',
    'dur_disease'
]


date_pars = [
    'date_exposed',
    'date_infectious',
    'date_symptomatic',
    'date_severe',
    'data_critical',
    'date_recovered'
]


# Set values for susceptibility based on age. From https://doi.org/10.1371/journal.pcbi.1009149
susDict = {1: [0.525, 0.001075, 0.000055, 0.00002],
           2: [0.6, 0.0072, 0.00036, 0.0001],
           3: [0.65, 0.0208, 0.00104, 0.00032],
           4: [0.7, 0.0343, 0.00216, 0.00098],
           5: [0.75, 0.07650, 0.00933, 0.00265],
           6: [0.8, 0.1328, 0.03639, 0.00766],
           7: [0.85, 0.20655, 0.08923, 0.02439],
           8: [0.9, 0.2457, 0.1742, 0.08292],
           9: [0.9, 0.2457, 0.1742, 0.1619]}


ages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
age_weights = [0.25, 0.18, 0.15, 0.14, 0.12, 0.08, 0.05, 0.01, 0.01]


def set_defaults(**kwargs):
    pars = {
        'days': 90,
        'id': 0,
        'city': 'micropolis',
        'inf_perc': 0.01,
        'res_inf_rate': 0.05,
        'nonres_inf_rate': 0.01
    }

    pars.update(kwargs)

    # Set the number of initial number of infectious
    if 'pop_size' in pars:
        pars['int_infectious'] = pars['pop_size'] * pars['inf_perc']
    else:
        errormsg = f'No population size defined.'
        raise ValueError(errormsg)

    # set duration parameters
    pars['exp2inf'] = dict(dist='lognormal_int', par1=4.5, par2=1.5) # Duration from exposed to infectious; see Lauer et al., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/, appendix table S2, subtracting inf2sym duration
    pars['inf2sym'] = dict(dist='lognormal_int', par1=1.1, par2=0.9) # Duration from infectious to symptomatic; see Linton et al., https://doi.org/10.3390/jcm9020538, from Table 2, 5.6 day incubation period - 4.5 day exp2inf from Lauer et al.
    pars['sym2sev'] = dict(dist='lognormal_int', par1=6.6, par2=4.9) # duration from symptomatic to severe symptoms; see linton et al., https://doi.org/10.3390/jcm9020538, from table 2, 6.6 day onset to hospital admission (deceased); see also wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044, 7 days (table 1)
    pars['sev2crit'] = dict(dist='lognormal_int', par1=1.5, par2=2.0) # duration from severe symptoms to requiring icu; average of 1.9 and 1.0; see chen et al., https://www.sciencedirect.com/science/article/pii/s0163445320301195, 8.5 days total - 6.6 days sym2sev = 1.9 days; see also wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044, table 3, 1 day, iqr 0-3 days; std=2.0 is an estimate

    # duration parameters: time for disease recovery
    pars['asym2rec'] = dict(dist='lognormal_int', par1=8.0,  par2=2.0) # Duration for asymptomatic people to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['mild2rec'] = dict(dist='lognormal_int', par1=8.0,  par2=2.0) # Duration for people with mild symptoms to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['sev2rec'] = dict(dist='lognormal_int', par1=18.1, par2=6.3) # Duration for people with severe symptoms to recover, 24.7 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 18.1 days = 24.7 onset-to-recovery - 6.6 sym2sev; 6.3 = 0.35 coefficient of variation * 18.1; see also https://doi.org/10.1017/S0950268820001259 (22 days) and https://doi.org/10.3390/ijerph17207560 (3-10 days)
    pars['crit2rec'] = dict(dist='lognormal_int', par1=18.1, par2=6.3) # Duration for people with critical symptoms to recover; as above (Verity et al.)
    pars['crit2die'] = dict(dist='lognormal_int', par1=10.7, par2=4.8) # Duration from critical symptoms to death, 18.8 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 10.7 = 18.8 onset-to-death - 6.6 sym2sev - 1.5 sev2crit; 4.8 = 0.45 coefficient of variation * 10.7

    return pars
