import os
import sys
if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings
from Hydraulic_abm_SEIR import ConsumerModel
import pandas as pd
from time import localtime, strftime, perf_counter
# from utils import clean_epanet
import os
import wntr
from wntr.network.io import write_inpfile
from tqdm import tqdm

# warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def run_sim(city, id=0, days=90, seed=None, write_inp=False, **kwargs):
    curr_dt = strftime("%Y-%m-%d_%H-%M_" + str(id), localtime())
    if 'output_loc' in kwargs:
        output_loc = kwargs['output_loc'] + str(id)
    else:
        output_loc = 'Output Files/' + curr_dt + '_results'
    os.mkdir(output_loc)

    if city == 'micropolis':
        pop = 4606
    elif city == 'mesopolis':
        pop = 146716
    else:
        pop = 5000
        # raise ValueError(f"City {city} not implemented.")

    start = perf_counter()

    model = ConsumerModel(pop, city, days=days, id=id, seed=seed, **kwargs) #seed=123, wfh_lag=0, no_wfh_perc=0.4

    ''' Save the parameters of the model '''
    model.save_pars(output_loc)

    if kwargs['verbose'] > 0:
        print('Starting simulation ............................')
    # run a warmup period if warmup appears in kwargs
    while model.warmup:
        model.step()
    if kwargs['verbose'] > 0:
        print(f'Warmup period finished with slope {model.water_age_slope}.')

    # run the number of days required by the input
    if kwargs['verbose'] == 0.5:
        for _ in tqdm(range(1, 24*days+1)):
            model.step()
    else:
        for _ in range(1, 24*days+1):
            model.step()

    stop = perf_counter()

    if kwargs['verbose'] > 0:
        print('Time to complete: ', stop - start)

    ''' Save the model outputs '''
    status_tot = convert_to_pd(
        model.status_tot,
        ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild',
         'Sev', 'Crit', 'sum_I', 'wfh']
    )
    # agent_matrix = pd.DataFrame(
    #     model.agent_matrix,
    #     index=[n for n in model.buildings]
    # )

    status_tot.to_pickle(output_loc + "/seir_data.pkl")
    # model.param_out.to_pickle(output_loc + "/params.pkl")
    if model.hyd_sim == 'eos':
        model.demand_matrix.to_pickle(output_loc + "/demand.pkl")
        model.pressure_matrix.to_pickle(output_loc + "/pressure.pkl")
        model.age_matrix.to_pickle(output_loc + "/age.pkl")
        model.flow_matrix.to_pickle(output_loc + "/flow.pkl")
    elif model.hyd_sim in ['hourly', 'monthly']:
        if write_inp:
            write_inpfile(
                model.wn,
                'final_inp_' + str(id) + '.inp',
                units=model.wn.options.hydraulic.inpfile_units
            )
        model.sim.close()
        results = wntr.epanet.io.BinFile().read('temp' + str(id) + '.bin')
        # convert from m^3/s to lps
        demand = results.node['demand'] * 1000
        demand.to_pickle(output_loc + "/demand.pkl")
        results.node['pressure'].to_pickle(output_loc + "/pressure.pkl")
        results.node['quality'].to_pickle(output_loc + "/age.pkl")
        # convert from m^3/s to lps
        flow = results.link['flowrate'] * 1000
        flow.to_pickle(output_loc + "/flow.pkl")
    # agent_matrix.to_pickle(output_loc + "/agent_loc.pkl")

    agents = [str(i) for i in range(model.num_agents)]

    demo = dict()
    demo['white'] = list()
    demo['hispanic'] = list()
    demo['renter'] = list()
    for name, building in model.buildings.items():
        if building.households is not None:
            for house in building.households:
                demo['white'].append(house.white)
                demo['hispanic'].append(house.hispanic)
                demo['renter'].append(house.renter)

    cov_pers = convert_to_pd(model.cov_pers, agents)
    cov_ff = convert_to_pd(model.cov_ff, agents)
    media = convert_to_pd(model.media_exp, agents)
    wfh_dec = convert_to_pd(model.wfh_dec, agents)
    dine_dec = convert_to_pd(model.dine_dec, agents)
    groc_dec = convert_to_pd(model.groc_dec, agents)
    ppe_dec = convert_to_pd(model.ppe_dec, agents)
    bw_cost = convert_to_pd(model.bw_cost, model.h_id)
    tw_cost = convert_to_pd(model.tw_cost, model.h_id)
    bw_demand = convert_to_pd(model.bw_demand, model.h_id)
    hygiene = convert_to_pd(model.hygiene, model.h_id)
    drink = convert_to_pd(model.drink, model.h_id)
    cook = convert_to_pd(model.cook, model.h_id)
    demo = convert_to_pd(demo, model.h_id)
    # income = convert_to_pd({0: model.income}, households)
    # income_level = convert_to_pd({0: model.income_level}, households)

    cov_pers.to_pickle(output_loc + "/cov_pers.pkl")
    cov_ff.to_pickle(output_loc + "/cov_ff.pkl")
    media.to_pickle(output_loc + "/media.pkl")
    wfh_dec.to_pickle(output_loc + "/wfh.pkl")
    dine_dec.to_pickle(output_loc + "/dine.pkl")
    groc_dec.to_pickle(output_loc + "/groc.pkl")
    ppe_dec.to_pickle(output_loc + "/ppe.pkl")
    bw_cost.to_pickle(output_loc + "/bw_cost.pkl")
    tw_cost.to_pickle(output_loc + "/tw_cost.pkl")
    bw_demand.to_pickle(output_loc + "/bw_demand.pkl")
    hygiene.to_pickle(output_loc + "/hygiene.pkl")
    drink.to_pickle(output_loc + "/drink.pkl")
    cook.to_pickle(output_loc + "/cook.pkl")
    # income.to_pickle(output_loc + "/income.pkl")
    # income_level.to_pickle(output_loc + "/income_level.pkl")
    model.income_comb.to_pickle(output_loc + "/income.pkl")
    demo.to_pickle(output_loc + '/demo.pkl')


def convert_to_pd(in_list, columns):
    return pd.DataFrame.from_dict(in_list, orient='index', columns=columns)
