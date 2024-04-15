import os
import sys
if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
from Hydraulic_abm_SEIR import ConsumerModel
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime, perf_counter
# from utils import clean_epanet
import os
import wntr
from tqdm import tqdm
import logging
import numpy as np
from wntr.network.io import write_inpfile

warnings.simplefilter("ignore", UserWarning)


def run_sim(city, id=0, days=90, plot=False, seed=218, **kwargs):
    curr_dt = strftime("%Y-%m-%d_%H-%M_" + str(id), localtime())
    output_loc = 'Output Files/' + curr_dt + '_results'
    os.mkdir(output_loc)
    output_file = 'datasheet.xlsx'

    if city == 'micropolis':
        pop = 4606
    elif city == 'mesopolis':
        pop = 146716
    else:
        print(f"City {city} not implemented.")

    if 'hyd_sim' in kwargs:
        hyd_sim = kwargs['hyd_sim']
    else:
        hyd_sim = 'eos'

    start = perf_counter()

    model = ConsumerModel(pop, city, days=days, id=id, seed=seed, **kwargs) #seed=123, wfh_lag=0, no_wfh_perc=0.4

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

    # save the input file to test run time
    # write_inpfile(
    #     model.wn,
    #     'final_wnm.inp',
    #     units=model.wn.options.hydraulic.inpfile_units,
    #     version=2.2
    # )

    stop = perf_counter()

    if kwargs['verbose'] != 0:
        print('Time to complete: ', stop - start)

    # model.status_tot['t'] = pop * model.status_tot['t'] / 24

    if plot:
        Demands_test = np.zeros(24 * days + 1)
        # print(model.demand_matrix)
        for node in model.terminal_nodes:
            add_demands = model.demand_matrix[node]
            Demands_test = np.add(Demands_test, add_demands)
        plt.plot(Demands_test)
        # plt.plot(Demands_test, label='Total Demand')
        plt.xlabel("Time (sec)")
        plt.ylabel("Demand (L)")
        plt.legend(loc='best')
        plt.savefig(output_loc + '/' + 'demands.png')
        plt.close()
        print('Total demands are ' + str(Demands_test.sum(axis = 0)))

        # print(Demands_test[:,0])

        plt.plot('t', 'S', data = model.status_tot, label = 'Susceptible')
        plt.plot('t', 'E', data = model.status_tot, label = 'Exposed')
        plt.plot('t', 'I', data = model.status_tot, label = 'Infected')
        plt.plot('t', 'R', data = model.status_tot, label = 'Recovered')
        plt.plot('t', 'D', data = model.status_tot, label = 'Dead')
        plt.xlabel('Time (days)')
        plt.ylabel('Percent Population')
        plt.legend()
        plt.savefig(output_loc + '/' + 'seir.png')
        plt.close()

        plt.plot('t', 'I', data=model.status_tot, label='Infected')
        plt.plot('t', 'sum_I', data=model.status_tot, label='Cumulative I')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.legend()
        plt.savefig(output_loc + '/' + 'infected.png')
        plt.close()

    ''' Save the model outputs '''
    # model.status_tot['t'] = model.status_tot['t'] * 24 * 3600
    # model.status_tot['t'] = pd.to_numeric(model.status_tot['t'],downcast="integer")
    # model.status_tot = model.status_tot.set_index('t')
    # model.status_tot = pd.concat([model.status_tot, Demands_test], axis=1)

    # convert list of lists to pandas dataframes
    # print(model.status_tot)
    status_tot = convert_to_pd(
        model.status_tot,
        ['t', 'S', 'E', 'I', 'R', 'D', 'Symp', 'Asymp', 'Mild',
         'Sev', 'Crit', 'sum_I', 'wfh']
    )
    agent_matrix = convert_to_pd(
        model.agent_matrix,
        [n for n in model.nodes_capacity]
    )

    status_tot.to_pickle(output_loc + "/seir_data.pkl")
    model.param_out.to_pickle(output_loc + "/params.pkl")
    if hyd_sim == 'eos':
        model.demand_matrix.to_pickle(output_loc + "/demand.pkl")
        model.pressure_matrix.to_pickle(output_loc + "/pressure.pkl")
        model.age_matrix.to_pickle(output_loc + "/age.pkl")
        model.flow_matrix.to_pickle(output_loc + "/flow.pkl")
    elif hyd_sim == 'hourly' or hyd_sim == 'monthly':
        model.sim.close()
        results = wntr.epanet.io.BinFile().read('temp' + str(id) + '.bin')
        demand = results.node['demand'] * 1000000
        demand.to_pickle(output_loc + "/demand.pkl")
        results.node['pressure'].to_pickle(output_loc + "/pressure.pkl")
        results.node['quality'].to_pickle(output_loc + "/age.pkl")
        flow = results.link['flowrate'] * 1000000
        flow.to_pickle(output_loc + "/flow.pkl")
    agent_matrix.to_pickle(output_loc + "/agent_loc.pkl")

    agents = [str(i) for i in range(model.num_agents)]
    households = [h.node for i, hs in model.households.items() for h in hs]
    # income = list()
    # for node, houses in model.households.items():
    #     for house in houses:
    #         income.append(house.income)

    cov_pers = convert_to_pd(model.cov_pers, agents)
    cov_ff = convert_to_pd(model.cov_ff, agents)
    media = convert_to_pd(model.media_exp, agents)
    wfh_dec = convert_to_pd(model.wfh_dec, agents)
    dine_dec = convert_to_pd(model.dine_dec, agents)
    groc_dec = convert_to_pd(model.groc_dec, agents)
    ppe_dec = convert_to_pd(model.ppe_dec, agents)
    bw_cost = convert_to_pd(model.bw_cost, households)
    tw_cost = convert_to_pd(model.tw_cost, households)
    bw_demand = convert_to_pd(model.bw_demand, households)
    hygiene = convert_to_pd(model.hygiene, households)
    drink = convert_to_pd(model.drink, households)
    cook = convert_to_pd(model.cook, households)
    # income = convert_to_pd({0: model.income}, households)
    # income_level = convert_to_pd({0: model.income_level}, households)
    traditional = convert_to_pd(model.traditional, [0])
    burden = convert_to_pd(model.burden, [0])

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
    traditional.to_pickle(output_loc + "/traditional.pkl")
    burden.to_pickle(output_loc + "/burden.pkl")

    with pd.ExcelWriter(output_loc + '/' + output_file) as writer:
        model.param_out.to_excel(writer, sheet_name='params')


def convert_to_pd(in_list, columns):
    return pd.DataFrame.from_dict(in_list, orient='index', columns=columns)
