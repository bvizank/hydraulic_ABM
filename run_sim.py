import os
import sys
print(sys.platform)
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

warnings.simplefilter("ignore", UserWarning)


def run_sim(city, id=0, days=90, plot=False, **kwargs):
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

    model = ConsumerModel(pop, city, days=days, id=id, **kwargs) #seed=123, wfh_lag=0, no_wfh_perc=0.4
    if kwargs['verbose'] == 0.5:
        for t in tqdm(range(24*days)):
            model.step()
    else:
        for t in range(24*days):
            model.step()

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
    elif hyd_sim == 'hourly' or isinstance(hyd_sim, int):
        model.sim.close()
        results = wntr.epanet.io.BinFile().read('temp' + str(id) + '.bin')
        demand = results.node['demand'] * 1000000
        demand.to_pickle(output_loc + "/demand.pkl")
        results.node['pressure'].to_pickle(output_loc + "/pressure.pkl")
        results.node['quality'].to_pickle(output_loc + "/age.pkl")
        flow = results.link['flowrate'] * 1000000
        flow.to_pickle(output_loc + "/flow.pkl")
    agent_matrix.to_pickle(output_loc + "/agent_loc.pkl")

    cov_pers = convert_to_pd(model.cov_pers, [str(i) for i in range(model.num_agents)])
    cov_ff = convert_to_pd(model.cov_ff, [str(i) for i in range(model.num_agents)])
    media = convert_to_pd(model.media_exp, [str(i) for i in range(model.num_agents)])
    wfh_dec = convert_to_pd(model.wfh_dec, [str(i) for i in range(model.num_agents)])
    dine_dec = convert_to_pd(model.dine_dec, [str(i) for i in range(model.num_agents)])
    groc_dec = convert_to_pd(model.groc_dec, [str(i) for i in range(model.num_agents)])
    ppe_dec = convert_to_pd(model.ppe_dec, [str(i) for i in range(model.num_agents)])
    income = pd.DataFrame.from_dict(model.income)

    cov_pers.to_pickle(output_loc + "/cov_pers.pkl")
    cov_ff.to_pickle(output_loc + "/cov_ff.pkl")
    media.to_pickle(output_loc + "/media.pkl")
    wfh_dec.to_pickle(output_loc + "/wfh.pkl")
    dine_dec.to_pickle(output_loc + "/dine.pkl")
    groc_dec.to_pickle(output_loc + "/groc.pkl")
    ppe_dec.to_pickle(output_loc + "/ppe.pkl")
    income.to_pickle(output_loc + "/income.pkl")

    with pd.ExcelWriter(output_loc + '/' + output_file) as writer:
        # model.status_tot.to_excel(writer, sheet_name='seir_data')
        model.param_out.to_excel(writer, sheet_name='params')
    #     model.demand_matrix.to_excel(writer, sheet_name='demand')
    #     model.pressure_matrix.to_excel(writer, sheet_name='pressure')
    #     model.age_matrix.to_excel(writer, sheet_name='age')
    #     model.agent_matrix.to_excel(writer, sheet_name='agent locations')
    #     model.flow_matrix.to_excel(writer, sheet_name='flow')

    # with pd.ExcelWriter(output_loc + '/' + 'agent_params.xlsx') as writer:
    #     model.cov_pers.to_excel(writer, sheet_name='cov_pers')
    #     model.cov_ff.to_excel(writer, sheet_name='cov_ff')
    #     model.media_exp.to_excel(writer, sheet_name='media')
    #     model.wfh_dec.to_excel(writer, sheet_name='wfh')
    #     model.dine_dec.to_excel(writer, sheet_name='dine')
    #     model.groc_dec.to_excel(writer, sheet_name='groc')
    #     model.ppe_dec.to_excel(writer, sheet_name='ppe')

    #  clean_epanet('.')  # clean all input, bin, and rpt files


def convert_to_pd(in_list, columns):
    return pd.DataFrame.from_dict(in_list, orient='index', columns=columns)
