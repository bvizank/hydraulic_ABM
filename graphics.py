import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import networkx as nx
import os

output_loc = 'Output Files/2022-10-27_18-25_all_pb_current_results/'
data_file = output_loc + 'datasheet.xlsx'

pkls = [file for file in os.listdir(output_loc) if file.endswith(".pkl")]
print(pkls)

if not pkls:
    print('Importing data from excel')
    seir = pd.read_excel(data_file, sheet_name='seir_data', index_col=1)
    demand = pd.read_excel(data_file, sheet_name='demand', index_col=0)
    pressure = pd.read_excel(data_file, sheet_name='pressure', index_col=0)
    age = pd.read_excel(data_file, sheet_name='age', index_col=0)
    agent = pd.read_excel(data_file, sheet_name='agent locations', index_col=0)
    flow = pd.read_excel(data_file, sheet_name='flow', index_col=0)
    seir.to_pickle(output_loc + 'seir_data.pkl')
    demand.to_pickle(output_loc + 'demand.pkl')
    pressure.to_pickle(output_loc + 'pressure.pkl')
    age.to_pickle(output_loc + 'age.pkl')
    agent.to_pickle(output_loc + 'agent locations.pkl')
    flow.to_pickle(output_loc + 'flow.pkl')
else:
    for pkl in pkls:
        file_name = pkl[:-4]
        if file_name == 'seir_data':
            print('Reading seir pickle data')
            seir = pd.read_pickle(output_loc + pkl)
        elif file_name == 'demand':
            print('Reading demand pickle data')
            demand = pd.read_pickle(output_loc + pkl)
        elif file_name == 'pressure':
            print('Reading pressure pickle data')
            pressure = pd.read_pickle(output_loc + pkl)
        elif file_name == 'age':
            print('Reading age pickle data')
            age = pd.read_pickle(output_loc + pkl)
        elif file_name == 'agent locations':
            print('Redaing agent pickle data')
            agent = pd.read_pickle(output_loc + pkl)
        elif file_name == 'flow':
            print('Reading flow pickle data')
            flow = pd.read_pickle(output_loc + pkl)

    if 'seir_data.pkl' not in pkls:
        seir = pd.read_excel(data_file, sheet_name='seir_data', index_col=1)
        seir.to_pickle(output_loc + 'seir_data.pkl')
    if 'demand.pkl' not in pkls:
        demand = pd.read_excel(data_file, sheet_name='demand', index_col=0)
        demand.to_pickle(output_loc + 'demand.pkl')
    if 'pressure.pkl' not in pkls:
        pressure = pd.read_excel(data_file, sheet_name='pressure', index_col=0)
        pressure.to_pickle(output_loc + 'pressure.pkl')
    if 'age.pkl' not in pkls:
        age = pd.read_excel(data_file, sheet_name='age', index_col=0)
        age.to_pickle(output_loc + 'age.pkl')
    if 'agent locations.pkl' not in pkls:
        agent = pd.read_excel(data_file, sheet_name='agent locations', index_col=0)
        agent.to_pickle(output_loc + 'agent locations.pkl')
    if 'flow.pkl' not in pkls:
        flow = pd.read_excel(data_file, sheet_name='flow', index_col=0)
        flow.to_pickle(output_loc + 'flow.pkl')

'''Import water network'''
inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()


def calc_difference(data_time_1, data_time_2):
    '''Function to take the difference between two time points '''
    return (data_time_2 - data_time_1)


def calc_flow_diff(data_time_1, data_time_2):
    output = dict()
    for i in range(len(data_time_1)):
        if data_time_2[i] * data_time_1[i] < 0:
            print(data_time_2)
    return ([1 if data_time_2[i]*data_time_1[i] < 0 else 0 for i in range(len(data_time_1))])


def make_contour(graph, data, data_type, fig_name,
                 label=False, label_val='', pts=100000, **plots):
    '''Function to make contour plot given a network structure and supplied data'''
    x_coords = list()
    y_coords = list()
    data_list = list()
    pos = dict()
    if data_type != 'agent':
        for node in graph.nodes:
            x_coord = graph.nodes[node]['pos'][0]
            y_coord = graph.nodes[node]['pos'][1]
            curr_data = data[node]
            x_coords.append(x_coord)
            y_coords.append(y_coord)
            data_list.append(curr_data)

            pos[node] = x_coord, y_coord
    else:
        for node in graph.nodes:
            x_coord = graph.nodes[node]['pos'][0]
            y_coord = graph.nodes[node]['pos'][1]
            if node in data:
                curr_data = data[node]
            else:
                curr_data = 0
            x_coords.append(x_coord)
            y_coords.append(y_coord)
            data_list.append(curr_data)
                
            pos[node] = x_coord, y_coord
                
    # x_mesh = np.linspace(np.min(x_coords), np.max(x_coords), int(np.sqrt(pts)))
    # y_mesh = np.linspace(np.min(y_coords), np.max(y_coords), int(np.sqrt(pts)))
    # [x,y] = np.meshgrid(x_mesh, y_mesh)

    # z = griddata((x_coords, y_coords), data_list, (x, y), method='linear')
    # x = np.matrix.flatten(x); #Gridded longitude
    # y = np.matrix.flatten(y); #Gridded latitude
    # z = np.matrix.flatten(z); #Gridded elevation

    # if 'vmax' in plots:
    #     plt.scatter(x,y,1,z,vmin=plots['vmin'], vmax=plots['vmax'])
    # else:
    #     plt.scatter(x,y,1,z,vmin=plots['vmin'])

    # nx.draw_networkx(graph, pos=pos, with_labels=False, arrowstyle='-',
    #                  node_size=0)
    # if label:
    #     plt.colorbar(label=label_val)
    # plt.savefig(fig_name)
    # plt.close()
    
    ax = wntr.graphics.plot_network(wn, node_attribute=data_list,
                                    node_colorbar_label=label_val)

    # nx.draw_networkx(graph, pos=pos, with_labels=False, arrowstyle='-',
    #                  node_size=10, node_color=data_list)
    plt.savefig(fig_name + 'nodes')
    plt.close()


def make_sector_plot(wn, data, ylabel, type=None):
    '''
    Function to plot the average data for a given sector
    Sectors include: residential, commercial, industrial
    '''
    if type == 'residential':
        nodes = [name for name,node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name == '2']
    elif type == 'industrial':
        nodes = [name for name,node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name == '3']
    elif type == 'commercial':
        nodes = [name for name,node in wn.junctions()
                 if (node.demand_timeseries_list[0].pattern_name == '4' or
                     node.demand_timeseries_list[0].pattern_name == '5' or
                     node.demand_timeseries_list[0].pattern_name == '6')]
    elif type == None:
        res_nodes = [name for name,node in wn.junctions()
                     if node.demand_timeseries_list[0].pattern_name == '2']
        ind_nodes = [name for name,node in wn.junctions()
                     if node.demand_timeseries_list[0].pattern_name == '3']
        com_nodes = [name for name,node in wn.junctions()
                     if node.demand_timeseries_list[0].pattern_name == '4' or
                     node.demand_timeseries_list[0].pattern_name == '5' or
                     node.demand_timeseries_list[0].pattern_name == '6']

    if type != None:
        y_data = data[nodes].mean(axis=1)
        print(y_data)
    else:
        res_data = data[res_nodes].mean(axis=1)
        ind_data = data[ind_nodes].mean(axis=1)
        com_data = data[com_nodes].mean(axis=1)
        res_data.rolling(24).mean().plot()
        ind_data.rolling(24).mean().plot()
        com_data.rolling(24).mean().plot()
        plt.legend(['Residential', 'Industrial', 'Commercial'])
        plt.xlabel('Time [sec]')
        plt.ylabel(ylabel)
        plt.savefig(output_loc + ylabel)


def make_flow_plot(wn, data):
    '''
    Function to make a plot showing the flow direction change
    '''
    ax = wntr.graphics.plot_network(wn, link_attribute=data,
                                    node_colorbar_label='Change in Direction')
    plt.show()


def export_agent_loc(wn, locations):
    res_nodes = [name for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name == '2'
                 and name in locations.columns]
    ind_nodes = [name for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name == '3'
                 and name in locations.columns]
    com_nodes = [name for name, node in wn.junctions()
                 if (node.demand_timeseries_list[0].pattern_name == '4' or
                 node.demand_timeseries_list[0].pattern_name == '5' or
                 node.demand_timeseries_list[0].pattern_name == '6')
                 and name in locations.columns]
    rest_nodes = [name for name, node in wn.junctions()
                  if node.demand_timeseries_list[0].pattern_name == '1'
                  and name in locations.columns]

    res_loc = locations[res_nodes].sum(axis=1)
    ind_loc = locations[ind_nodes].sum(axis=1)
    com_loc = locations[com_nodes].sum(axis=1)
    rest_loc = locations[rest_nodes].sum(axis=1)
    output = pd.DataFrame({'res': res_loc,
                           'ind': ind_loc,
                           'com': com_loc,
                           'rest': rest_loc})
    output.to_csv(output_loc + 'locations.csv')


max_wfh = seir.wfh.loc[int(seir.wfh.idxmax())]
times = []
# times = times + [seir.wfh.searchsorted(max_wfh/4)+12]
times = times + [seir.wfh.searchsorted(max_wfh/2)+12]
# print(seir.wfh.searchsorted(max_wfh/2))
# times = times + [seir.wfh.searchsorted(max_wfh*3/4)+12]
times = times + [seir.wfh.searchsorted(max_wfh)-12]
print(times)
times_hour = [time % 24 for time in times]
print(times_hour)

def check_stats(new_list, old_stats):
    if new_list.max() > old_stats[0]:
        old_stats[0] = new_list.max()
    if new_list.min() < old_stats[1]:
        old_stats[1] = new_list.min()
    
    return old_stats

demand_stats = [0,0]
pressure_stats = [0,0]
age_stats = [0,0]
agent_stats = [0,0]
demand_diff = list()
pressure_diff = list()
age_diff = list()
agent_diff = list()
flow_diff = list()

for i, time in enumerate(times):
    if time >= len(demand):
        time = time - 1
    demand_diff.append(calc_difference(demand.iloc[times_hour[i]], demand.iloc[time]) * 1000)
    pressure_diff.append(calc_difference(pressure.iloc[times_hour[i]], pressure.iloc[time]))
    age_diff.append(calc_difference(age.iloc[times_hour[i]], pressure.iloc[time]))
    agent_diff.append(calc_difference(agent.iloc[times_hour[i]], agent.iloc[time]))
    demand_stats = check_stats(demand_diff[i], demand_stats)
    pressure_stats = check_stats(pressure_diff[i], pressure_stats)
    age_stats = check_stats(age_diff[i], age_stats)
    agent_stats = check_stats(agent_diff[i], agent_stats)
    print(demand_stats)
    # flow_diff.append(calc_flow_diff(flow.iloc[times_hour[i]], flow.iloc[time]))
    # print(f"Flow at time {time} changed in {sum(flow_diff[i])} pipes.")

for i, time in enumerate(times):
    if time >= len(demand):
        time = time - 1
  # make_contour(G, demand_diff[i], 'demand', output_loc + 'demand_' + str(time), True,
  #              'Demand [ML]', vmin=demand_stats[1], vmax=demand_stats[0])
  # make_contour(G, pressure_diff[i], 'pressure', output_loc + 'pressure_' + str(time), True,
  #              'Pressure [m]', vmin=pressure_stats[1], vmax=pressure_stats[0])
  # make_contour(G, age_diff[i], 'age', output_loc + 'age_' + str(time), True,
  #              'Age [sec]', vmin=age_stats[1], vmax=age_stats[0])
  # make_contour(G, agent_diff[i], 'agent', output_loc + 'locations_' + str(time), True,
  #              '# of Agents', vmin=agent_stats[1], vmax=agent_stats[0])
    # make_flow_plot(wn, flow_diff[i])

make_sector_plot(wn, demand, 'Demand [L]')
make_sector_plot(wn, age/3600, 'Age [hr]')
make_sector_plot(wn, pressure, 'Pressure [m]')

export_agent_loc(wn, agent)

seir['I'] = seir['I'] / 4606
seir.plot(y=['S', 'E', 'I', 'R', 'D', 'wfh'], xlabel='Time (days)',
          ylabel='Percent Population', legend=True)
# plt.plot(y='S', data=seir, label='Susceptible', use_index=True)
# plt.plot(y='E', data=seir, label='Exposed', use_index=True)
# plt.plot(y='I', data=seir, label='Infected', use_index=True)
# plt.plot(y='R', data=seir, label='Recovered', use_index=True)
# plt.plot(y='D', data=seir, label='Dead', use_index=True)
# plt.plot(y='wfh', data=seir, label='WFH', use_index=True)
plt.axvline(x=times[0]*3600, color='black')
plt.axvline(x=times[1]*3600, color='black')
# plt.axvline(x=times[2]*3600, color='black')
# plt.axvline(x=times[3]*3600, color='black')
plt.xlabel('Time (days)')
plt.ylabel('Percent Population')
plt.legend()
plt.savefig(output_loc + '/' + 'seir_wfh.png')
plt.close()
