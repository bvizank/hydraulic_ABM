import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import plotly.graph_objects as go
from scipy.interpolate import griddata
import networkx as nx
import os
import math
from utils import read_data

no_wfh_comp_dir = 'Output Files/no_pb_30/'
wfh_loc = 'Output Files/2022-12-15_12-09_all_pm_current_results/'
no_wfh_loc = 'Output Files/2022-10-27_17-53_no_pb_current_results/'
day200_loc = 'Output Files/2022-12-12_14-33_ppe_200Days_results/'
day400_loc = 'Output Files/2022-12-14_10-08_no_PM_400Days_results/'
read_list = ['seir', 'demand', 'pressure', 'age', 'agent', 'flow']
plt.rcParams['figure.figsize'] = [3.5, 3.5]
format = 'png'
publication = False
if publication:
    pub_loc = 'Output Files/publication_figures/'
    plt.rcParams['figure.dpi'] = 800
    format = 'eps'


def read_comp_data(loc, read_list, out_dict):
    for item in read_list:
        out_dict['avg_'+item] = pd.read_pickle(loc + 'avg_' + item + '.pkl')
        out_dict['sd_'+item] = pd.read_pickle(loc + 'sd_' + item + '.pkl')

    return out_dict


'''Import water network and data'''
inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()
# wfh = read_data(wfh_loc, read_list)
# no_wfh = read_data(no_wfh_loc, read_list)
wfh = dict()
comp_list = ['seir', 'demand', 'age']
no_wfh = read_comp_data(no_wfh_comp_dir, comp_list, wfh)
# days_200 = read_data(day200_loc, ['seir', 'demand', 'age'])
# days_400 = read_data(day400_loc, ['seir', 'demand', 'age'])
ind_nodes = [node for name, node in wn.junctions()
             if node.demand_timeseries_list[0].pattern_name == '3']


def calc_difference(data_time_1, data_time_2):
    '''Function to take the difference between two time points '''
    return (data_time_2 - data_time_1)


def calc_flow_diff(data):
    flow_data = dict()
    for (pipe, colData) in data.iteritems():
        curr_flow_changes = list()
        for i in range(len(colData)-1):
            if colData[(i+1)*3600] * colData[i*3600] < 0:
                curr_flow_changes.append(1)
            else:
                curr_flow_changes.append(0)
        flow_data[pipe] = curr_flow_changes

    output = pd.DataFrame(flow_data)

    # output = list()
    # for i in range(len(data_time_1)):
    #     if ('V' not in data_time_2.index[i]
    #             and 'TowerPipe' not in data_time_2.index[i]):
    #         if data_time_2[i] * data_time_1[i] < 0:
    #             output.append(1)
    #             print(f"Pipe {data_time_2.index[i]} has flow 2: {data_time_2[i]} and flow 1: {data_time_1[i]}")
    #         else:
    #             output.append(0)
    #     else:
    #         pass
    return output


def calc_distance(node1, node2):
    p1x, p1y = node1.coordinates
    p2x, p2y = node2.coordinates

    return math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)


def calc_industry_distance(wn):
    '''
    Function to calculate the distance to the nearest industrial node.
    '''
    all_nodes = [node for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].pattern_name != '3']

    ind_distances = dict()
    close_node = dict()
    for node in all_nodes:
        curr_node_dis = dict()
        for ind_node in ind_nodes:
            curr_node_dis[ind_node.name] = calc_distance(node, ind_node)
        # find the key with the min value, i.e. the node with the lowest distance
        ind_distances[node.name] = min(curr_node_dis.values())
        close_node[node.name] = min(curr_node_dis, key=curr_node_dis.get)

    return (ind_distances, close_node)


def calc_closest_node(wn):
    res_nodes = [node for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].base_value > 0
                 and node.demand_timeseries_list[0].pattern_name == '2']
    all_nodes = [node for name, node in wn.junctions()
                 if node.demand_timeseries_list[0].base_value > 0]

    closest_distances = dict()
    for node1 in res_nodes:
        curr_node_close = dict()
        for node2 in all_nodes:
            if node1.name != node2.name:
                curr_node_close[node2.name] = calc_distance(node1, node2)
        closest_distances[node1.name] = min(curr_node_close.values())

    return closest_distances


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

    x_mesh = np.linspace(np.min(x_coords), np.max(x_coords), int(np.sqrt(pts)))
    y_mesh = np.linspace(np.min(y_coords), np.max(y_coords), int(np.sqrt(pts)))
    [x,y] = np.meshgrid(x_mesh, y_mesh)

    z = griddata((x_coords, y_coords), data_list, (x, y), method='linear')
    x = np.matrix.flatten(x); #Gridded longitude
    y = np.matrix.flatten(y); #Gridded latitude
    z = np.matrix.flatten(z); #Gridded elevation

    if 'vmax' in plots:
        plt.scatter(x,y,1,z,vmin=plots['vmin'], vmax=plots['vmax'])
    else:
        plt.scatter(x,y,1,z,vmin=plots['vmin'])

    nx.draw_networkx(graph, pos=pos, with_labels=False, arrowstyle='-',
                     node_size=0)
    if label:
        plt.colorbar(label=label_val)

    if publication:
        plt.savefig(pub_loc + fig_name + '.' + format, format=format,
                    bbox_inches='tight')
    else:
        plt.savefig(fig_name + '.' + format, format=format, bbox_inches='tight')
    plt.close()

    ax = wntr.graphics.plot_network(wn, node_attribute='pressure',
                                    node_colorbar_label=label_val)

    # fig = go.Figure(data =
    #     go.Contour(
    #         z = z,
    #         x = x,
    #         y = y
    #     )
    # )

    # fig.write_image(fig_name + 'nodes' + '.png')


    nx.draw_networkx(graph, pos=pos, with_labels=False, arrowstyle='-',
                     node_size=10, node_color=data_list)
    if publication:
        plt.savefig(pub_loc + fig_name + 'nodes.' + format, format=format,
                    bbox_inches='tight')
    else:
        plt.savefig(fig_name + 'nodes.' + format, format=format,
                    bbox_inches='tight')
    plt.close()


def make_sector_plot(wn, data, ylabel, output_loc, op, fig_name,
                     data2=None, sd=None, sd2=None,
                     type=None, data_type='node', sub=False,
                     days=90):
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
    elif type == 'all':
        if data_type == 'node':
            nodes = [name for name, node in wn.junctions()
                     if node.demand_timeseries_list[0].base_value > 0]
        elif data_type == 'link':
            nodes = [name for name,link in wn.links() if 'V' not in name]

    if type is not None:
        y_data = getattr(data[nodes], op)(axis=1)
        x_values = np.array([x for x in np.arange(0, days, days/len(y_data))])
        if data2 is not None:
            wfh_data = getattr(data2[nodes], op)(axis=1)
            data = pd.DataFrame(data={'primary': y_data, 'wfh': wfh_data,
                                      't': x_values})
            data.rolling(24).mean().plot(x='t', y=['primary', 'wfh'],
                                         xlabel='Time (days)', ylabel=ylabel,
                                         legend=True)
            plt.legend(['Base', 'PM'])
        else:
            data = pd.DataFrame(data={'demand': y_data, 't': x_values})
            data.plot(x='t', y='demand', xlabel='Time (days)', ylabel=ylabel,
                      legend=False)
        if publication:
            output_loc = pub_loc
        plt.savefig(output_loc + fig_name + '.' + format, format=format,
                    bbox_inches='tight')
        plt.close()
    else:
        res_data = getattr(data[res_nodes], op)(axis=1)
        ind_data = getattr(data[ind_nodes], op)(axis=1)
        com_data = getattr(data[com_nodes], op)(axis=1)
        if sd is not None:
            res_sd = getattr(sd[res_nodes], op)(axis=1)
            ind_sd = getattr(sd[ind_nodes], op)(axis=1)
            com_sd = getattr(sd[com_nodes], op)(axis=1)
            sd = pd.DataFrame(data={'res': res_sd, 'com':com_sd,
                                    'ind': ind_sd})
            rolling_sd = sd.rolling(24).mean()
            print(rolling_sd)

        x_values = np.array([x for x in np.arange(0, days, days/len(res_data))])
        if not sub:
            cols = ['res', 'com', 'ind']
            data = pd.DataFrame(data={'res': res_data, 'com': com_data,
                                      'ind': ind_data, 't': x_values})
            rolling_data = data.rolling(24).mean()
            for i in range(3):
                plt.plot(rolling_data.t, rolling_data[cols[i]])
                if sd is not None:
                    print(rolling_data[cols[i]] - rolling_sd[cols[i]])
                    plt.fill_between(rolling_data.t,
                                     rolling_data[cols[i]] - rolling_sd[cols[i]],
                                     rolling_data[cols[i]] + rolling_sd[cols[i]],
                                     alpha=0.5)
            plt.xlabel('Time (days)')
            plt.ylabel(ylabel)
            plt.legend(['Residential', 'Commercial', 'Industrial'])
            if publication:
                # plt.gcf().set_size_inches(3.5, 3.5)
                output_loc = pub_loc
        else:
            res_data2 = getattr(data2[res_nodes], op)(axis=1)
            com_data2 = getattr(data2[com_nodes], op)(axis=1)
            ind_data2 = getattr(data2[ind_nodes], op)(axis=1)
            data = pd.DataFrame(data={'res': res_data, 'com': com_data,
                                      'ind': ind_data,
                                      't': x_values})
            data2 = pd.DataFrame(data={'res2': res_data2, 'com2': com_data2,
                                       'ind2': ind_data2,
                                       't': x_values})
            fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
            data.rolling(24).mean().plot(x='t', y=['res', 'com', 'ind'],
                                         xlabel='Time (days)', ylabel=ylabel,
                                         legend=False, ax=axes[0])
            data2.rolling(24).mean().plot(x='t', y=['res2', 'com2', 'ind2'],
                                          xlabel='Time (days)',
                                          legend=False, ax=axes[1])
            axes[0].legend(['Residential', 'Commercial', 'Industrial'])
            if publication:
                # plt.gcf().set_size_inches(7, 3.5)
                output_loc = pub_loc
                plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(output_loc + fig_name + '.' + format, format=format,
                    bbox_inches='tight')
        plt.close()


def make_flow_plot(wn, data):
    '''
    Function to make a plot showing the flow direction change
    '''
    ax = wntr.graphics.plot_network(wn, link_attribute=data,
                                    node_colorbar_label='Change in Direction')
    plt.show()


def export_agent_loc(wn, output_loc, locations):
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


def make_seir_plot(data, output_loc, input):
    ''' Function to make the seir plot with S, E, I, R, and WFH  '''
    data['I'] = data['I'] / 4606
    # seir.reset_index(inplace=True)
    # seir['t'] = seir['t'] / 3600 /24
    x_values = np.array([x for x in np.arange(0, 90, 90/len(data))])
    data['t_new'] = x_values
    data.plot(x='t_new', y=input, xlabel='Time (days)',
              ylabel='Percent Population', legend=True)
    # plt.axvline(x=times[0]/24, color='black')
    # plt.axvline(x=times[1]/24, color='black')
    plt.xlabel('Time (days)')
    plt.ylabel('Percent Population')
    plt.legend(['Susceptible', 'Exposed', 'Infected', 'Removed'])
    if publication:
        output_loc = pub_loc
    plt.savefig(output_loc + '/' + 'seir_wfh.' + format, format=format,
                bbox_inches='tight')
    plt.close()


def make_distance_plot(x, y, xlabel, ylabel, loc, name):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0,500])
    if publication:
        loc = pub_loc
    plt.savefig(loc + name + '.' + format, format=format, bbox_inches='tight')
    plt.close()


def calc_model_stats(wn, seir, age):
    '''
    Function for calculating comparison stats for decision models:
        - average water age at the end of the simluation
        - peak infection rate
        - final susceptible count
    '''
    nodes = [name for name, node in wn.junctions()
             if node.demand_timeseries_list[0].base_value > 0]
    age_data = getattr(age[nodes], 'mean')(axis=1)
    final_age = age_data[(len(age_data)-1)*3600]
    max_inf = seir.I.loc[int(round(seir.I.idxmax(), 0))]
    final_sus = seir.S[(len(seir.S)-1)*3600]

    return (final_age, max_inf/4606, final_sus)


# max_wfh = wfh['seir'].wfh.loc[int(wfh['seir'].wfh.idxmax())]
# times = []
# # times = times + [seir.wfh.searchsorted(max_wfh/4)+12]
# times = times + [wfh['seir'].wfh.searchsorted(max_wfh/2)]
# # print(seir.wfh.searchsorted(max_wfh/2))
# # times = times + [seir.wfh.searchsorted(max_wfh*3/4)+12]
# times = times + [wfh['seir'].wfh.searchsorted(max_wfh)]
# print(times)
# times_hour = [time % 24 for time in times]
# print(times_hour)

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
wfh_flow_diff = list()
no_wfh_flow_diff = list()

# for i, time in enumerate(times):
#     if time >= len(demand_wfh):
#         time = time - 1
#     print(time)
#     demand_diff.append(calc_difference(demand_wfh.iloc[times_hour[i]], demand_wfh.iloc[time]) * 1000)
#     demand_stats = check_stats(demand_diff[i], demand_stats)
#     make_contour(G, demand_diff[i], 'demand', wfh_loc + 'demand_' + str(time), True,
#                  'Demand [ML]', vmin=demand_stats[1], vmax=demand_stats[0])
#     pressure_diff.append(calc_difference(pressure_wfh.iloc[times_hour[i]], pressure_wfh.iloc[time]))
#     pressure_stats = check_stats(pressure_diff[i], pressure_stats)
#     make_contour(G, pressure_diff[i], 'pressure', wfh_loc + 'pressure_' + str(time), True,
#                  'Pressure [m]', vmin=pressure_stats[1], vmax=pressure_stats[0])
#     age_diff.append(calc_difference(age_wfh.iloc[times_hour[i]], age_wfh.iloc[time]))
#     age_stats = check_stats(age_diff[i], age_stats)
#     make_contour(G, age_diff[i], 'age', wfh_loc + 'age_' + str(time), True,
#                  'Age [sec]', vmin=age_stats[1], vmax=age_stats[0])
#     agent_diff.append(calc_difference(agent_wfh.iloc[times_hour[i]], agent_wfh.iloc[time]))
#     agent_stats = check_stats(agent_diff[i], agent_stats)
#     make_contour(G, agent_diff[i], 'agent', wfh_loc + 'locations_' + str(time), True,
#                  '# of Agents', vmin=agent_stats[1], vmax=agent_stats[0])
    # make_flow_plot(wn, flow_diff[i])

''' Sector plots '''

''' Flow direction change plots '''
# make_sector_plot(wn, calc_flow_diff(wfh['flow']), 'Number of Flow Changes', wfh_loc,
#                  'sum', 'flow_changes', calc_flow_diff(no_wfh['flow']), 'all', 'link')

''' Make demand plots for by sector with PM data '''
make_sector_plot(wn, no_wfh['avg_demand'], 'Demand (L)', no_wfh_comp_dir,
                 'sum', 'sum_demand', sd=no_wfh['sd_demand'])
# make_sector_plot(wn, wfh['demand'], 'Demand (L)', wfh_loc, 'max', 'max_demand')
# make_sector_plot(wn, wfh['demand'], 'Demand (L)', wfh_loc, 'mean', 'mean_demand')

''' Make age plot by sector for both base and PM '''
# make_sector_plot(wn, no_wfh['age']/3600, 'Age (hr)', wfh_loc, 'mean', 'mean_age',
#                  data2=wfh['age']/3600, sub=True)
make_sector_plot(wn, no_wfh['avg_age']/3600, 'Age (hr)', no_wfh_comp_dir,
                 'mean', 'mean_age', sd=no_wfh['sd_age']/3600)
# make_sector_plot(wn, days_200['age']/3600, 'Age (hr)', day200_loc, 'mean',
#                  'mean_age', days=200)
# make_sector_plot(wn, days_400['age']/3600, 'Age (hr)', day400_loc, 'mean',
#                  'mean_age', days=400)

''' Make age plot comparing base and PM '''
# make_sector_plot(wn, no_wfh['age']/3600, 'Age [hr]', wfh_loc, 'mean', 'mean_age_aggregate',
#                  wfh['age']/3600, type='all')

''' Make plots of aggregate demand data '''
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'sum', 'sum_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'max', 'max_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'mean', 'mean_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, pressure, 'Pressure (m)', pressure_wfh, type='all')

''' Export the agent locations '''
# export_agent_loc(wn, agent)

''' SEIR plot '''
# make_seir_plot(seir, [S, E, I, R])

''' Export comparison stats '''
only_wfh_loc = 'Output Files/2022-12-12_11-52_wfh_current_results/'
dine_loc = 'Output Files/2022-12-12_12-44_dine_current_results/'
grocery_loc = 'Output Files/2022-12-12_13-18_grocery_current_results/'
ppe_loc = 'Output Files/2022-12-15_13-00_ppe_current_results/'

only_wfh = read_data(only_wfh_loc, ['seir', 'age'])
dine = read_data(dine_loc, ['seir', 'age'])
grocery = read_data(grocery_loc, ['seir', 'age'])
ppe = read_data(ppe_loc, ['seir', 'age'])
print("WFH model stats: " + str(calc_model_stats(wn, only_wfh['seir'], only_wfh['age']/3600)))
print("Dine model stats: " + str(calc_model_stats(wn, dine['seir'], dine['age']/3600)))
print("Grocery model stats: " + str(calc_model_stats(wn, grocery['seir'], grocery['age']/3600)))
print("PPE model stats: " + str(calc_model_stats(wn, ppe['seir'], ppe['age']/3600)))
# print("All PM model stats: " + str(calc_model_stats(wn, wfh['seir'], wfh['age']/3600)))
print("No PM model stats: " + str(calc_model_stats(wn, no_wfh['avg_seir'], no_wfh['avg_age']/3600)))

# ind_distances, ind_closest = calc_industry_distance(wn)
# age_values = list()
# curr_age_values = wfh['age'].iloc[len(wfh['age'])-1]/3600
# for age in curr_age_values.items():
#     if age[0] in ind_distances.keys():
#         age_values.append(age[1])
# make_distance_plot(ind_distances.values(), age_values,
#                    'Distance (m)', 'Age (hr)', wfh_loc, 'age_ind_distance')

# closest_distances = calc_closest_node(wn)
# age_values = list()
# curr_age_values = wfh['age'].iloc[len(wfh['age'])-1]/3600
# for age in curr_age_values.items():
#     if age[0] in closest_distances.keys():
#         age_values.append(age[1])
# make_distance_plot(closest_distances.values(), age_values, 'Distance (m)',
#                    'Age (hr)', wfh_loc, 'age_closest_node')
