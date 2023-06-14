import wntr
import numpy as np
import pandas as pd
import copy
import utils as ut
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import binned_statistic
import networkx as nx
# import os
import math


no_wfh_comp_dir = 'Output Files/30_no_pm/'
wfh_comp_dir = 'Output Files/30_all_pm/'
# wfh_loc = 'Output Files/2022-12-15_12-09_all_pm_current_results/'
# no_wfh_loc = 'Output Files/2022-10-27_17-53_no_pb_current_results/'
day200_loc = 'Output Files/2022-12-12_14-33_ppe_200Days_results/'
day400_loc = 'Output Files/2022-12-14_10-08_no_PM_400Days_results/'
plt.rcParams['figure.figsize'] = [3.5, 3.5]
plt.rcParams['figure.dpi'] = 500
format = 'png'
error = 'se'
publication = True
if publication:
    pub_loc = 'Output Files/publication_figures/'
    plt.rcParams['figure.dpi'] = 800
    format = 'pdf'

prim_colors = ['#253494', '#2c7fb8', '#41b6c4', '#a1dab4', '#f1f174']
# sec_colors = ['#454545', '#929292', '#D8D8D8']

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=prim_colors)
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = ['serif']
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 0.7
plt.rcParams['ytick.major.width'] = 0.7
plt.rcParams['xtick.major.size'] = 3.0
plt.rcParams['ytick.major.size'] = 3.0


'''Import water network and data'''
inp_file = 'Input Files/MICROPOLIS_v1_inc_rest_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.to_graph()
# wfh = read_data(wfh_loc, read_list)
# no_wfh = read_data(no_wfh_loc, read_list)
# comp_list = ['seir', 'demand', 'age', 'flow']
comp_list = ['seir_data', 'demand', 'age', 'flow', 'ppe', 'cov_ff', 'cov_pers',
             'wfh', 'dine', 'groc', 'ppe']
wfh = ut.read_comp_data(wfh_comp_dir, comp_list)
no_wfh = ut.read_comp_data(no_wfh_comp_dir, comp_list)
# days_200 = read_data(day200_loc, ['seir', 'demand', 'age'])
# days_400 = read_data(day400_loc, ['seir', 'demand', 'age'])
# print(wfh['var_seir_data'])
# print(wfh['avg_age'])
# print(wfh['avg_ppe'])


def calc_difference(data_time_1, data_time_2):
    '''Function to take the difference between two time points '''
    return (data_time_2 - data_time_1)


def calc_flow_diff(data, hours):
    flow_data = dict()
    flow_changes_sum = dict()
    for (pipe, colData) in data.items():
        if 'MA' in pipe:
            curr_flow_changes = list()
            for i in range(len(colData)-1):
                if colData[(i+1)*3600] * colData[i*3600] < 0:
                    curr_flow_changes.append(1)
                else:
                    curr_flow_changes.append(0)
        # print(curr_flow_changes[0:hours])
            flow_changes_sum[pipe] = sum(curr_flow_changes[0:hours])/24
            flow_data[pipe] = curr_flow_changes

    # output = pd.DataFrame(flow_data)
    # change_sum = pd.Series(flow_changes_sum)

    return flow_data, flow_changes_sum
    # return output, change_sum


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


def make_sector_plot(wn, data, ylabel, op, fig_name,
                     data2=None, sd=None, sd2=None,
                     type=None, data_type='node', sub=False,
                     days=90):
    '''
    Function to plot the average data for a given sector
    Sectors include: residential, commercial, industrial
    '''
    output_loc = 'Output Files/png_figures/'
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
        if sd is not None:
            sd = getattr(sd[nodes], op)(axis=1)
        x_values = np.array([x for x in np.arange(0, days, days/len(y_data))])
        if data2 is not None:
            cols = ['primary', 'wfh']
            y_data2 = getattr(data2[nodes], op)(axis=1)
            plot_data = pd.DataFrame(data={'primary': y_data, 'wfh': y_data2})
            rolling_data = plot_data.rolling(24).mean()
            if sd is not None:
                sd2 = getattr(sd2[nodes], op)(axis=1)
                plot_sd = pd.DataFrame(data={'primary': sd, 'wfh': sd2})
                rolling_sd = plot_sd.rolling(24).mean()
            for i in range(2):
                plt.plot(x_values, rolling_data[cols[i]], color='C'+str(i*2))
            if sd is not None:
                for i in range(2):
                    plt.fill_between(x_values,
                                     rolling_data[cols[i]] - rolling_sd[cols[i]],
                                     rolling_data[cols[i]] + rolling_sd[cols[i]],
                                     color='C'+str(i*2), alpha=0.5)
            plt.legend(['Base', 'PM'])
        else:
            data = pd.DataFrame(data={'demand': y_data, 't': x_values})
            data.plot(x='t', y='demand', xlabel='Time (days)', ylabel=ylabel,
                      legend=False)
        if publication:
            output_loc = pub_loc

        plt.xlabel('Time (days)')
        plt.ylabel(ylabel)
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
            roll_sd = sd.rolling(24).mean()

        x_values = np.array([x for x in np.arange(0, days, days/len(res_data))])
        cols = ['res', 'com', 'ind']
        if not sub:
            data = pd.DataFrame(data={'res': res_data, 'com': com_data,
                                      'ind': ind_data})
            rolling_data = data.rolling(24).mean()
            for i in range(3):
                plt.plot(x_values, rolling_data[cols[i]], color='C'+str(i*2))
            if sd is not None:
                for i in range(3):
                    plt.fill_between(x_values,
                                     rolling_data[cols[i]] - roll_sd[cols[i]],
                                     rolling_data[cols[i]] + roll_sd[cols[i]],
                                     alpha=0.5, color='C'+str(i*2))
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
                                      'ind': ind_data})
            data2 = pd.DataFrame(data={'res': res_data2, 'com': com_data2,
                                       'ind': ind_data2})
            if sd2 is not None:
                res_sd2 = getattr(sd2[res_nodes], op)(axis=1)
                com_sd2 = getattr(sd2[com_nodes], op)(axis=1)
                ind_sd2 = getattr(sd2[ind_nodes], op)(axis=1)
                sd2 = pd.DataFrame(data={'res': res_sd2, 'com': com_sd2,
                                         'ind': ind_sd2})
                roll_sd2 = sd2.rolling(24).mean()
            fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
            roll_data = data.rolling(24).mean()
            roll_data2 = data2.rolling(24).mean()
            for i in range(3):
                axes[0].plot(x_values, roll_data[cols[i]], color='C'+str(i*2))
                axes[1].plot(x_values, roll_data2[cols[i]], color='C'+str(i*2))
            if sd is not None:
                for i in range(3):
                    axes[0].fill_between(x_values,
                                         roll_data[cols[i]] - roll_sd[cols[i]],
                                         roll_data[cols[i]] + roll_sd[cols[i]],
                                         alpha=0.5, color='C'+str(i*2))
                    axes[1].fill_between(x_values,
                                         roll_data2[cols[i]] - roll_sd2[cols[i]],
                                         roll_data2[cols[i]] + roll_sd2[cols[i]],
                                         alpha=0.5, color='C'+str(i*2))
                # elif sd is not None and sd2 is None:
                #     print('Missing standard deviation for second dataset')
            axes[0].legend(['Residential', 'Commercial', 'Industrial'])
            if publication:
                # plt.gcf().set_size_inches(7, 3.5)
                output_loc = pub_loc
            axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                         transform=axes[0].transAxes)
            axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                         transform=axes[1].transAxes)
            fig.supxlabel('Time (days)', y=-0.03)
            fig.supylabel(ylabel, x=0.04)
            plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(output_loc + fig_name + '.' + format, format=format,
                    bbox_inches='tight')
        plt.close()


def make_avg_plot(data, sd, cols, xlabel, ylabel, fig_name, x_values,
                  logx=False, sub=False, data2=None, sd2=None):
    '''
    Function to plot data with error.

    Parameters:
        data (pd.DataFrame): data to be plotted
        sd (pd.DataFrame): error of data to be plotted
        xlabel (string): x label for figure
        ylabel (string): y label for figure
        fig_name (string): save name for figure
        x_values (list): x values for the plot
    '''
    if publication:
        output_loc = pub_loc
    else:
        output_loc = 'Output Files/png_figures/'

    if not sub:
        ''' Plot a single figure with the input data '''
        # plot each column of data
        for i, col in enumerate(cols):
            plt.plot(x_values, data[col], color='C'+str(i*2))

        #  need to separate so that the legend fills correctly
        for i, col in enumerate(cols):
            plt.fill_between(x_values, data[col] - sd[col],
                             data[col] + sd[col],
                             color='C'+str(i*2), alpha=0.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(cols)
        if logx:
            plt.xscale('log')
    else:
        ''' Sub refers to a subplot with two columns of figures '''
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        for i, col in enumerate(cols):
            axes[0].plot(x_values, data[col], color='C'+str(i*2))
            axes[1].plot(x_values, data2[col], color='C'+str(i*2))

        # again need to separate sd so that legend fills correctly
        for i, col in enumerate(cols):
            axes[0].fill_between(x_values, data[col] - sd[col],
                                 data[col] + sd[col],
                                 color='C'+str(i*2), alpha=0.5)
            axes[1].fill_between(x_values, data2[col] - sd2[col],
                                 data2[col] + sd2[col],
                                 color='C'+str(i*2), alpha=0.5)
        axes[0].legend(cols)
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel(xlabel, y=-0.03)
        fig.supylabel(ylabel, x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

    plt.savefig(output_loc + fig_name + '.' + format, format=format,
                bbox_inches='tight')
    plt.close()


def make_flow_plot(change_data, sum_data, percent, dir, legend_text,
                   title, change_data2=None, sum_data2=None, days=90,
                   ax=None):
    '''
    Function to make a plot showing the flow direction change
    '''

    # filter out zero values from sum_data
    sum_data = sum_data[sum_data != 0]
    if sum_data2 is not None:
        sum_data2 = sum_data2[sum_data2 != 0]

    roll_change = change_data.rolling(24).mean()
    percentiles = sum_data.quantile(percent)
    if change_data2 is not None:
        roll_change2 = change_data2.rolling(24).mean()
        percentiles2 = sum_data2.quantile(percent)

    # print(percentiles)
    x_values = np.array([x for x in np.arange(0, days, days/len(roll_change))])

    if dir == 'top':
        y_1 = roll_change[sum_data[sum_data > percentiles].index]
        y_2 = roll_change2[sum_data2[sum_data2 > percentiles2].index]
    elif dir == 'bottom':
        y_1 = roll_change[sum_data[sum_data < percentiles].index]
        y_2 = roll_change2[sum_data2[sum_data2 < percentiles2].index]
    elif dir == 'middle':
        pipes = sum_data[sum_data > percentiles[percent[0]]]
        pipes = sum_data[sum_data < percentiles[percent[1]]]
        y_1 = roll_change[pipes.index]
        y_2 = roll_change2[pipes.index]

    if ax is None:
        plt.plot(x_values, y_1.mean(axis=1), color='C'+str(0))
        plt.plot(x_values, y_2.mean(axis=1), color='C'+str(2))
        plt.xlabel('Time (days)')
        plt.ylabel('Daily Average Flow Changes')
        plt.legend(legend_text)
        if publication:
            loc = pub_loc
        else:
            loc = 'Output Files/png_figures/'
        plt.savefig(loc + title + '.' + format, format=format, bbox_inches='tight')
        plt.close()
    else:
        ax.plot(x_values, y_1.mean(axis=1), color=prim_colors[0])
        ax.plot(x_values, y_2.mean(axis=1), color=prim_colors[2])


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


def make_seir_plot(data, input, leg_text, title,
                   data2=None, sd=None, sd2=None, sub=False):
    ''' Function to make the seir plot with the input columns '''
    in_data = copy.deepcopy(data)
    in_data2 = copy.deepcopy(data2)
    in_data = in_data * 100
    in_data2 = in_data2 * 100

    if sd is not None:
        in_sd = copy.deepcopy(sd)
        in_sd2 = copy.deepcopy(sd2)
        in_sd = in_sd * 100
        in_sd2 = in_sd2 * 100

    x_values = np.array([x for x in np.arange(0, 90, 90/len(data))])

    if sub:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    for item in input:
        if sub:
            axes[0].plot(x_values, in_data[item])
            axes[1].plot(x_values, in_data2[item])
        else:
            plt.plot(in_data['t_new'], in_data[item])
    if sd is not None:
        for item in input:
            if sub:
                axes[0].fill_between(x_values, in_data[item]-in_sd[item],
                                     in_data[item]+in_sd[item], alpha=0.5)
                axes[1].fill_between(x_values, in_data2[item]-in_sd2[item],
                                     in_data2[item]+in_sd2[item], alpha=0.5)
            else:
                plt.fill_between(x_values, in_data[item] - in_sd[item],
                                 in_data[item] + in_sd[item], alpha=0.5)
    # plt.axvline(x=times[0]/24, color='black')
    # plt.axvline(x=times[1]/24, color='black')
    if sub:
        plt.gcf().set_size_inches(7, 3.5)
        axes[0].legend(leg_text, loc='upper left')
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Percent Population', x=0.04)
    else:
        plt.legend(leg_text, loc='upper left')
        plt.ylabel('Percent Population')
        plt.xlabel('Time (days)')

    plt.ylim(0, 100)

    if publication:
        output_loc = pub_loc
    else:
        output_loc = 'Output Files/png_figures/'
    plt.savefig(output_loc + '/' + 'seir_' + title + '_' + error + '.' + format,
                format=format, bbox_inches='tight')
    plt.close()


def make_distance_plot(x, y1, y2, sd1, sd2, xlabel, ylabel, name, data_names):
    '''
    Make scatter plot plus binned levels of input data. Accepts one x
    vector and two y vectors.
    '''
    mean_y1 = binned_statistic(x, y1, statistic='mean',
                               bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    mean_y2 = binned_statistic(x, y2, statistic='mean',
                               bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500])

    sd_y1 = binned_statistic(x, sd1, statistic='mean',
                             bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    sd_y2 = binned_statistic(x, sd2, statistic='mean',
                             bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    # print(mean_y1.bin_edges)

    bin_names = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500',
                 '2500-3000', '3000-3500']

    data_dict = {data_names[0]: mean_y1.statistic,
                 data_names[1]: mean_y2.statistic}

    sd_dict = {data_names[0]: sd_y1.statistic,
               data_names[1]: sd_y2.statistic}

    bar_x = np.arange(len(bin_names))
    width = 0.25  # width of the bars
    multiplier = 0  # iterator

    # plt.figure()
    # plt.plot(x, y, '.', c=prim_colors[0], lw=2)
    # if y2 is not None:
        # plt.plot(x, y2, '.', c=prim_colors[2], lw=2)
        # plt.hlines(mean_y2.statistic, mean_y2.bin_edges[:-1],
                   # mean_y2.bin_edges[1:], colors=prim_colors[2], lw=3)
    # plt.hlines(mean_y.statistic, mean_y.bin_edges[:-1],
               # mean_y.bin_edges[1:], colors=prim_colors[0], lw=3)
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data_dict.items():
        # print(measurement)
        offset = width * multiplier
        ax.bar(bar_x + offset, measurement, width, label=attribute,
               color='C'+str(multiplier*2), yerr=sd_dict[attribute],
               error_kw=dict(lw=0.5, capsize=2, capthick=0.5))
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.legend()
    ax.set_xticks(bar_x + width, bin_names, rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if publication:
        loc = pub_loc
    else:
        loc = 'Output Files/png_figures/'

    plt.savefig(loc + name + '.' + format, format=format, bbox_inches='tight')
    plt.close()


def make_heatmap(data, xlabel, ylabel, name, vmax):
    ''' heatmap plot of all agents '''
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect=0.55, vmax=vmax)  # for 100 agents: 0.03, for 1000 agents: 0.003
    ax.figure.colorbar(im, ax=ax)
    plt.xlim(1, data.shape[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x_tick_labels = [0, 20, 40, 60, 80]
    ax.set_xticks([i*24 for i in x_tick_labels])
    ax.set_xticklabels(x_tick_labels)

    if publication:
        loc = pub_loc
    else:
        loc = 'Output Files/png_figures/'

    plt.savefig(loc + name + '.' + format,
                format=format,
                bbox_inches='tight')
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


# print(wfh['avg_seir_data'])
# index_vals = wfh['avg_seir_data'].index
# for i, item in enumerate(wfh['avg_seir_data'].wfh):
#     print(index_vals[i])
#     print(item)

max_wfh = wfh['avg_seir_data'].wfh.loc[int(wfh['avg_seir_data'].wfh.idxmax())]
times = []
# # times = times + [seir.wfh.searchsorted(max_wfh/4)+12]
times = times + [wfh['avg_seir_data'].wfh.searchsorted(max_wfh/10)]
times = times + [wfh['avg_seir_data'].wfh.searchsorted(max_wfh/2)]
# # print(seir.wfh.searchsorted(max_wfh/2))
# # times = times + [seir.wfh.searchsorted(max_wfh*3/4)+12]
times = times + [wfh['avg_seir_data'].wfh.searchsorted(max_wfh)]
# print(times)
times_hour = [time % 24 for time in times]
# print(times_hour)


def check_stats(new_list, old_stats):
    if new_list.max() > old_stats[0]:
        old_stats[0] = new_list.max()
    if new_list.min() < old_stats[1]:
        old_stats[1] = new_list.min()

    return old_stats


# demand_stats = [0,0]
# pressure_stats = [0,0]
# age_stats = [0,0]
# agent_stats = [0,0]
# demand_diff = list()
# pressure_diff = list()
# age_diff = list()
# agent_diff = list()
# wfh_flow_diff = list()
# no_wfh_flow_diff = list()

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
wfh_flow_change, wfh_flow_sum = calc_flow_diff(wfh['avg_flow'], times[len(times)-1])
no_wfh_flow_change, no_wfh_flow_sum = calc_flow_diff(no_wfh['avg_flow'], times[len(times)-1])

ax = wntr.graphics.plot_network(wn, link_attribute=wfh_flow_sum,
                                link_colorbar_label='Flow Changes',
                                node_size=0, link_width=2)
if publication:
    loc = pub_loc
else:
    loc = 'Output Files/png_figures/'

plt.savefig(loc + 'flow_network_all_pm.' + format, format=format,
            bbox_inches='tight')
plt.close()

ax = wntr.graphics.plot_network(wn, link_attribute=no_wfh_flow_sum,
                                node_size=0, link_width=2, add_colorbar=False)
if publication:
    loc = pub_loc
else:
    loc = 'Output Files/png_figures/'

plt.savefig(loc + 'flow_network_no_pm.' + format, format=format,
            bbox_inches='tight')
plt.close()
# fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)

# make_flow_plot(no_wfh_flow_change, no_wfh_flow_sum, 0.8, 'top', ['Base', 'PM'],
#                'top10_flow_changes', wfh_flow_change,
#                wfh_flow_sum, ax=axes[0])
# make_flow_plot(no_wfh_flow_change, no_wfh_flow_sum, 0.2, 'bottom', ['Base', 'PM'],
#                'bottom10_flow_changes', wfh_flow_change,
#                wfh_flow_sum, ax=axes[2])
# make_flow_plot(no_wfh_flow_change, no_wfh_flow_sum, [0.2, 0.8], 'middle', ['Base', 'PM'],
#                'middle80_flow_changes', wfh_flow_change,
#                wfh_flow_sum, ax=axes[1])
# # max_flow_change = wfh_flow_sum.loc[int(wfh_flow_sum.idxmax())]

# plt.gcf().set_size_inches(6, 3.5)
# fig.supxlabel('Time (days)', y=-0.05)
# fig.supylabel('Daily Average Flow Changes')
# axes[0].text(0.5, -0.12, "Top 20%", size=12, ha="center",
#              transform=axes[0].transAxes)
# axes[1].text(0.5, -0.12, "Middle 60%", size=12, ha="center",
#              transform=axes[1].transAxes)
# axes[2].text(0.5, -0.12, "Bottom 20%", size=12, ha="center",
#              transform=axes[2].transAxes)
# # plt.xlabel('Time (days)')
# # plt.ylabel('Daily Average Flow Changes')
# axes[0].legend(['Base', 'PM'], loc='lower left')
# if publication:
#     loc = pub_loc
# else:
#     loc = 'Output Files/png_figures/'
# plt.savefig(loc + 'flow_change_mid60.' + format, format=format, bbox_inches='tight')
# plt.close()

# make_flow_plot(no_wfh_flow_change, no_wfh_flow_sum, 0, 'top', ['Base', 'PM'],
#                'all_flow_changes', wfh_flow_change,
#                wfh_flow_sum)

''' Make demand plots for by sector with PM data '''
# make lists of sector nodes
res_nodes = [name for name, node in wn.junctions()
             if node.demand_timeseries_list[0].pattern_name == '2']
ind_nodes = [name for name, node in wn.junctions()
             if node.demand_timeseries_list[0].pattern_name == '3']
com_nodes = [name for name, node in wn.junctions()
             if node.demand_timeseries_list[0].pattern_name == '4' or
             node.demand_timeseries_list[0].pattern_name == '5' or
             node.demand_timeseries_list[0].pattern_name == '6']

# define the columns of the input data and the x_values
cols = ['Residential', 'Industrial', 'Commercial']
x_values = np.array([x for x in np.arange(0, 90, 90/len(wfh['avg_demand']))])

# collect demands
res_dem = wfh['avg_demand'][res_nodes]
com_dem = wfh['avg_demand'][com_nodes]
ind_dem = wfh['avg_demand'][ind_nodes]

res_sd = wfh['var_demand'][res_nodes]
com_sd = wfh['var_demand'][com_nodes]
ind_sd = wfh['var_demand'][ind_nodes]

# make input data and sd
sector_dem = pd.concat([res_dem.sum(axis=1).rolling(24).mean(),
                        com_dem.sum(axis=1).rolling(24).mean(),
                        ind_dem.sum(axis=1).rolling(24).mean()],
                       axis=1, keys=cols)
sector_dem_var = pd.concat([res_sd.sum(axis=1).rolling(24).mean(),
                            com_sd.sum(axis=1).rolling(24).mean(),
                            ind_sd.sum(axis=1).rolling(24).mean()],
                           axis=1, keys=cols)

sector_dem_err = ut.calc_error(sector_dem_var, error)

# plot demand by sector
make_avg_plot(sector_dem, sector_dem_err, cols, 'Time (days)', 'Demand (L)',
              'sum_demand_' + error, x_values)

''' Make age plot by sector for both base and PM '''
res_age_all_pm = wfh['avg_age'][res_nodes].mean(axis=1)
com_age_all_pm = wfh['avg_age'][com_nodes].mean(axis=1)
ind_age_all_pm = wfh['avg_age'][ind_nodes].mean(axis=1)

res_sd_all_pm = wfh['var_age'][res_nodes].mean(axis=1)
com_sd_all_pm = wfh['var_age'][com_nodes].mean(axis=1)
ind_sd_all_pm = wfh['var_age'][ind_nodes].mean(axis=1)

res_age_no_pm = no_wfh['avg_age'][res_nodes].mean(axis=1)
com_age_no_pm = no_wfh['avg_age'][com_nodes].mean(axis=1)
ind_age_no_pm = no_wfh['avg_age'][ind_nodes].mean(axis=1)

res_sd_no_pm = no_wfh['var_age'][res_nodes].mean(axis=1)
com_sd_no_pm = no_wfh['var_age'][com_nodes].mean(axis=1)
ind_sd_no_pm = no_wfh['var_age'][ind_nodes].mean(axis=1)

# make input data and sd
all_pm_age = pd.concat([res_age_all_pm.rolling(24).mean(),
                        com_age_all_pm.rolling(24).mean(),
                        ind_age_all_pm.rolling(24).mean()],
                       axis=1, keys=cols)
all_pm_age_var = pd.concat([res_sd_all_pm.rolling(24).mean(),
                            com_sd_all_pm.rolling(24).mean(),
                            ind_sd_all_pm.rolling(24).mean()],
                           axis=1, keys=cols)
all_pm_age_err = ut.calc_error(all_pm_age_var, error)
non_pm_age = pd.concat([res_age_no_pm.rolling(24).mean(),
                        com_age_no_pm.rolling(24).mean(),
                        ind_age_no_pm.rolling(24).mean()],
                       axis=1, keys=cols)
non_pm_age_var = pd.concat([res_sd_no_pm.rolling(24).mean(),
                            com_sd_no_pm.rolling(24).mean(),
                            ind_sd_no_pm.rolling(24).mean()],
                           axis=1, keys=cols)
non_pm_age_err = ut.calc_error(non_pm_age_var, error)
make_avg_plot(non_pm_age / 3600, non_pm_age_err / 3600,
              cols, 'Time (days)', 'Age (hr)', 'mean_age_' + error,
              x_values, data2=all_pm_age / 3600, sd2=all_pm_age_err / 3600,
              sub=True)
# make_sector_plot(wn, no_wfh['avg_age']/3600, 'Age (hr)', no_wfh_comp_dir,
#                  'mean', 'mean_age', sd=no_wfh['var_age']/3600)
# make_sector_plot(wn, days_200['age']/3600, 'Age (hr)', day200_loc, 'mean',
#                  'mean_age', days=200)
# make_sector_plot(wn, days_400['age']/3600, 'Age (hr)', day400_loc, 'mean',
#                  'mean_age', days=400)

''' Make age plot comparing base and PM '''
make_sector_plot(wn, no_wfh['avg_age'] / 3600, 'Age (hr)', 'mean',
                 'mean_age_aggregate_' + error, wfh['avg_age'] / 3600,
                 sd=ut.calc_error(no_wfh['var_age'] / 3600, error),
                 sd2=ut.calc_error(wfh['var_age'] / 3600, error), type='all')

''' Make plots of aggregate demand data '''
make_sector_plot(wn, no_wfh['avg_demand'], 'Demand (L)', 'sum',
                 'sum_demand_aggregate_' + error, wfh['avg_demand'], type='all',
                 sd=ut.calc_error(no_wfh['var_demand'], error),
                 sd2=ut.calc_error(wfh['var_demand'], error))
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'max', 'max_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'mean', 'mean_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, pressure, 'Pressure (m)', pressure_wfh, type='all')

''' Export the agent locations '''
# export_agent_loc(wn, agent)

''' SEIR plot '''
make_seir_plot(no_wfh['avg_seir_data'], ['S', 'E', 'I', 'R', 'wfh'],
               leg_text=['Susceptible', 'Exposed', 'Infected', 'Removed', 'WFH'],
               title='combined', data2=wfh['avg_seir_data'],
               sd=ut.calc_error(no_wfh['var_seir_data'], error),
               sd2=ut.calc_error(wfh['var_seir_data'], error),
               sub=True)

''' Export comparison stats '''
# only_wfh_loc = 'Output Files/30_wfh/'
# dine_loc = 'Output Files/30_dine/'
# grocery_loc = 'Output Files/30_grocery/'
# ppe_loc = 'Output Files/30_ppe/'

# only_wfh = ut.read_comp_data(only_wfh_loc, ['seir_data', 'age'])
# dine = ut.read_comp_data(dine_loc, ['seir_data', 'age'])
# grocery = ut.read_comp_data(grocery_loc, ['seir_data', 'age'])
# ppe = ut.read_comp_data(ppe_loc, ['seir_data', 'age'])
# print("WFH model stats: " + str(calc_model_stats(wn, only_wfh['avg_seir_data'], only_wfh['avg_age']/3600)))
# print("Dine model stats: " + str(calc_model_stats(wn, dine['avg_seir_data'], dine['avg_age']/3600)))
# print("Grocery model stats: " + str(calc_model_stats(wn, grocery['avg_seir_data'], grocery['avg_age']/3600)))
# print("PPE model stats: " + str(calc_model_stats(wn, ppe['avg_seir_data'], ppe['avg_age']/3600)))
# print("All PM model stats: " + str(calc_model_stats(wn, wfh['avg_seir_data'], wfh['avg_age']/3600)))
# print("No PM model stats: " + str(calc_model_stats(wn, no_wfh['avg_seir_data'], no_wfh['avg_age']/3600)))

ind_nodes = [node for name, node in wn.junctions()
             if node.demand_timeseries_list[0].pattern_name == '3']
ind_distances, ind_closest = calc_industry_distance(wn)
pm_age_values = list()
no_pm_age_values = list()
pm_age_sd = list()
no_pm_age_sd = list()
pm_curr_age_values = wfh['avg_age'].iloc[len(wfh['avg_age'])-1]/3600
no_pm_curr_age_values = no_wfh['avg_age'].iloc[len(no_wfh['avg_age'])-1]/3600
pm_curr_age_sd = wfh['var_age'].iloc[len(wfh['var_age'])-1]/3600
no_pm_curr_age_sd = no_wfh['var_age'].iloc[len(no_wfh['var_age'])-1]/3600
# print(pm_curr_age_values)
''' Collect the age for each residential node '''
for i, age in pm_curr_age_values.items():
    no_pm_age = no_pm_curr_age_values[i]
    pm_sd = pm_curr_age_sd[i]
    no_pm_sd = no_pm_curr_age_sd[i]
    # if the node is in the distance list, add the age
    # otherwise delete the node from the distance list
    if i in ind_distances.keys():
        if age < 500:
            pm_age_values.append(age)
            no_pm_age_values.append(no_pm_age)
            pm_age_sd.append(pm_sd)
            no_pm_age_sd.append(no_pm_sd)
        else:
            del ind_distances[i]

dist_values = [i for i in ind_distances.values()]
make_distance_plot(dist_values, no_pm_age_values, pm_age_values,
                   ut.calc_error(no_pm_age_sd, error),
                   ut.calc_error(pm_age_sd, error),
                   'Distance (m)', 'Age (hr)', 'pm_age_ind_distance',
                   ['Base', 'PM'])

# closest_distances = calc_closest_node(wn)
# age_values = list()
# curr_age_values = wfh['age'].iloc[len(wfh['age'])-1]/3600
# for age in curr_age_values.items():
#     if age[0] in closest_distances.keys():
#         age_values.append(age[1])
# make_distance_plot(closest_distances.values(), age_values, 'Distance (m)',
#                    'Age (hr)', wfh_loc, 'age_closest_node')


''' Make agent state variable plots '''
all_pm_sv = ut.read_data('Output Files/30_all_pm/2023-05-30_15-29_0_results/',
                         ['cov_pers', 'cov_ff', 'media'])
no_pm_sv = ut.read_data('Output Files/30_no_pm/2023-05-26_08-33_0_results/',
                        ['cov_pers', 'cov_ff', 'media'])

agent = '124'
cols = ['Personal', 'Friends-Family', 'Media']
data = pd.concat([all_pm_sv['cov_pers'][agent],
                  all_pm_sv['cov_ff'][agent],
                  all_pm_sv['media'][agent]],
                 axis=1, keys=cols)
plt.plot(np.delete(x_values, 0), data)
plt.xlabel('Time (day)')
plt.ylabel('Value')
loc = 'Output Files/png_figures/'

plt.savefig(loc + 'state_variable_plot.' + format, format=format,
            bbox_inches='tight')
plt.close()

make_heatmap(all_pm_sv['cov_ff'].T,
             'Time (day)', 'Agent', 'ff_heatmap_all_pm', 6)
make_heatmap(no_pm_sv['cov_ff'].T,
             'Time (day)', 'Agent', 'ff_heatmap_no_pm', 6)

''' State variable scenario comparisons '''
data = pd.concat([no_wfh['avg_cov_ff'].mean(axis=1),
                  wfh['avg_cov_ff'].mean(axis=1)],
                 axis=1, keys=['Base', 'PM'])
var = pd.concat([no_wfh['var_cov_ff'].mean(axis=1),
                 wfh['var_cov_ff'].mean(axis=1)],
                axis=1, keys=['Base', 'PM'])
err = ut.calc_error(var, error)
make_avg_plot(data, err, ['Base', 'PM'],
              'Time (day)', 'Average Value', 'ff_avg',
              np.delete(x_values, 0))
data = pd.concat([no_wfh['avg_cov_pers'].mean(axis=1),
                  wfh['avg_cov_pers'].mean(axis=1)],
                 axis=1, keys=['Base', 'PM'])
var = pd.concat([no_wfh['var_cov_pers'].mean(axis=1),
                 wfh['var_cov_pers'].mean(axis=1)],
                axis=1, keys=['Base', 'PM'])
err = ut.calc_error(var, error)
make_avg_plot(data, err, ['Base', 'PM'],
              'Time (day)', 'Average Value', 'pers_avg',
              np.delete(x_values, 0))

''' BBN decisions scenario comparisons '''
cols = ['WFH', 'Dine out less', 'Grocery shop less', 'Wear PPE']
data = pd.concat([wfh['avg_wfh'].mean(axis=1),
                  wfh['avg_dine'].mean(axis=1),
                  wfh['avg_groc'].mean(axis=1),
                  wfh['avg_ppe'].mean(axis=1)],
                 axis=1, keys=cols)
var = pd.concat([wfh['var_wfh'].mean(axis=1),
                 wfh['var_dine'].mean(axis=1),
                 wfh['var_groc'].mean(axis=1),
                 wfh['var_ppe'].mean(axis=1)],
                axis=1, keys=cols)
err = ut.calc_error(var, error)
make_avg_plot(data, err, cols,
              'Time (day)', 'Average Value', 'bbn_decision_all_pm',
              np.delete(x_values, 0))
