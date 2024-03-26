import math
import wntr
import numpy as np
import pandas as pd
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import utils as ut


class BaseGraphics:
    def __init__(self):
        pass

    def calc_difference(self, data_time_1, data_time_2):
        '''Function to take the difference between two time points '''
        return (data_time_2 - data_time_1)

    def get_times(self, pm):
        max_pm = pm['avg_seir_data'].wfh.loc[int(pm['avg_seir_data'].wfh.idxmax())]
        times = []
        times = times + [pm['avg_seir_data'].wfh.searchsorted(max_pm/10)]
        times = times + [pm['avg_seir_data'].wfh.searchsorted(max_pm/2)]
        times = times + [pm['avg_seir_data'].wfh.searchsorted(max_pm)]
        self.times_hour = [time % 24 for time in times]

        self.times = times

    def get_nodes(self, wn):
        data = pd.read_excel('Input Files/micropolis/Micropolis_pop_at_node.xlsx')
        node_capacity = dict(zip(
                                 data['Node'].tolist(),
                                 data['Max Population'].tolist()
                             ))
        self.res_nodes = [name for name, node in wn.junctions()
                          if node.demand_timeseries_list[0].pattern_name == '2' and
                          node.demand_timeseries_list[0].base_value > 0 and
                          node_capacity[name] != 0]
        self.ind_nodes = [name for name, node in wn.junctions()
                          if node.demand_timeseries_list[0].pattern_name == '3' and
                          node.demand_timeseries_list[0].base_value > 0]
        self.com_nodes = [name for name, node in wn.junctions()
                          if (node.demand_timeseries_list[0].pattern_name == '4' or
                          node.demand_timeseries_list[0].pattern_name == '5' or
                          node.demand_timeseries_list[0].pattern_name == '6') and
                          node.demand_timeseries_list[0].base_value > 0 and
                          node_capacity[name] != 0]
        self.rest_nodes = [name for name, node in wn.junctions()
                           if node.demand_timeseries_list[0].pattern_name == '1' and
                           node.demand_timeseries_list[0].base_value > 0 and
                           node_capacity[name] != 0]
        self.all_nodes = [name for name, node in wn.junctions()
                          if node.demand_timeseries_list[0].base_value > 0 and
                          node_capacity[name] != 0]

        self.ind_nodes_obj = [name for name, node in wn.junctions()
                              if node.demand_timeseries_list[0].pattern_name == '3']

    def calc_flow_diff(self, data, hours):
        flow_data = dict()
        flow_changes_sum = dict()
        for (pipe, colData) in data.items():
            if 'MA' in pipe:
                curr_flow_changes = list()
                for i in range(len(colData) - 1):
                    if colData[(i+1)*3600] * colData[i*3600] < 0:
                        curr_flow_changes.append(1)
                    else:
                        curr_flow_changes.append(0)
                flow_changes_sum[pipe] = sum(curr_flow_changes[0:hours])/24
                flow_data[pipe] = curr_flow_changes

        # output = pd.DataFrame(flow_data)
        # change_sum = pd.Series(flow_changes_sum)

        return flow_data, flow_changes_sum
        # return output, change_sum

    def calc_distance(self, node1, node2):
        p1x, p1y = node1.coordinates
        p2x, p2y = node2.coordinates

        return math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)

    def calc_age_diff(self, data):
        out_dict = dict()
        for (node, colData) in data.items():
            out_data = colData.iloc[-1] / 3600
            if 'TN' in node and out_data < 500:
                out_dict[node] = out_data

        return out_dict

    def calc_industry_distance(self, wn):
        '''
        Function to calculate the distance from each residential node
        to the nearest industrial node.
        '''
        ind_distances = dict()
        close_node = dict()
        for res_name in self.res_nodes:
            node = wn.get_node(res_name)
            curr_node_dis = dict()
            for ind_name in self.ind_nodes:
                ind_node = wn.get_node(ind_name)
                curr_node_dis[ind_node.name] = self.calc_distance(node, ind_node)
            # find the key with the min value, i.e. the node with the lowest distance
            ind_distances[node.name] = min(curr_node_dis.values())
            close_node[node.name] = min(curr_node_dis, key=curr_node_dis.get)

        return ind_distances, close_node

    def calc_closest_node(self, wn):
        closest_distances = dict()
        for node1 in self.res_nodes:
            curr_node_close = dict()
            for node2 in self.all_nodes:
                if node1.name != node2.name:
                    curr_node_close[node2.name] = self.calc_distance(node1, node2)
            closest_distances[node1.name] = min(curr_node_close.values())

        return closest_distances

    def make_avg_plot(self, ax, data, sd, cols, x_values,
                      xlabel=None, ylabel=None, fig_name=None,
                      show_labels=False, logx=False, sd_plot=True):
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

        ''' Plot a single figure with the input data '''
        # set multiplier m to ensure plots with 3 or less lines
        # get the correct colors.
        if data.shape[1] > 3:
            m = 1
        else:
            m = 2

        # plot each column of data
        for i, col in enumerate(cols):
            ax.plot(x_values, data[col], color='C' + str(i * m))

        if sd_plot:
            #  need to separate so that the legend fills correctly
            for i, col in enumerate(cols):
                ax.fill_between(x_values, data[col] - sd[col],
                                data[col] + sd[col],
                                color='C' + str(i * m), alpha=0.5)

        if show_labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(cols)

        if logx:
            ax.xscale('log')

        return ax

    def make_flow_plot(self, change_data, sum_data, percent, dir, legend_text,
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
            plt.savefig(self.pub_loc + title + '.' + self.format,
                        format=self.format, bbox_inches='tight')
            plt.close()
        else:
            ax.plot(x_values, y_1.mean(axis=1), color=self.prim_colors[0])
            ax.plot(x_values, y_2.mean(axis=1), color=self.prim_colors[2])

    def export_agent_loc(self, wn, output_loc, locations):
        res_loc = locations[self.res_nodes].sum(axis=1)
        ind_loc = locations[self.ind_nodes].sum(axis=1)
        com_loc = locations[self.com_nodes].sum(axis=1)
        rest_loc = locations[self.rest_nodes].sum(axis=1)
        output = pd.DataFrame({'res': res_loc,
                               'ind': ind_loc,
                               'com': com_loc,
                               'rest': rest_loc})
        output.to_csv(output_loc + 'locations.csv')

    def make_seir_plot(self):
        ''' Function to make the seir plot with the input columns '''
        base_data = dcp(self.base['avg_seir_data'])
        pm_data = dcp(self.pm['avg_seir_data'])
        base_sd = dcp(ut.calc_error(self.base['var_seir_data'], self.error))
        pm_sd = dcp(ut.calc_error(self.pm['var_seir_data'], self.error))
        base_data = base_data * 100
        pm_data = pm_data * 100
        base_sd = base_sd * 100
        pm_sd = pm_sd * 100

        input = ['S', 'E', 'I', 'R', 'wfh']
        leg_text = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'WFH']

        x_values = np.array([x for x in np.arange(0, 90, 90 / len(base_data))])

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        axes[0] = self.make_avg_plot(axes[0], base_data, base_sd, input, x_values)
        axes[1] = self.make_avg_plot(axes[1], pm_data, pm_sd, input, x_values)

        plt.ylim(0, 100)
        plt.gcf().set_size_inches(7, 3.5)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Percent Population', x=0.04)
        axes[0].legend(leg_text, loc='upper left')
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)

        plt.savefig(self.pub_loc + '/' + 'seir_' + self.error + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def make_distance_plot(self, x, y1, y2, sd1, sd2,
                           xlabel, ylabel, name, data_names):
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
        sd_y1 = ut.calc_error(sd_y1.statistic, self.error) / 3600
        sd_y2 = ut.calc_error(sd_y2.statistic, self.error) / 3600
        # print(mean_y1.bin_edges)

        bin_names = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500',
                     '2500-3000', '3000-3500']

        data_dict = {data_names[0]: mean_y1.statistic,
                     data_names[1]: mean_y2.statistic}

        sd_dict = {data_names[0]: sd_y1,
                   data_names[1]: sd_y2}

        bar_x = np.arange(len(bin_names))
        width = 0.25  # width of the bars
        multiplier = 0  # iterator

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in data_dict.items():
            offset = width * multiplier
            ax.bar(bar_x + offset, measurement, width, label=attribute,
                   color='C' + str(multiplier * 2), yerr=sd_dict[attribute],
                   error_kw=dict(lw=0.5, capsize=2, capthick=0.5))
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.legend()
        ax.set_xticks(bar_x + width, bin_names, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.savefig(self.pub_loc + name + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def make_heatmap(self, data, xlabel, ylabel, name, aspect):
        ''' heatmap plot of all agents '''
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect=aspect)  # for 100 agents: 0.03, for 1000 agents: 0.003
        ax.figure.colorbar(im, ax=ax)
        # plt.xlim(1, data.shape[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # x_tick_labels = [0, 20, 40, 60, 80]
        # ax.set_xticks([i * 24 for i in x_tick_labels])
        # ax.set_xticklabels(x_tick_labels)

        plt.savefig(name + '.' + self.format,
                    format=self.format,
                    bbox_inches='tight')
        plt.close()

    def calc_model_stats(self, wn, seir, age):
        '''
        Function for calculating comparison stats for decision models:
            - average water age at the end of the simluation
            - peak infection rate
            - final susceptible count
            - peak exposure rate
        '''
        age_data = getattr(age[self.nodes], 'mean')(axis=1)
        final_age = age_data[(len(age_data) - 1) * 3600]
        max_inf = seir.I.loc[int(round(seir.I.idxmax(), 0))]
        max_exp = seir.E.loc[int(round(seir.E.idxmax(), 0))]
        final_sus = seir.S[(len(seir.S) - 1)]

        return final_age, max_inf, final_sus, max_exp

    def check_stats(self, new_list, old_stats):
        if new_list.max() > old_stats[0]:
            old_stats[0] = new_list.max()
        if new_list.min() < old_stats[1]:
            old_stats[1] = new_list.min()

        return old_stats


class Graphics(BaseGraphics):
    '''
    Main class for making graphics. The boilerplate methods are in
    BaseGraphics.

    parameters:
        publication (bool): whether to plot pngs or pdfs
        error        (str): the type of error to use
                            options include: se, ci95, and sd
    '''

    def __init__(self, publication, error):
        self.base_comp_dir = 'Output Files/30_no_pm/'
        self.pm_comp_dir = 'Output Files/30_all_pm/'
        self.wfh_loc = 'Output Files/30_wfh/'
        self.dine_loc = 'Output Files/30_dine/'
        self.groc_loc = 'Output Files/30_grocery/'
        self.ppe_loc = 'Output Files/30_ppe/'
        self.comp_list = ['seir_data', 'demand', 'age', 'flow',
                          'cov_ff', 'cov_pers',
                          'wfh', 'dine', 'groc', 'ppe']
        self.pm = ut.read_comp_data(self.pm_comp_dir, self.comp_list)
        self.base = ut.read_comp_data(self.base_comp_dir, self.comp_list)
        self.wfh = ut.read_comp_data(self.wfh_loc, ['seir_data', 'age'])
        self.dine = ut.read_comp_data(self.dine_loc, ['seir_data', 'age'])
        self.grocery = ut.read_comp_data(self.groc_loc, ['seir_data', 'age'])
        self.ppe = ut.read_comp_data(self.ppe_loc, ['seir_data', 'age'])

        # day200_loc = 'Output Files/2022-12-12_14-33_ppe_200Days_results/'
        # day400_loc = 'Output Files/2022-12-14_10-08_no_PM_400Days_results/'
        # days_200 = read_data(day200_loc, ['seir', 'demand', 'age'])
        # days_400 = read_data(day400_loc, ['seir', 'demand', 'age'])

        ''' Set figure parameters '''
        plt.rcParams['figure.figsize'] = [3.5, 3.5]
        self.error = error
        if publication:
            self.pub_loc = 'Output Files/publication_figures/'
            plt.rcParams['figure.dpi'] = 800
            self.format = 'pdf'
        else:
            self.pub_loc = 'Output Files/png_figures/'
            self.format = 'png'
            plt.rcParams['figure.dpi'] = 500

        self.prim_colors = ['#253494', '#2c7fb8', '#41b6c4', '#a1dab4', '#f1f174']
        # sec_colors = ['#454545', '#929292', '#D8D8D8']

        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.prim_colors)
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

        ''' Import water network and data '''
        inp_file = 'Input Files/micropolis/MICROPOLIS_v1_inc_rest_consumers.inp'
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.x_values = np.array([
            x for x in np.arange(0, 90, 90 / len(self.pm['avg_demand']))
        ])

        ''' Get times list: first time is max wfh, 75% wfh, 50% wfh, 25% wfh '''
        self.get_times(self.pm)

        ''' Set the various node lists '''
        self.get_nodes(self.wn)

    def flow_plots(self):
        ''' Make the flow direction changes plot '''
        pm_flow_change, pm_flow_sum = self.calc_flow_diff(
            self.pm['avg_flow'],
            self.times[len(self.times) - 1]
        )
        base_flow_change, base_flow_sum = self.calc_flow_diff(
            self.base['avg_flow'],
            self.times[len(self.times) - 1]
        )

        # print(pm_flow_sum['MA728'])
        # print(base_flow_sum['MA728'])

        ax = wntr.graphics.plot_network(self.wn, link_attribute=pm_flow_sum,
                                        link_colorbar_label='Flow Changes',
                                        node_size=0, link_width=2)
        plt.savefig(self.pub_loc + 'flow_network_pm.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ax = wntr.graphics.plot_network(self.wn, link_attribute=base_flow_sum,
                                        link_colorbar_label='Flow Changes',
                                        node_size=0, link_width=2)
        plt.savefig(self.pub_loc + 'flow_network_base.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def demand_plots(self):
        ''' Make demand plots by sector with PM data '''
        # define the columns of the input data and the x_values
        cols = ['Residential', 'Commercial', 'Industrial']

        # collect demands
        res_dem = self.pm['avg_demand'][self.res_nodes]
        com_dem = self.pm['avg_demand'][self.com_nodes]
        ind_dem = self.pm['avg_demand'][self.ind_nodes]

        res_var = self.pm['var_demand'][self.res_nodes]
        com_var = self.pm['var_demand'][self.com_nodes]
        ind_var = self.pm['var_demand'][self.ind_nodes]

        # make input data and sd
        sector_dem = pd.concat([res_dem.sum(axis=1).rolling(24).mean(),
                                com_dem.sum(axis=1).rolling(24).mean(),
                                ind_dem.sum(axis=1).rolling(24).mean()],
                               axis=1, keys=cols)
        sector_dem_var = pd.concat([res_var.sum(axis=1).rolling(24).mean(),
                                    com_var.sum(axis=1).rolling(24).mean(),
                                    ind_var.sum(axis=1).rolling(24).mean()],
                                   axis=1, keys=cols)

        sector_dem_err = ut.calc_error(sector_dem_var, self.error)

        # plot demand by sector
        ax = plt.subplot()
        self.make_avg_plot(ax, sector_dem, sector_dem_err, cols, self.x_values,
                           'Time (days)', 'Demand (L)',
                           show_labels=True) 
        plt.savefig(self.pub_loc + 'sector_demand' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make plots of aggregate demand data '''
        demand_base = self.base['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_pm = self.pm['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand = pd.concat(
            [demand_base.sum(axis=1).rolling(24).mean(),
             demand_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'PM']
        )

        var_base = self.base['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        var_pm = self.pm['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_var = pd.concat(
            [var_base.sum(axis=1).rolling(24).mean(),
             var_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'PM']
        )

        demand_err = ut.calc_error(demand_var, self.error)

        ax = plt.subplot()
        self.make_avg_plot(ax, demand, demand_err, ['Base', 'PM'],
                           self.x_values,
                           'Time (days)', 'Demand (L)', show_labels=True)
        plt.savefig(self.pub_loc + 'sum_demand_aggregate' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def age_plots(self):
        ''' Make age plot by sector for both base and PM '''
        cols = ['Residential', 'Commercial', 'Industrial']
        res_age_pm = self.pm['avg_age'][self.res_nodes].mean(axis=1)
        com_age_pm = self.pm['avg_age'][self.com_nodes].mean(axis=1)
        ind_age_pm = self.pm['avg_age'][self.ind_nodes].mean(axis=1)

        res_sd_pm = self.pm['var_age'][self.res_nodes].mean(axis=1)
        com_sd_pm = self.pm['var_age'][self.com_nodes].mean(axis=1)
        ind_sd_pm = self.pm['var_age'][self.ind_nodes].mean(axis=1)

        res_age_base = self.base['avg_age'][self.res_nodes].mean(axis=1)
        com_age_base = self.base['avg_age'][self.com_nodes].mean(axis=1)
        ind_age_base = self.base['avg_age'][self.ind_nodes].mean(axis=1)

        res_sd_base = self.base['var_age'][self.res_nodes].mean(axis=1)
        com_sd_base = self.base['var_age'][self.com_nodes].mean(axis=1)
        ind_sd_base = self.base['var_age'][self.ind_nodes].mean(axis=1)

        # make input data and sd
        pm_age = pd.concat([res_age_pm.rolling(24).mean(),
                            com_age_pm.rolling(24).mean(),
                            ind_age_pm.rolling(24).mean()],
                           axis=1, keys=cols)
        pm_age_var = pd.concat([res_sd_pm.rolling(24).mean(),
                                com_sd_pm.rolling(24).mean(),
                                ind_sd_pm.rolling(24).mean()],
                               axis=1, keys=cols)
        pm_age_err = ut.calc_error(pm_age_var, self.error)
        base_age = pd.concat([res_age_base.rolling(24).mean(),
                              com_age_base.rolling(24).mean(),
                              ind_age_base.rolling(24).mean()],
                             axis=1, keys=cols)
        base_age_var = pd.concat([res_sd_base.rolling(24).mean(),
                                  com_sd_base.rolling(24).mean(),
                                  ind_sd_base.rolling(24).mean()],
                                 axis=1, keys=cols)
        base_age_err = ut.calc_error(base_age_var, self.error)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        axes[0] = self.make_avg_plot(axes[0], base_age / 3600, base_age_err / 3600,
                                     cols, self.x_values)
        axes[1] = self.make_avg_plot(axes[1], pm_age / 3600, pm_age_err / 3600,
                                     cols, self.x_values)

        axes[0].legend(cols)
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Age (hrs)', x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(self.pub_loc + 'mean_age_sector.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make age plot comparing base and PM '''
        # make_sector_plot(self.wn, no_wfh['avg_age'] / 3600, 'Age (hr)', 'mean',
        #                  'mean_age_aggregate_' + error, wfh['avg_age'] / 3600,
        #                  sd=ut.calc_error(no_wfh['var_age'], error)/3600,
        #                  sd2=ut.calc_error(wfh['var_age'], error)/3600, type='all')

        pm_age = self.calc_age_diff(self.pm['avg_age'])
        base_age = self.calc_age_diff(self.base['avg_age'])

        ax = wntr.graphics.plot_network(self.wn, node_attribute=pm_age,
                                        node_colorbar_label='Age (hrs)',
                                        node_size=4, link_width=0.3)
        plt.savefig(self.pub_loc + 'age_network_pm.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ax = wntr.graphics.plot_network(self.wn, node_attribute=base_age,
                                        node_colorbar_label='Age (hrs)',
                                        node_size=4, link_width=0.3)
        plt.savefig(self.pub_loc + 'age_network_base.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def ind_dist_plots(self):
        ''' Calculate the distance to the closest industrial node '''
        ind_distances, ind_closest = self.calc_industry_distance(self.wn)

        ''' Make lists of the age values and age error values to plot '''
        pm_age_values = list()
        base_age_values = list()
        pm_age_sd = list()
        base_age_sd = list()
        pm_curr_age_values = self.pm['avg_age'].iloc[len(self.pm['avg_age'])-1]/3600
        base_curr_age_values = self.base['avg_age'].iloc[len(self.base['avg_age'])-1]/3600
        pm_curr_age_sd = self.pm['var_age'].iloc[len(self.pm['var_age'])-1]
        base_curr_age_sd = self.base['var_age'].iloc[len(self.base['var_age'])-1]
        # print(pm_curr_age_values)
        ''' Collect the age for each residential node '''
        for i, age in pm_curr_age_values.items():
            base_age = base_curr_age_values[i]
            pm_sd = pm_curr_age_sd[i]
            base_sd = base_curr_age_sd[i]
            # if the node is in the distance list, add the age
            # otherwise delete the node from the distance list
            if i in ind_distances.keys():
                if age < 500:
                    pm_age_values.append(age)
                    base_age_values.append(base_age)
                    pm_age_sd.append(pm_sd)
                    base_age_sd.append(base_sd)
                else:
                    del ind_distances[i]

        dist_values = [i for i in ind_distances.values()]
        self.make_distance_plot(dist_values, base_age_values, pm_age_values,
                                base_age_sd,
                                pm_age_sd,
                                'Distance (m)', 'Age (hr)', 'pm_age_ind_distance',
                                ['Base', 'PM'])

    def sv_heatmap_plots(self):
        ''' Make agent state variable plots '''
        base_sv = ut.read_data('Output Files/30_all_pm/2023-05-30_15-29_0_results/',
                               ['cov_pers', 'cov_ff', 'media'])
        no_pm_sv = ut.read_data('Output Files/30_no_pm/2023-05-26_08-33_0_results/',
                                ['cov_pers', 'cov_ff', 'media'])

        agent = '124'
        cols = ['Personal', 'Friends-Family', 'Media']
        data = pd.concat([base_sv['cov_pers'][agent],
                          base_sv['cov_ff'][agent],
                          base_sv['media'][agent]],
                         axis=1, keys=cols)
        plt.plot(np.delete(self.x_values, 0), data)
        plt.xlabel('Time (day)')
        plt.ylabel('Value')

        plt.savefig(self.pub_loc + 'state_variable_plot.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        self.make_heatmap(base_sv['cov_ff'].T,
                          'Time (day)', 'Agent', 'ff_heatmap_all_pm', 6)
        self.make_heatmap(no_pm_sv['cov_ff'].T,
                          'Time (day)', 'Agent', 'ff_heatmap_no_pm', 6)

    def sv_comp_plots(self):
        ''' State variable scenario comparisons '''
        ff = pd.concat([self.base['avg_cov_ff'].mean(axis=1),
                        self.pm['avg_cov_ff'].mean(axis=1)],
                       axis=1, keys=['Base', 'PM'])
        ff_var = pd.concat([self.base['var_cov_ff'].mean(axis=1),
                            self.pm['var_cov_ff'].mean(axis=1)],
                           axis=1, keys=['Base', 'PM'])
        ff_err = ut.calc_error(ff_var, self.error)

        # ax = plt.subplot()
        # ax = self.make_avg_plot(ax, data, err, ['Base', 'PM'],
        #                         np.delete(self.x_values, 0),
        #                         'Time (day)', 'Average Value',
        #                         show_labels=True)
        # plt.savefig(self.pub_loc + 'ff_avg' + '.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        pers = pd.concat([self.base['avg_cov_pers'].mean(axis=1),
                          self.pm['avg_cov_pers'].mean(axis=1)],
                         axis=1, keys=['Base', 'PM'])
        pers_var = pd.concat([self.base['var_cov_pers'].mean(axis=1),
                              self.pm['var_cov_pers'].mean(axis=1)],
                             axis=1, keys=['Base', 'PM'])
        pers_err = ut.calc_error(pers_var, self.error)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False)
        axes[0] = self.make_avg_plot(axes[0], pers, pers_err, ['Base', 'PM'],
                                     np.delete(self.x_values, 0))
        axes[1] = self.make_avg_plot(axes[1], ff, ff_err, ['Base', 'PM'],
                                     np.delete(self.x_values, 0))

        axes[0].legend(['Base', 'PM'])
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Average Values', x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(self.pub_loc + 'sv_comparison.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # ax = plt.subplot()
        # ax = self.make_avg_plot(ax, data, err, ['Base', 'PM'],
        #                         np.delete(self.x_values, 0),
        #                         'Time (day)', 'Average Value',
        #                         show_labels=True)
        # plt.savefig(self.pub_loc + 'pers_avg' + '.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

    def bbn_plots(self):
        ''' BBN decisions scenario comparisons '''
        cols = ['Dine out less', 'Grocery shop less', 'WFH', 'Wear PPE']
        data = pd.concat([self.pm['avg_dine'].mean(axis=1),
                          self.pm['avg_groc'].mean(axis=1),
                          self.pm['avg_wfh'].mean(axis=1),
                          self.pm['avg_ppe'].mean(axis=1)],
                         axis=1, keys=cols)
        var = pd.concat([self.pm['var_dine'].mean(axis=1),
                         self.pm['var_groc'].mean(axis=1),
                         self.pm['var_wfh'].mean(axis=1),
                         self.pm['var_ppe'].mean(axis=1)],
                        axis=1, keys=cols)
        err = ut.calc_error(var, self.error)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                     sharex=True, sharey=True)
        ax1.plot(np.delete(self.x_values, 0), data[cols[0]])
        ax1.fill_between(np.delete(self.x_values, 0), data[cols[0]] - err[cols[0]],
                         data[cols[0]] + err[cols[0]], alpha=0.5)
        ax2.plot(np.delete(self.x_values, 0), data[cols[1]])
        ax2.fill_between(np.delete(self.x_values, 0), data[cols[1]] - err[cols[1]],
                         data[cols[1]] + err[cols[1]], alpha=0.5)
        ax3.plot(np.delete(self.x_values, 0), data[cols[2]])
        ax3.fill_between(np.delete(self.x_values, 0), data[cols[2]] - err[cols[2]],
                         data[cols[2]] + err[cols[2]], alpha=0.5)
        ax4.plot(np.delete(self.x_values, 0), data[cols[3]])
        ax4.fill_between(np.delete(self.x_values, 0), data[cols[3]] - err[cols[3]],
                         data[cols[3]] + err[cols[3]], alpha=0.5)
        ax1.text(0.5, -0.14, "(a)", size=12, ha="center",
                 transform=ax1.transAxes)
        ax2.text(0.5, -0.14, "(b)", size=12, ha="center",
                 transform=ax2.transAxes)
        ax3.text(0.5, -0.24, "(c)", size=12, ha="center",
                 transform=ax3.transAxes)
        ax4.text(0.5, -0.24, "(d)", size=12, ha="center",
                 transform=ax4.transAxes)
        fig.supxlabel('Time (days)', y=-0.02)
        fig.supylabel('Percent Adoption', x=-0.03)

        plt.savefig(self.pub_loc + 'bbn_decision_all_pm.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def make_single_plots(self, file, days):
        ''' Set the warmup period '''
        x_len = days * 24

        ''' Make SEIR plot without error '''
        loc = 'Output Files/' + file + '/'
        comp_list = self.comp_list + ['bw_cost',
                                      'tw_cost',
                                      'bw_demand',
                                      'income',
                                      'hygiene',
                                      'drink',
                                      'cook']
        data = ut.read_data(loc, comp_list)
        households = len(data['income'].columns)
        print(households)
        data['tot_cost'] = data['bw_cost'] + data['tw_cost']
        leg_text = ['S', 'E', 'I', 'R', 'wfh']
        ax = plt.subplot()
        x_values = np.array([
            x for x in np.arange(0, days, days / x_len)
        ])
        self.make_avg_plot(
            ax, data['seir_data'].iloc[-x_len:]*100, None, leg_text, x_values,
            'Time (days)', 'Percent Population', show_labels=True, sd_plot=False)
        plt.savefig(loc + 'seir' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make demand plot '''
        # base_loc = 'Output Files/base_results/'
        # base_data = ut.read_data(base_loc, self.comp_list)
        # ax = plt.subplot()
        # pm_demand = data['demand'].loc[:, self.all_nodes]
        # base_demand = base_data['demand'].loc[:, self.all_nodes]
        # demand = pd.concat([pm_demand.sum(axis=1).rolling(24).mean(),
        #                     base_demand.sum(axis=1).rolling(24).mean()],
        #                    axis=1, keys=['PM', 'Base'])
        # x_values = np.array([
        #     x for x in np.arange(0, 90, 90 / len(data['demand'].index))
        # ])
        # print(demand[:-1])

        # self.make_avg_plot(
        #     ax, demand[:-1], None, ['PM', 'Base'],
        #     x_values, xlabel='Time (days)', ylabel='Demand',
        #     show_labels=True, sd_plot=False
        # )
        # plt.savefig(self.pub_loc + file + 'demand' + '.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        demand = data['demand'].loc[:, self.all_nodes].sum(axis=1)
        x_values = np.array([
            x for x in np.arange(0, days, days / x_len)
        ])
        plt.plot(x_values, demand.iloc[-x_len:])
        plt.savefig(loc + '_demand_' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make age plots '''
        age = data['age'][self.all_nodes].mean(axis=1)
        # print(data['age'].loc[8470800, self.com_nodes].sort_values() / 3600)
        # print(data['age'].loc[8470800, self.res_nodes].sort_values() / 3600)
        plt.plot(x_values, age.iloc[-x_len:] / 3600)
        plt.savefig(loc + 'age' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        cols = ['Residential', 'Commercial', 'Industrial']
        res_age_pm = data['age'][self.res_nodes].mean(axis=1)
        com_age_pm = data['age'][self.com_nodes].mean(axis=1)
        ind_age_pm = data['age'][self.ind_nodes].mean(axis=1)

        # make input data and sd
        pm_age = pd.concat([res_age_pm.rolling(24).mean(),
                            com_age_pm.rolling(24).mean(),
                            ind_age_pm.rolling(24).mean()],
                           axis=1, keys=cols)
        pm_age_sd = pd.concat([res_age_pm.rolling(24).std(),
                               com_age_pm.rolling(24).std(),
                               ind_age_pm.rolling(24).std()],
                              axis=1, keys=cols)
        ax = plt.subplot()
        self.make_avg_plot(
            ax, pm_age.iloc[-x_len:] / 3600, pm_age_sd[-x_len:] / 3600, cols,
            x_values, 'Time (days)', 'Water Age (hr)', show_labels=True,
            sd_plot=True
        )

        plt.savefig(loc + '_sector_age.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # plt.show()

        ''' Heatmap and map of costs '''
        # convert the annual income to an income that is specific to timeframe
        data['income'] = data['income'] * days / 365
        self.make_heatmap(
            data['tot_cost'].T,
            'Time (weeks)',
            'Household',
            loc + 'tot_cost_heatmap',
            0.01
        )

        cols = ['Tap Water', 'Bottled Water', 'Total']
        for i in data['tw_cost'].max():
            print(i)
        cost = pd.concat([data['tw_cost'].mean(axis=1),
                          data['bw_cost'].mean(axis=1),
                          data['tot_cost'].mean(axis=1)],
                         axis=1, keys=cols)
        print(cost)
        ax = plt.subplot()
        self.make_avg_plot(
            ax, cost, None, cols, cost.index / 24,
            'Time (Weeks)', 'Water Cost ($)', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'water_cost.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        print(data['income'].iloc[0, :].groupby(level=0).mean())
        print(data['tot_cost'].iloc[-1, :].groupby(level=0).mean())

        ax = wntr.graphics.plot_network(
            self.wn,
            node_attribute=data['income'].iloc[0, :].groupby(level=0).mean(),
            node_size=5
        )
        plt.savefig(loc + 'income_map.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ax = wntr.graphics.plot_network(
            self.wn,
            node_attribute=data['tot_cost'].iloc[-1, :].groupby(level=0).mean(),
            node_size=5
        )
        plt.savefig(loc + 'tot_cost_map.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Plot of TWA parameters '''
        twa = pd.concat([data['hygiene'].sum(axis=1),
                         data['drink'].sum(axis=1),
                         data['cook'].sum(axis=1)],
                        axis=1, keys=['Hygiene', 'Drink', 'Cook'])

        ax = plt.subplot()
        self.make_avg_plot(
            ax, twa / households * 100, None, ['Hygiene', 'Drink', 'Cook'], twa.index / 24,
            'Time (days)', 'Percent of Households', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'twa.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()
