import math
import wntr
import os
import numpy as np
import pandas as pd
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import binned_statistic
import utils as ut
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


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

    def calc_sec_averages(self, data, op='mean', rolling=True):
        '''
        Calculate the average of the input data by sector
        '''
        output = dict()
        if op == 'mean':
            output['res'] = data[self.res_nodes].mean(axis=1)
            output['com'] = data[self.com_nodes].mean(axis=1)
            output['ind'] = data[self.ind_nodes].mean(axis=1)
        elif op == 'sum':
            output['res'] = data[self.res_nodes].sum(axis=1)
            output['com'] = data[self.com_nodes].sum(axis=1)
            output['ind'] = data[self.ind_nodes].sum(axis=1)
        else:
            raise NotImplementedError(f"Operation {op} not implemented.")

        if rolling:
            cols = ['Residential', 'Commercial', 'Industrial']
            output = pd.concat([output['res'].rolling(24).mean(),
                                output['com'].rolling(24).mean(),
                                output['ind'].rolling(24).mean()],
                               axis=1, keys=cols)

        return output

    def calc_twa_averages(self, data, keys):
        '''
        Calculate the average and package TWA data for plotting
        '''
        output = pd.concat(
            [data['drink'].sum(axis=0),
             data['cook'].sum(axis=0)],
            axis=1, keys=keys
        )

        return output

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

    def collect_data(self, folder, param):
        '''
        Helper method for get_household. Collects data from n runs in folder
        '''
        output = pd.DataFrame()
        files = os.listdir(folder)
        files.sort()
        for file in files:
            if param in file:
                curr_data = pd.read_pickle(os.path.join(folder, file))
                if param == 'income':
                    output = pd.concat([output, curr_data])
                else:
                    output = pd.concat([output, curr_data.T])

        return output

    def get_household(self, folder, param):
        '''
        Combine per household data from a group of n runs

        Parameters
        ----------
        folder : str
            data folder

        param : (str | list)
            param(s) to combine
        '''
        output = dict()
        if isinstance(param, str):
            output[param] = self.collect_data(folder, param)
        elif isinstance(param, list):
            for item in param:
                output[item] = self.collect_data(folder, item)

        return output

    def package_household(self, data, dir, base=False):
        '''
        Package household data necessary for plotting later
        '''
        # data interpolated from HUD extremely low income values using average
        # clinton household size of 2.56
        extreme_income = 23452.8

        data['cost'] = self.get_household(
            dir + '/hh_results/', ['bw_cost', 'tw_cost'] if not base else ['tw_cost']
        )
        data['twa'] = self.get_household(
            dir + '/hh_results/', ['drink', 'cook']
        )
        data['income'] = self.get_household(
            dir + '/hh_results/', 'income'
        )['income']

        if base:
            # no bottled water cost in the base case
            data['cost']['total'] = data['cost']['tw_cost']
        else:
            data['cost']['total'] = data['cost']['bw_cost'] + data['cost']['tw_cost']

        data['cowpi'] = pd.DataFrame(
            {'level': data['income'].loc[:, 'level'],
             'cowpi': data['cost']['total'].iloc[:, -1] / (data['income'].loc[:, 'income'] * self.days / 365),
             'income': data['income'].loc[:, 'income']},
            index=data['cost']['total'].index
        )

        data['cowpi'].loc[data['cowpi']['income'] < extreme_income, 'level'] = 0

    def post_household(self):
        '''
        Manipulate household data to be ready to plot
        '''
        # get base data ready
        self.package_household(self.base, self.base_comp_dir, base=True)

        # get base+bw data ready
        self.package_household(self.basebw, self.base_bw_comp_dir)

        # get pm data ready
        self.package_household(self.pm, self.pm_comp_dir)

        # get pm 25ind data ready
        self.package_household(self.pm25ind, self.pm_25ind_comp_dir)

        # get pm 50ind data ready
        self.package_household(self.pm50ind, self.pm_50ind_comp_dir)

        # get pm 75ind data ready
        self.package_household(self.pm75ind, self.pm_75ind_comp_dir)

        # get pm 100ind data ready
        self.package_household(self.pm100ind, self.pm_100ind_comp_dir)

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
        if isinstance(data, pd.Series):
            m = 1
        else:
            if len(data.columns) > 3:
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

    def make_seir_plot(self, days):
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

        x_values = np.array([x for x in np.arange(0, days, days / len(base_data))])

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

    def make_cowpi_plot(self, data, err, name):
        ''' Make barchart of cowpi '''
        data.plot(
            kind='bar', log=True,
            yerr=err, capsize=3,
            ylabel='% of Income', rot=0
        )
        plt.gcf().set_size_inches(3.5, 3.5)
        plt.savefig(self.pub_loc + name + '_cow_comparison.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # plot without the extremely low income households
        data = data.iloc[1:4, :]
        # print(cost_comp)
        data.plot(
            kind='bar', ylabel='% of Income',
            yerr=err, capsize=3, rot=0
        )
        plt.gcf().set_size_inches(3.5, 3.5)
        plt.savefig(self.pub_loc + name + '_cow_comparison_no_low_in.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def linear_regression(self, data, xname=None, yname=None, norm_x=True):
        '''
        Wrapper function to perform a linear regression.
        '''
        x = data[[xname]]
        y = data[yname]

        if norm_x:
            x = (
                (x - np.min(x)) / (np.max(x) - np.min(x))
            )

        # x_with_intercept = np.empty(shape=(len(x_norm), 2), dtype=np.float64)
        # x_with_intercept[:, 0] = 1
        # x_with_intercept[:, 1] = x_norm

        # model = sm.OLS(y_norm, x_with_intercept).fit()
        # # sse = np.sum((model.fittedvalues - y_norm)**2)
        # # ssr = np.sum((model.fittedvalues - y_norm.mean())**2)
        # # sst = ssr + sse
        # print(model.summary())

        model = LinearRegression()
        # x = x_norm[:, np.newaxis]
        # y = y_norm[:, np.newaxis]
        model.fit(x, y)
        print(f"R^2: {model.score(x, y)}")
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficient: {model.coef_}")

        return model, x

    def clinton_bg_plot(self, ax):
        res_nodes = ut.calc_clinton_ind_dists()
        results_ind = res_nodes.groupby(['group', 'bg']).mean().loc[:, 'min']

        # read in income distribution data
        income = pd.read_csv(
            'Input Files/clinton_income_data.csv',
            delimiter=',',
            names=['bg', 'income']
        )

        print(income)
        print(pd.Series(results_ind.values))

        data = pd.concat(
            [income['income'], pd.Series(results_ind.values)],
            axis=1,
            keys=['Income', 'Distance']
        )

        print(data)

        # x = results_ind.values
        # y = income[:, 1]

        model, mod_x = self.linear_regression(data, 'Distance', 'Income')

        # ax.scatter(x, y)
        ax.plot(mod_x, model.predict(mod_x))

        return ax

    def scatter_plot(self, ind_dist, income, ax):
        x = ind_dist
        x_norm = (
            (x - np.min(x)) / (np.max(x) - np.min(x))
        )
        x_with_intercept = np.empty(shape=(len(x_norm), 2), dtype=np.float64)
        x_with_intercept[:, 0] = 1
        x_with_intercept[:, 1] = x_norm

        y = income

        ols_model = sm.OLS(y, x_with_intercept).fit()
        sse = np.sum((ols_model.fittedvalues - y)**2)
        ssr = np.sum((ols_model.fittedvalues - y.mean())**2)
        sst = ssr + sse
        print(f'R2 = {ssr/sst}')
        print(len(x_norm))
        print(np.sqrt(sse/(len(x_norm) - 2)))
        print(ols_model.summary())

        lr_model = LinearRegression()
        x = x_norm[:, np.newaxis]
        lr_model.fit(x, y)
        print(lr_model.score(x, y))
        print(lr_model.coef_)
        print(lr_model.intercept_)

        ax.plot(x, lr_model.predict(x))
        ax.scatter(x, y)
        ax.set_xlabel('Normalized Industrial Distance')
        ax.set_ylabel('Household Income')

        return ax


class Graphics(BaseGraphics):
    '''
    Main class for making graphics. The boilerplate methods are in
    BaseGraphics.

    parameters:
        publication : bool
            whether to plot pngs or pdfs
        error : str
            the type of error to use
            options include: se, ci95, and sd
    '''

    def __init__(self, publication, error, days):
        self.days = days
        self.x_len = days * 24
        ''' Define data directories '''
        # self.base_comp_dir = 'Output Files/30_no_pm/'
        # self.pm_comp_dir = 'Output Files/30_all_pm/'
        self.base_comp_dir = 'Output Files/30_base_equity/'
        self.base_bw_comp_dir = 'Output Files/30_base-bw_equity/'
        self.pm_25ind_comp_dir = 'Output Files/30_all_pm_25ind_equity/'
        self.pm_50ind_comp_dir = 'Output Files/30_all_pm_50ind_equity/'
        self.pm_75ind_comp_dir = 'Output Files/30_all_pm_75ind_equity/'
        self.pm_100ind_comp_dir = 'Output Files/30_all_pm_100ind_equity/'
        self.pm_comp_dir = 'Output Files/30_all_pm-bw_equity/'
        self.pm_nobw_comp_dir = 'Output Files/30_all_pm_no-bw_equity/'
        self.wfh_loc = 'Output Files/30_wfh_equity/'
        self.dine_loc = 'Output Files/30_dine_equity/'
        self.groc_loc = 'Output Files/30_groc_equity/'
        self.ppe_loc = 'Output Files/30_ppe_equity/'
        self.comp_list = ['seir_data', 'demand', 'age', 'flow',
                          'cov_ff', 'cov_pers', 'agent_loc',
                          'wfh', 'dine', 'groc', 'ppe']
        ''' List that will be truncated based on the number of days the
        simulation was run. Does not include the warmup period '''
        self.truncate_list = [
            'seir_data', 'demand', 'age', 'flow'
        ]

        ''' Read in data from data directories '''
        self.pm = ut.read_comp_data(
            self.pm_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.pm_nobw = ut.read_comp_data(
            self.pm_nobw_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.base = ut.read_comp_data(
            self.base_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.basebw = ut.read_comp_data(
            self.base_bw_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.pm25ind = ut.read_comp_data(
            self.pm_25ind_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.pm50ind = ut.read_comp_data(
            self.pm_50ind_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.pm75ind = ut.read_comp_data(
            self.pm_75ind_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.pm100ind = ut.read_comp_data(
            self.pm_100ind_comp_dir, self.comp_list, days, self.truncate_list
        )
        self.wfh = ut.read_comp_data(
            self.wfh_loc, ['seir_data', 'age'], days, self.truncate_list
        )
        self.dine = ut.read_comp_data(
            self.dine_loc, ['seir_data', 'age'], days, self.truncate_list
        )
        self.grocery = ut.read_comp_data(
            self.groc_loc, ['seir_data', 'age'], days, self.truncate_list
        )
        self.ppe = ut.read_comp_data(
            self.ppe_loc, ['seir_data', 'age'], days, self.truncate_list
        )

        # print(self.pm['avg_wfh'])

        ''' Read and distill household level data '''
        self.post_household()

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
            self.pub_loc = 'Output Files/png_figures_equity/'
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
        self.x_values_hour = np.array([
            x for x in np.arange(0, days, days / len(self.pm['avg_demand']))
        ])
        self.x_values_day = np.array([
            x for x in range(180)
        ])

        ''' Get times list: first time is max wfh, 75% wfh, 50% wfh, 25% wfh '''
        self.get_times(self.pm)

        ''' Set the various node lists '''
        self.get_nodes(self.wn)

        ''' Get industrial distances for each residential node in the network '''
        self.ind_distances, ind_closest = self.calc_industry_distance(self.wn)
        # self.dist_values = [v for k, v in ind_distances.items() if k in self.res_nodes]

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

        # sort data
        sector_dem = self.calc_sec_averages(self.pm['avg_demand'], op='sum')
        sector_dem_var = self.calc_sec_averages(self.pm['var_demand'], op='sum')
        sector_dem_err = ut.calc_error(sector_dem_var, self.error)

        # sort data for 25ind case
        sector_dem50 = self.calc_sec_averages(self.pm25ind['avg_demand'], op='sum')
        sector_dem_var50 = self.calc_sec_averages(self.pm25ind['var_demand'], op='sum')
        sector_dem_err50 = ut.calc_error(sector_dem_var50, self.error)

        # sort data for 50ind case
        sector_dem50 = self.calc_sec_averages(self.pm50ind['avg_demand'], op='sum')
        sector_dem_var50 = self.calc_sec_averages(self.pm50ind['var_demand'], op='sum')
        sector_dem_err50 = ut.calc_error(sector_dem_var50, self.error)

        # sort data for 75ind case
        sector_dem75 = self.calc_sec_averages(self.pm75ind['avg_demand'], op='sum')
        sector_dem_var75 = self.calc_sec_averages(self.pm75ind['var_demand'], op='sum')
        sector_dem_err75 = ut.calc_error(sector_dem_var75, self.error)

        # sort data for 100ind case
        sector_dem100 = self.calc_sec_averages(self.pm100ind['avg_demand'], op='sum')
        sector_dem_var100 = self.calc_sec_averages(self.pm100ind['var_demand'], op='sum')
        sector_dem_err100 = ut.calc_error(sector_dem_var100, self.error)

        # plot demand by sector
        ax = plt.subplot()
        self.make_avg_plot(ax, sector_dem, sector_dem_err, cols, self.x_values_hour) 
        # ax1 = ax.twinx()
        # ax1.plot(self.x_values_day, self.pm['avg_wfh'].mean(axis=1) * 100, color='k')
        ax.legend(['Residential', 'Commercial', 'Industrial'])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Demand (L)')
        # ax1.set_ylabel('Percent of Population WFH')
        plt.savefig(self.pub_loc + 'sector_demand' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # plot demand by sector for 50ind
        ax = plt.subplot()
        self.make_avg_plot(ax, sector_dem50, sector_dem_err50, cols, self.x_values_hour) 
        # ax1 = ax.twinx()
        # ax1.plot(self.x_values_day, self.pm['avg_wfh'].mean(axis=1) * 100, color='k')
        ax.legend(['Residential', 'Commercial', 'Industrial'])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Demand (L)')
        # ax1.set_ylabel('Percent of Population WFH')
        plt.savefig(self.pub_loc + 'sector_demand_50ind' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make plots of aggregate demand data '''
        demand_base = self.base['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_basebw = self.basebw['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_pm = self.pm['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_pm_nobw = self.pm_nobw['avg_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand = pd.concat(
            [demand_base.sum(axis=1).rolling(24).mean(),
             demand_basebw.sum(axis=1).rolling(24).mean(),
             demand_pm_nobw.sum(axis=1).rolling(24).mean(),
             demand_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'Base+BW', 'PM', 'PM+BW']
        )

        var_base = self.base['var_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        var_basebw = self.basebw['var_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        var_pm = self.pm['var_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        var_pm_nobw = self.pm_nobw['var_demand'][
            self.res_nodes + self.com_nodes + self.ind_nodes
        ]
        demand_var = pd.concat(
            [var_base.sum(axis=1).rolling(24).mean(),
             var_basebw.sum(axis=1).rolling(24).mean(),
             var_pm_nobw.sum(axis=1).rolling(24).mean(),
             var_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'Base+BW', 'PM', 'PM+BW']
        )

        demand_err = ut.calc_error(demand_var, self.error)

        fig, axes = plt.subplots(1, 2)
        # format the y axis ticks to have a dollar sign and thousands commas
        fmt = '{x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        axes[0].yaxis.set_major_formatter(tick) 
        axes[1].yaxis.set_major_formatter(tick) 

        axes[0] = self.make_avg_plot(
            axes[0], demand, demand_err, ['Base', 'Base+BW', 'PM', 'PM+BW'],
            self.x_values_hour, show_labels=False
        )

        ''' Plot residential demand in a subfigure '''
        demand_base = self.base['avg_demand'][
            self.res_nodes
        ]
        demand_basebw = self.basebw['avg_demand'][
            self.res_nodes
        ]
        demand_pm = self.pm['avg_demand'][
            self.res_nodes
        ]
        demand_pm_nobw = self.pm_nobw['avg_demand'][
            self.res_nodes
        ]
        demand = pd.concat(
            [demand_base.sum(axis=1).rolling(24).mean(),
             demand_basebw.sum(axis=1).rolling(24).mean(),
             demand_pm_nobw.sum(axis=1).rolling(24).mean(),
             demand_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'Base+BW', 'PM', 'PM+BW']
        )

        var_base = self.base['var_demand'][
            self.res_nodes
        ]
        var_basebw = self.basebw['var_demand'][
            self.res_nodes
        ]
        var_pm = self.pm['var_demand'][
            self.res_nodes
        ]
        var_pm_nobw = self.pm_nobw['var_demand'][
            self.res_nodes
        ]
        demand_var = pd.concat(
            [var_base.sum(axis=1).rolling(24).mean(),
             var_basebw.sum(axis=1).rolling(24).mean(),
             var_pm_nobw.sum(axis=1).rolling(24).mean(),
             var_pm.sum(axis=1).rolling(24).mean()],
            axis=1,
            keys=['Base', 'Base+BW', 'PM', 'PM+BW']
        )

        demand_err = ut.calc_error(demand_var, self.error)

        axes[1] = self.make_avg_plot(
            axes[1], demand, demand_err,
            ['Base', 'Base+BW', 'PM', 'PM+BW'], self.x_values_hour
        )

        axes[0].legend(['Base', 'Base+BW', 'PM', 'PM+BW'])
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel('Time (days)', y=-0.06)
        fig.supylabel('Demand (L)', x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(self.pub_loc + 'sum_demand_aggregate' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def age_plots(self):
        ''' Make age plot by sector for both base and PM '''
        cols = ['Residential', 'Commercial', 'Industrial']

        age_pm = self.calc_sec_averages(self.pm['avg_age'])
        age_sd_pm = self.calc_sec_averages(self.pm['var_age'])
        age_pm_err = ut.calc_error(age_sd_pm, self.error)

        age_pm_nobw = self.calc_sec_averages(self.pm_nobw['avg_age'])
        age_sd_pm_nobw = self.calc_sec_averages(self.pm_nobw['var_age'])
        age_pm_nobw_err = ut.calc_error(age_sd_pm_nobw, self.error)

        age_base = self.calc_sec_averages(self.base['avg_age'])
        age_sd_base = self.calc_sec_averages(self.base['var_age'])
        age_base_err = ut.calc_error(age_sd_base, self.error)

        age_basebw = self.calc_sec_averages(self.basebw['avg_age'])
        age_sd_basebw = self.calc_sec_averages(self.basebw['var_age'])
        age_basebw_err = ut.calc_error(age_sd_basebw, self.error)

        age_pm50 = self.calc_sec_averages(self.pm50ind['avg_age'])
        age_sd_pm50 = self.calc_sec_averages(self.pm50ind['var_age'])
        age_pm50_err = ut.calc_error(age_sd_pm50, self.error)

        age_pm75 = self.calc_sec_averages(self.pm75ind['avg_age'])
        age_sd_pm75 = self.calc_sec_averages(self.pm75ind['var_age'])
        age_pm75_err = ut.calc_error(age_sd_pm75, self.error)

        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
        axes[0] = self.make_avg_plot(axes[0], age_base / 3600, age_base_err / 3600,
                                     cols, self.x_values_hour)
        axes[1] = self.make_avg_plot(axes[1], age_pm_nobw / 3600, age_pm_nobw_err / 3600,
                                     cols, self.x_values_hour)
        axes[2] = self.make_avg_plot(axes[2], age_pm / 3600, age_pm_err / 3600,
                                     cols, self.x_values_hour)

        axes[0].legend(cols)
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        axes[2].text(0.5, -0.14, "(c)", size=12, ha="center",
                     transform=axes[2].transAxes)
        fig.supxlabel('Time (days)', y=-0.06)
        fig.supylabel('Age (hrs)', x=0.04)
        plt.gcf().set_size_inches(8, 3.5)

        plt.savefig(self.pub_loc + 'mean_age_sector.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make plot of industrial demand SA '''
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
        axes[0] = self.make_avg_plot(axes[0], age_pm / 3600, age_pm_err / 3600,
                                     cols, self.x_values_hour)
        axes[1] = self.make_avg_plot(axes[1], age_pm50 / 3600, age_pm50_err / 3600,
                                     cols, self.x_values_hour)
        axes[2] = self.make_avg_plot(axes[2], age_pm75 / 3600, age_pm75_err / 3600,
                                     cols, self.x_values_hour)

        axes[0].legend(cols)
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        axes[2].text(0.5, -0.14, "(c)", size=12, ha="center",
                     transform=axes[2].transAxes)
        fig.supxlabel('Time (days)', y=-0.06)
        fig.supylabel('Age (hrs)', x=0.04)
        plt.gcf().set_size_inches(8, 3.5)

        plt.savefig(self.pub_loc + 'mean_age_sector_indSA.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make age plot comparing base and PM '''
        # make_sector_plot(self.wn, no_wfh['avg_age'] / 3600, 'Age (hr)', 'mean',
        #                  'mean_age_aggregate_' + error, wfh['avg_age'] / 3600,
        #                  sd=ut.calc_error(no_wfh['var_age'], error)/3600,
        #                  sd2=ut.calc_error(wfh['var_age'], error)/3600, type='all')

        pm_age = self.calc_age_diff(self.pm['avg_age'])
        base_age = self.calc_age_diff(self.base['avg_age'])
        basebw_age = self.calc_age_diff(self.basebw['avg_age'])

        fig, axes = plt.subplots(nrows=1, ncols=3)
        wntr.graphics.plot_network(
            self.wn, node_attribute=pm_age,
            node_colorbar_label='Age (hrs)',
            add_colorbar=False, node_range=[0, 450],
            node_size=4, link_width=0.3, ax=axes[2]
        )

        wntr.graphics.plot_network(
            self.wn, node_attribute=basebw_age,
            node_colorbar_label='Age (hrs)', node_range=[0, 450],
            node_size=4, link_width=0.3, ax=axes[1]
        )

        wntr.graphics.plot_network(
            self.wn, node_attribute=base_age,
            add_colorbar=False, node_range=[0, 450],
            node_size=4, link_width=0.3, ax=axes[0]
        )
        plt.gcf().set_size_inches(7, 3.5)
        plt.savefig(self.pub_loc + 'age_network_comp.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make plot comparing pm with pm+bw '''

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
        plt.plot(self.x_values_day, data)
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
                                     np.delete(self.x_values_day, 0))
        axes[1] = self.make_avg_plot(axes[1], ff, ff_err, ['Base', 'PM'],
                                     np.delete(self.x_values_day, 0))

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
        ax1.plot(np.delete(self.x_values_day, 0), data[cols[0]])
        ax1.fill_between(np.delete(self.x_values_day, 0), data[cols[0]] - err[cols[0]],
                         data[cols[0]] + err[cols[0]], alpha=0.5)
        ax2.plot(np.delete(self.x_values_day, 0), data[cols[1]])
        ax2.fill_between(np.delete(self.x_values_day, 0), data[cols[1]] - err[cols[1]],
                         data[cols[1]] + err[cols[1]], alpha=0.5)
        ax3.plot(np.delete(self.x_values_day, 0), data[cols[2]])
        ax3.fill_between(np.delete(self.x_values_day, 0), data[cols[2]] - err[cols[2]],
                         data[cols[2]] + err[cols[2]], alpha=0.5)
        ax4.plot(np.delete(self.x_values_day, 0), data[cols[3]])
        ax4.fill_between(np.delete(self.x_values_day, 0), data[cols[3]] - err[cols[3]],
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

    def make_equity_plots(self):
        metrics = pd.concat([self.base['avg_burden'].iloc[:, 0],
                             self.pm['avg_burden'].iloc[:, 0]],
                            axis=1, keys=['Base', 'PM'])
        metrics_var = pd.concat([self.base['var_burden'].iloc[:, 0],
                                 self.pm['var_burden'].iloc[:, 0]],
                                axis=1, keys=['Base', 'PM'])
        err = ut.calc_error(metrics_var, self.error)

        warmup = metrics.index[-1] - self.x_len

        ax = plt.subplot()
        self.make_avg_plot(
            ax, metrics * 100, err * 100, ['Base', 'PM'],
            (metrics.index - warmup) / 24,
            'Time (days)', '% of Income', show_labels=True
        )

        plt.savefig(self.pub_loc + 'equity_metrics.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def make_cost_plots(self):
        wntr.graphics.plot_network(
            self.wn,
            node_attribute=self.base['cost']['total'].iloc[:, -1].groupby(level=0).mean(),
            node_size=5,
            node_colorbar_label='Water Cost ($)'
        )
        plt.savefig(self.pub_loc + 'tot_cost_base_map.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # pm_tot_cost = self.pm['avg_bw_cost'] + self.pm['avg_tw_cost']
        # ax = wntr.graphics.plot_network(
        #     self.wn,
        #     node_attribute=pm_tot_cost.iloc[-1, :],
        #     node_size=5,
        #     node_range=[0, 15000],
        #     node_colorbar_label='Water Cost ($)'
        # )
        # plt.savefig(self.pub_loc + 'tot_cost_pm_map.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        # base_lower_income = self.base['avg_income'].quantile(0.2, axis=1)[0] * self.days / 365
        # pm_lower_income = self.base['avg_income'].quantile(0.2, axis=1)[0] * self.days / 365
        # base_mean_income = self.base['avg_income'].quantile(0.5, axis=1)[0] * self.days / 365
        # pm_mean_income = self.base['avg_income'].quantile(0.5, axis=1)[0] * self.days / 365

        # pm_mean_cost = pm_tot_cost.iloc[-1, :].mean()
        # base_mean_cost = base_tot_cost.iloc[-1, :].mean()

        ''' Make total cost plots showing tap, bottle, and total cost '''
        exclude = ['TN460', 'TN459', 'TN458']
        print(
            self.basebw['cost']['bw_cost'].loc[
                [r for r in self.basebw['cost']['bw_cost'].index if r not in exclude],
                :
            ].std(axis=0)
        )
        print(self.pm['cost']['bw_cost'].mean(axis=0))
        print(self.basebw['cost']['tw_cost'].mean(axis=0))
        print(self.pm['cost']['tw_cost'].mean(axis=0))
        print(self.basebw['cost']['total'].mean(axis=0))
        print(self.pm['cost']['total'].mean(axis=0))

        plt.boxplot(self.base['cost']['total'].groupby(level=0).mean())
        plt.savefig(self.pub_loc + 'cost_box.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        cost_leg = ['Bottled Water', 'Tap Water', 'Total']
        cost_b = pd.concat(
            [self.basebw['cost']['bw_cost'].mean(axis=0),
             self.basebw['cost']['tw_cost'].mean(axis=0),
             self.basebw['cost']['total'].mean(axis=0)],
            axis=1,
            keys=cost_leg
        )
        cost_p = pd.concat(
            [self.pm['cost']['bw_cost'].mean(axis=0),
             self.pm['cost']['tw_cost'].mean(axis=0),
             self.pm['cost']['total'].mean(axis=0)],
            axis=1,
            keys=cost_leg
        )

        fig, axes = plt.subplots(1, 2, sharey=True)
        axes[0] = self.make_avg_plot(
            axes[0], cost_b, None, cost_leg,
            cost_b.index / 24, sd_plot=False
        )
        axes[1] = self.make_avg_plot(
            axes[1], cost_p, None, cost_leg,
            cost_b.index / 24, sd_plot=False
        )

        fmt = '${x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        axes[0].yaxis.set_major_formatter(tick)

        plt.gcf().set_size_inches(7, 3.5)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Cost', x=0.04)
        axes[0].legend(cost_leg, loc='upper left')
        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)

        plt.savefig(self.pub_loc + 'cost.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Make cowpi plots (boxplots or barcharts) '''
        print(self.base['cowpi'][self.base['cowpi']['level'] == 1])

        cowpi_b = [
            self.base['cowpi'][self.base['cowpi']['level'] == 1]['cowpi'].groupby(level=0).mean()*100,
            self.base['cowpi'][self.base['cowpi']['level'] == 2]['cowpi'].groupby(level=0).mean()*100,
            self.base['cowpi'][self.base['cowpi']['level'] == 3]['cowpi'].groupby(level=0).mean()*100
        ]

        plt.boxplot(cowpi_b)
        plt.savefig(self.pub_loc + 'cow_boxplot.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        level_cowpi_b = self.base['cowpi'].groupby('level').mean()['cowpi']
        std_cowpi_b = self.base['cowpi'].groupby('level').std()['cowpi']
        # print(level_cowpi_b)

        level_cowpi_bbw = self.basebw['cowpi'].groupby('level').mean()['cowpi']
        std_cowpi_bbw = self.basebw['cowpi'].groupby('level').std()['cowpi']
        # print(level_cowpi_bbw)

        level_cowpi_p = self.pm['cowpi'].groupby('level').mean()['cowpi']
        std_cowpi_p = self.pm['cowpi'].groupby('level').std()['cowpi']
        # print(level_cowpi_p)

        level_cowpi_p25 = self.pm25ind['cowpi'].groupby('level').mean()['cowpi']
        level_cowpi_p50 = self.pm50ind['cowpi'].groupby('level').mean()['cowpi']
        level_cowpi_p75 = self.pm75ind['cowpi'].groupby('level').mean()['cowpi']
        level_cowpi_p100 = self.pm100ind['cowpi'].groupby('level').mean()['cowpi']

        std_cowpi_p25 = self.pm25ind['cowpi'].groupby('level').std()['cowpi']
        std_cowpi_p50 = self.pm50ind['cowpi'].groupby('level').std()['cowpi']
        std_cowpi_p75 = self.pm75ind['cowpi'].groupby('level').std()['cowpi']
        std_cowpi_p100 = self.pm100ind['cowpi'].groupby('level').std()['cowpi']

        cost_comp_basepm = pd.DataFrame(
            {'Base': level_cowpi_b,
             'Base+BW': level_cowpi_bbw,
             'Social Distancing+BW': level_cowpi_p},
            index=[0, 1, 2, 3]
        )

        cost_std_basepm = pd.DataFrame(
            {'Base': std_cowpi_b,
             'Base+BW': std_cowpi_bbw,
             'Social Distancing+BW': std_cowpi_p},
            index=[0, 1, 2, 3]
        )

        # convert to percentages
        cost_comp_basepm = cost_comp_basepm * 100
        cost_std_basepm = cost_std_basepm
        print(cost_comp_basepm)
        print(cost_std_basepm)

        cost_comp_basepm = cost_comp_basepm.rename({0: 'Extremely Low', 1: 'Low', 2: 'Medium', 3: 'High'})
        cost_std_basepm = cost_std_basepm.rename({0: 'Extremely Low', 1: 'Low', 2: 'Medium', 3: 'High'})

        # make the barchart
        self.make_cowpi_plot(cost_comp_basepm, cost_std_basepm, 'basepm')

        cost_comp_sa = pd.DataFrame(
            {'No Minimum': level_cowpi_p,
             '25%': level_cowpi_p25,
             '50%': level_cowpi_p50,
             '75%': level_cowpi_p75,
             '100%': level_cowpi_p100,
             'Base': level_cowpi_b},
            index=[0, 1, 2, 3]
        )

        cost_std_sa = pd.DataFrame(
            {'No Minimum': std_cowpi_p,
             '25%': std_cowpi_p25,
             '50%': std_cowpi_p50,
             '75%': std_cowpi_p75,
             '100%': std_cowpi_p100,
             'Base': std_cowpi_b},
            index=[0, 1, 2, 3]
        )

        # convert to percentages
        cost_comp_sa = cost_comp_sa * 100
        cost_std_sa = cost_std_sa

        cost_comp_sa = cost_comp_sa.rename({0: 'Extremely Low', 1: 'Low', 2: 'Medium', 3: 'High'})
        cost_std_sa = cost_std_sa.rename({0: 'Extremely Low', 1: 'Low', 2: 'Medium', 3: 'High'})

        # make barchart
        self.make_cowpi_plot(cost_comp_sa, cost_std_sa, 'sa')

    def make_twa_plots(self):
        '''
        Tap water avoidance adoption plots
        '''
        # print(self.pm['twa'])
        twas = ['Drink', 'Cook']
        print(self.basebw['twa'])
        twa_basebw = self.calc_twa_averages(self.basebw['twa'], twas)
        twa_basebw.index = twa_basebw.index - 719
        twa_basebw.loc[0] = [0, 0]
        twa_basebw.sort_index(inplace=True)

        twa_pm = self.calc_twa_averages(self.pm['twa'], twas)
        twa_pm.index = twa_pm.index - 719
        twa_pm.loc[0] = [0, 0]
        twa_pm.sort_index(inplace=True)

        households = len(self.pm['twa']['drink'].index)
        # print(twa_pm)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        axes[0] = self.make_avg_plot(
            axes[0], twa_basebw / households * 100, None,
            twas, twa_basebw.index / 24,
            sd_plot=False
        )
        axes[1] = self.make_avg_plot(
            axes[1], twa_pm / households * 100, None,
            twas, twa_pm.index / 24,
            sd_plot=False
        )
        axes[0].legend(twas)

        axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
                     transform=axes[0].transAxes)
        axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
                     transform=axes[1].transAxes)
        fig.supxlabel('Time (days)', y=-0.03)
        fig.supylabel('Percent of Households', x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(self.pub_loc + 'twa_comp.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

    def income_plots(self):
        '''
        Make plot of Clinton, NC income by BG and micropolis income
        '''
        ax0 = plt.subplot()
        ax0 = self.clinton_bg_plot(ax0)

        income = self.base['income'].iloc[:, 0].groupby(level=0).median()
        dist_income = pd.concat(
            [income, pd.Series(self.ind_distances)],
            axis=1,
            keys=['Income', 'Distance']
        )
        print(dist_income)
        model, x = self.linear_regression(dist_income, 'Distance', 'Income')

        ax0.plot(x, model.predict(x))
        ax0.set(
            xlabel='Minimum Industrial Distance (normalized)',
            ylabel='Median Income'
        )

        # format the y axis ticks to have a dollar sign and thousands commas
        fmt = '${x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax0.yaxis.set_major_formatter(tick)

        ax0.legend(['Clinton, NC', 'Micropolis'])
        plt.savefig(self.pub_loc + 'income_bg-micropolis.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Income map '''
        loc = 'Output Files/30_base-bw_equity/2024-07-04_14-31_0_results/'
        comp_list = ['income']
        data = ut.read_data(loc, comp_list)

        income = data['income'].loc[:, 'income'].groupby(level=0).mean()

        ax1 = plt.subplot()
        ax1 = wntr.graphics.plot_network(
            self.wn,
            node_attribute=income,
            node_size=5, node_range=[0, 200000], node_colorbar_label='Income ($)',
            ax=ax1
        ) 
        plt.gcf().set_size_inches(4, 3.5)
        plt.savefig(self.pub_loc + 'income_map.' + self.format,
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
                                      'drink',
                                      'cook',
                                      'traditional',
                                      'burden']
        data = ut.read_data(loc, comp_list)
        # print(data['age'].loc[15559200, self.res_nodes].notna().sum())
        # for i, val in data['age'].items():
        #     if 'TN' in i:
        #         print(f"{i}: {val.iloc[-1] / 3600}")
        # for i in data['age'].loc[:, 'TN49']:
        #     print(i/3600)
        # for i in data['demand'].loc[:, 'TN49']:
        #     print(i)
        households = len(data['income'])
        warmup = data['bw_cost'].index[-1] - x_len
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
        plt.savefig(loc + 'aggregate_demand' + '.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        cols = ['Residential', 'Commercial', 'Industrial']
        demand_res = data['demand'].loc[:, self.res_nodes].sum(axis=1)
        demand_ind = data['demand'].loc[:, self.ind_nodes].sum(axis=1)
        demand_com = data['demand'].loc[:, self.com_nodes].sum(axis=1)
        demand = pd.concat([demand_res.rolling(24).mean(),
                            demand_ind.rolling(24).mean(),
                            demand_com.rolling(24).mean()],
                           axis=1, keys=cols)
        x_values = np.array([
            x for x in np.arange(0, days, days / x_len)
        ])
        ax = plt.subplot()
        self.make_avg_plot(
            ax, demand.iloc[-x_len:], sd=None, cols=cols, x_values=x_values,
            sd_plot=False
        )
        plt.savefig(loc + 'mean_res_demand' + '.' + self.format,
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
        data['income'] = data['income']
        self.make_heatmap(
            data['tot_cost'].T,
            'Time (weeks)',
            'Household',
            loc + 'tot_cost_heatmap',
            0.01
        )

        cols = ['Tap Water', 'Bottled Water', 'Total']
        cost = pd.concat([data['tw_cost'].mean(axis=1),
                          data['bw_cost'].mean(axis=1),
                          data['tot_cost'].mean(axis=1)],
                         axis=1, keys=cols)
        cost_max = pd.concat([data['tw_cost'].max(axis=1),
                              data['bw_cost'].max(axis=1),
                              data['tot_cost'].max(axis=1)],
                             axis=1, keys=cols)

        # average cost plot
        ax = plt.subplot()
        self.make_avg_plot(
            ax, cost, None, cols, (cost.index - warmup) / 24,
            'Time (Days)', 'Mean Water Cost ($)', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'mean_water_cost.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        # max cost plot
        ax = plt.subplot()
        self.make_avg_plot(
            ax, cost_max, None, cols, (cost_max.index - warmup) / 24,
            'Time (Days)', 'Maximum Water Cost ($)', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'max_water_cost.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Plot the income by node '''
        fig, axes = plt.subplots(1, 2)
        ind_distances, ind_closest = self.calc_industry_distance(self.wn)
        dist_values = [v for k, v in ind_distances.items() if k in data['income'].index]
        income = data['income'].loc[:, 'income'].groupby(level=0).mean()
        print(len(dist_values))
        print(len(income))

        axes[0] = self.scatter_plot(dist_values, income, axes[0])

        axes[1] = wntr.graphics.plot_network(
            self.wn,
            node_attribute=income,
            node_size=5, node_range=[0, 200000], node_colorbar_label='Income ($)',
            ax=axes[1]
        )
        plt.gcf().set_size_inches(7, 3.5)
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
        twa = pd.concat([data['drink'].sum(axis=1),
                         data['cook'].sum(axis=1)],
                        axis=1, keys=['Drink', 'Cook'])

        ax = plt.subplot()
        self.make_avg_plot(
            ax, twa / households * 100, None, ['Drink', 'Cook'], (twa.index / 24) - 30,
            'Time (days)', 'Percent of Households', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'twa.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' Equity metric costs '''
        metrics = pd.concat([data['traditional'],
                             data['burden']],
                            axis=1, keys=['Traditional', 'Burden'])

        ax = plt.subplot()
        self.make_avg_plot(
            ax, metrics * 100, None, ['Traditional', 'Burden'],
            (metrics.index - warmup) / 24,
            'Time (days)', '% of Income', show_labels=True, sd_plot=False
        )

        plt.savefig(loc + 'equity_metrics.' + self.format,
                    format=self.format, bbox_inches='tight')
        plt.close()

        ''' % of income figure by income level '''
        # print('Income')
        # print(data['income'].T[0])
        # print('Total cow')
        # print(data['tot_cost'].iloc[-1, :])
        # cowpi = pd.concat(
        #     [data['income'.T[0]],
        #      data['tot_cost'].iloc[-1, :] / (data['income'].T[0] * self.days / 365)],
        #     axis=1, keys=['Income', 'COWPI']
        # )

        # print('Cowpi')
        # print(cowpi * 100)

        # cowpi_levels = pd.concat(
        #     [cowpi[cowpi['income'] > ]]
        # )
