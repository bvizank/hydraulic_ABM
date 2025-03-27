import math
import wntr
import os
import numpy as np
import pandas as pd
import geopandas
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from scipy.stats import binned_statistic
import utils as ut
import city_info as ci
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


class BaseGraphics:
    def __init__(self):
        pass

    def calc_difference(self, data_time_1, data_time_2):
        """Function to take the difference between two time points"""
        return data_time_2 - data_time_1

    def get_times(self, pm):
        max_pm = pm["avg_seir_data"].wfh.loc[int(pm["avg_seir_data"].wfh.idxmax())]
        times = []
        times = times + [pm["avg_seir_data"].wfh.searchsorted(max_pm / 10)]
        times = times + [pm["avg_seir_data"].wfh.searchsorted(max_pm / 2)]
        times = times + [pm["avg_seir_data"].wfh.searchsorted(max_pm)]
        self.times_hour = [time % 24 for time in times]

        self.times = times

    def get_nodes(self, wn):
        data = pd.read_excel("Input Files/micropolis/Micropolis_pop_at_node.xlsx")
        node_capacity = dict(
            zip(data["Node"].tolist(), data["Max Population"].tolist())
        )
        self.res_nodes = [
            name
            for name, node in wn.junctions()
            if node.demand_timeseries_list[0].pattern_name == "2"
            and node.demand_timeseries_list[0].base_value > 0
            and node_capacity[name] != 0
        ]
        self.ind_nodes = [
            name
            for name, node in wn.junctions()
            if node.demand_timeseries_list[0].pattern_name == "3"
            and node.demand_timeseries_list[0].base_value > 0
        ]
        self.com_nodes = [
            name
            for name, node in wn.junctions()
            if (
                node.demand_timeseries_list[0].pattern_name == "4"
                or node.demand_timeseries_list[0].pattern_name == "5"
                or node.demand_timeseries_list[0].pattern_name == "6"
            )
            and node.demand_timeseries_list[0].base_value > 0
            and node_capacity[name] != 0
        ]
        self.rest_nodes = [
            name
            for name, node in wn.junctions()
            if node.demand_timeseries_list[0].pattern_name == "1"
            and node.demand_timeseries_list[0].base_value > 0
            and node_capacity[name] != 0
        ]
        self.all_nodes = [
            name
            for name, node in wn.junctions()
            if node.demand_timeseries_list[0].base_value > 0
            and node_capacity[name] != 0
        ]

        self.ind_nodes_obj = [
            name
            for name, node in wn.junctions()
            if node.demand_timeseries_list[0].pattern_name == "3"
        ]

    def calc_sec_averages(self, data, op="mean", rolling=True):
        """
        Calculate the average of the input data by sector
        """
        output = dict()
        if op == "mean":
            output["res"] = data[self.res_nodes].mean(axis=1)
            output["com"] = data[self.com_nodes].mean(axis=1)
            output["ind"] = data[self.ind_nodes].mean(axis=1)
        elif op == "sum":
            output["res"] = data[self.res_nodes].sum(axis=1)
            output["com"] = data[self.com_nodes].sum(axis=1)
            output["ind"] = data[self.ind_nodes].sum(axis=1)
        else:
            raise NotImplementedError(f"Operation {op} not implemented.")

        if rolling:
            cols = ["Residential", "Commercial", "Industrial"]
            output = pd.concat(
                [
                    output["res"].rolling(24).mean(),
                    output["com"].rolling(24).mean(),
                    output["ind"].rolling(24).mean(),
                ],
                axis=1,
                keys=cols,
            )

        return output

    def calc_twa_averages(self, data, keys):
        """
        Calculate the average and package TWA data for plotting
        """
        average = dict()
        variance = dict()
        for i in keys:
            tmp = data[i].groupby("i").sum() / data[i].groupby("i").count()
            average[i] = tmp.mean(axis=0)
            variance[i] = tmp.var(axis=0)

        average = pd.DataFrame(average)
        variance = pd.DataFrame(variance)

        # average = pd.concat(
        #     [data['drink'].groupby('i').sum().mean(axis=0),
        #      data['cook'].groupby('i').sum().mean(axis=0),
        #      data['hygiene'].groupby('i').sum().mean(axis=0)],
        #     axis=1, keys=keys
        # )

        # variance = pd.concat(
        #     [data['drink'].groupby('i').sum().var(axis=0),
        #      data['cook'].groupby('i').sum().var(axis=0),
        #      data['hygiene'].groupby('i').sum().var(axis=0)],
        #     axis=1, keys=keys
        # )

        return average, variance

    def filter_demo(self, level, filter, out, mult=1, data=None):
        """Filter cow cowpi data by demographic variable"""
        if data is None:
            data = [self.base, self.basebw, self.pm_nobw, self.pm]
        cowpi_filter = list()
        cowpi_nonfilter = list()
        for d in data:
            cowpi_filter.append(
                d["cowpi"][(d["cowpi"]["level"] == level) & (d["cowpi"][filter])][out]
                * mult
                # self.basebw["cowpi"][
                #     (self.basebw["cow pi"]["level"] == level)
                #     & (self.basebw["cowpi"][filter])
                # ][out]
                # * mult,
                # self.pm_nobw["cowpi"][
                #     (self.pm_nobw["cowpi"]["level"] == level)
                #     & (self.pm_nobw["cowpi"][filter])
                # ][out]
                # * mult,
                # self.pm["cowpi"][
                #     (self.pm["cowpi"]["level"] == level) & (self.pm["cowpi"][filter])
                # ][out]
                # * mult,
            )

            cowpi_nonfilter.append(
                d["cowpi"][(d["cowpi"]["level"] == level) & ~(d["cowpi"][filter])][out]
                * mult,
                # self.basebw["cowpi"][
                #     (self.basebw["cowpi"]["level"] == level)
                #     & ~(self.basebw["cowpi"][filter])
                # ][out]
                # * mult,
                # self.pm_nobw["cowpi"][
                #     (self.pm_nobw["cowpi"]["level"] == level)
                #     & ~(self.pm_nobw["cowpi"][filter])
                # ][out]
                # * mult,
                # self.pm["cowpi"][
                #     (self.pm["cowpi"]["level"] == level) & ~(self.pm["cowpi"][filter])
                # ][out]
                # * mult,
            )

        data = {
            filter: cowpi_filter,
            "non" + filter: cowpi_nonfilter,
        }

        return data

    def threshold_demo(self, filter, threshold=0.046, column_out=False):
        """Filter cow cowpi data by demographic variable"""
        if column_out:
            data = [
                self.base["cowpi"][
                    (self.base["cowpi"]["cowpi"] > threshold)
                    & (self.base["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.basebw["cowpi"][
                    (self.basebw["cowpi"]["cowpi"] > threshold)
                    & (self.basebw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm_nobw["cowpi"][
                    (self.pm_nobw["cowpi"]["cowpi"] > threshold)
                    & (self.pm_nobw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm["cowpi"][
                    (self.pm["cowpi"]["cowpi"] > threshold) & (self.pm["cowpi"][filter])
                ]["cowpi"]
                * 100,
            ]
        else:
            cowpi_filter = [
                self.base["cowpi"][
                    (self.base["cowpi"]["cowpi"] > threshold)
                    & (self.base["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.basebw["cowpi"][
                    (self.basebw["cowpi"]["cowpi"] > threshold)
                    & (self.basebw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm_nobw["cowpi"][
                    (self.pm_nobw["cowpi"]["cowpi"] > threshold)
                    & (self.pm_nobw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm["cowpi"][
                    (self.pm["cowpi"]["cowpi"] > threshold) & (self.pm["cowpi"][filter])
                ]["cowpi"]
                * 100,
            ]

            cowpi_nonfilter = [
                self.base["cowpi"][
                    (self.base["cowpi"]["cowpi"] > threshold)
                    & ~(self.base["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.basebw["cowpi"][
                    (self.basebw["cowpi"]["cowpi"] > threshold)
                    & ~(self.basebw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm_nobw["cowpi"][
                    (self.pm_nobw["cowpi"]["cowpi"] > threshold)
                    & ~(self.pm_nobw["cowpi"][filter])
                ]["cowpi"]
                * 100,
                self.pm["cowpi"][
                    (self.pm["cowpi"]["cowpi"] > threshold)
                    & ~(self.pm["cowpi"][filter])
                ]["cowpi"]
                * 100,
            ]

            data = {
                "low": cowpi_filter,
                "high": cowpi_nonfilter,
            }

        return data

    def calc_risk(self, ie, it, ce, ct):
        """First calculate risk ratio"""
        rr = (ie * ct) / (ce * it)

        arr = (ie / it) - (ce / ct)

        return rr, arr

    def plot_demand_by_node(self, ax, nodes):
        """Plot the sum of all demand in the network of the given nodes"""
        demand_base = self.base["avg_demand"].loc[:, nodes]
        demand_basebw = self.basebw["avg_demand"].loc[:, nodes]
        demand_pm = self.pm["avg_demand"].loc[:, nodes]
        demand_pm_nobw = self.pm_nobw["avg_demand"].loc[:, nodes]
        demand = pd.concat(
            [
                demand_base.sum(axis=1).rolling(24).mean(),
                demand_basebw.sum(axis=1).rolling(24).mean(),
                demand_pm_nobw.sum(axis=1).rolling(24).mean(),
                demand_pm.sum(axis=1).rolling(24).mean(),
            ],
            axis=1,
            keys=["Base", "TWA", "PM", "TWA+PM"],
        )

        var_base = self.base["var_demand"].loc[:, nodes]
        var_basebw = self.basebw["var_demand"].loc[:, nodes]
        var_pm = self.pm["var_demand"].loc[:, nodes]
        var_pm_nobw = self.pm_nobw["var_demand"].loc[:, nodes]
        demand_var = pd.concat(
            [
                var_base.sum(axis=1).rolling(24).mean(),
                var_basebw.sum(axis=1).rolling(24).mean(),
                var_pm_nobw.sum(axis=1).rolling(24).mean(),
                var_pm.sum(axis=1).rolling(24).mean(),
            ],
            axis=1,
            keys=["Base", "TWA", "PM", "TWA+PM"],
        )

        demand_err = ut.calc_error(demand_var, self.error)

        # format the y axis ticks to have a dollar sign and thousands commas
        # fmt = "{x:,.0f}"
        # tick = mtick.StrMethodFormatter(fmt)
        # ax.yaxis.set_major_formatter(tick)

        ax = self.make_avg_plot(
            ax,
            demand,
            demand_err,
            ["Base", "TWA", "PM", "TWA+PM"],
            self.x_values_hour,
            # xlabel="Time (days)",
            # ylabel="Demand (L/s)",
            show_labels=False,
        )

        return ax

    def plot_demand_res_nonres(self, ax, data, nodes_w_demand, legend_bool=False):
        res_data = data["demand"]["tw_demand"].groupby("i").sum()
        res_data = res_data.iloc[:, -(self.days + 1) :]
        for i in range(int((len(res_data.columns) - 1) / 30)):
            if i != 0:
                res_data.insert((31 * i), str(i), 0)
        res_data = res_data.iloc[:, ::-1].apply(
            lambda x: x - x.shift(-1, axis=0), axis=1
        )
        for i in range(int((len(res_data.columns) - 1) / 30)):
            if i != 0:
                res_data = res_data.drop(str(i), axis=1)

        res_data = res_data.iloc[:, ::-1]
        # res_data = res_data.iloc[:, -self.days :]
        # res_sd = res_data.std(axis=0).reset_index(drop=True)
        res_data = res_data.mean(axis=0).reset_index(drop=True)
        # res_sd = res_sd.drop(0).reset_index(drop=True)
        res_data = res_data.drop(0).reset_index(drop=True)

        nonres_data = (
            data["avg_demand"][nodes_w_demand].sum(axis=1) * 3600
        ).reset_index(drop=True)
        nonres_data = nonres_data.groupby(nonres_data.index // 24).sum()
        nonres_data = nonres_data - res_data

        print(nonres_data)
        print(res_data)

        # nonres_var = (
        #     data["var_demand"][nodes_w_demand].sum(axis=1) * 3600
        # ).reset_index(drop=True)
        # nonres_var = nonres_var.groupby(nonres_var.index // 24).sum()
        # nonres_var = nonres_var - res_data

        # nonres_sd = ut.calc_error(nonres_var, self.error)

        all_data = pd.concat(
            [res_data, nonres_data], axis=1, keys=["Residential", "Non-residential"]
        )

        all_data /= 1000000

        # all_sd = pd.concat(
        #     [res_sd, nonres_sd], axis=1, keys=["Residential", "Non-residential"]
        # )

        # ax = self.make_avg_plot(
        #     ax=ax,
        #     data=all_data,
        #     sd=all_sd,
        #     cols=["Residential", "Non-residential"],
        #     x_values=self.x_values_day,
        # )

        ax = all_data.plot.area(ax=ax, legend=legend_bool)
        ax.set_xlabel("")

        return ax

    def plot_demand_by_case(self, ax, data, perc_counts, legend_bool=False):
        demand_res = data.loc[
            :,
            perc_counts[
                (perc_counts["type"] == "res") | (perc_counts["type"] == "mfh")
            ].index,
        ]
        demand_ind = data.loc[:, perc_counts[perc_counts["type"] == "ind"].index]
        demand_mfh = data.loc[:, perc_counts[perc_counts["type"] == "mfh"].index]
        demand_com = data.loc[:, perc_counts[perc_counts["type"] == "com"].index]
        demand_caf = data.loc[:, perc_counts[perc_counts["type"] == "caf"].index]
        demand_gro = data.loc[:, perc_counts[perc_counts["type"] == "gro"].index]
        demand = pd.concat(
            [
                demand_res.sum(axis=1).rolling(24).mean().reset_index(drop=True),
                demand_ind.sum(axis=1).rolling(24).mean().reset_index(drop=True),
                demand_com.sum(axis=1).rolling(24).mean().reset_index(drop=True),
                demand_caf.sum(axis=1).rolling(24).mean().reset_index(drop=True),
                demand_gro.sum(axis=1).rolling(24).mean().reset_index(drop=True),
                pd.Series(self.x_values_hour),
            ],
            axis=1,
            keys=[
                "Residential",
                "Industrial",
                "Commercial",
                "Restaurant",
                "Grocery",
                "time",
            ],
        )

        demand = demand.set_index("time")
        print("Residential demand for the first day:")
        print((demand_res.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)
        print("MFH demand for the first day:")
        print((demand_mfh.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)
        print("Commercial demand for the first day:")
        print((demand_com.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)
        print("Industrial demand for the first day:")
        print((demand_ind.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)
        print("Cafe demand for the first day:")
        print((demand_caf.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)
        print("Grocery demand for the first day:")
        print((demand_gro.sum(axis=1) * 3600).sum() / 3.875 / 1000000 / self.days)

        # var_res = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "res"]]
        # var_mfh = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "mfh"]]
        # var_ind = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "ind"]]
        # var_com = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "com"]]
        # var_caf = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "caf"]]
        # var_gro = data["var_demand"].loc[:, perc_counts[perc_counts["type"] == "gro"]]
        # demand_var = pd.concat(
        #     [
        #         var_res.sum(axis=1).rolling(24).mean(),
        #         var_mfh.sum(axis=1).rolling(24).mean(),
        #         var_ind.sum(axis=1).rolling(24).mean(),
        #         var_com.sum(axis=1).rolling(24).mean(),
        #         var_caf.sum(axis=1).rolling(24).mean(),
        #         var_gro.sum(axis=1).rolling(24).mean(),
        #     ],
        #     axis=1,
        #     keys=["Res", "MFH", "Ind", "Com", "Caf", "Gro"],
        # )

        # demand_err = ut.calc_error(demand_var, self.error)

        ax = demand.plot.area(ax=ax, legend=legend_bool)
        ax.set_xlabel("")

        # format the y axis ticks to have a dollar sign and thousands commas
        # fmt = "{x:,.0f}"
        # tick = mtick.StrMethodFormatter(fmt)
        # ax.yaxis.set_major_formatter(tick)

        # ax = self.make_avg_plot(
        #     ax,
        #     demand,
        #     demand_err,
        #     ["Base", "TWA", "PM", "TWA+PM"],
        #     self.x_values_hour,
        #     # xlabel="Time (days)",
        #     # ylabel="Demand (L/s)",
        #     show_labels=False,
        # )

        return ax

    def calc_flow_diff(self, data, hours):
        flow_data = dict()
        flow_changes_sum = dict()
        for pipe, colData in data.items():
            if "MA" in pipe:
                curr_flow_changes = list()
                print(colData)
                for i in range(len(colData) - 1):
                    if colData[(i + 1) * 3600] * colData[i * 3600] < 0:
                        curr_flow_changes.append(1)
                    else:
                        curr_flow_changes.append(0)
                flow_changes_sum[pipe] = sum(curr_flow_changes[0:hours]) / 24
                flow_data[pipe] = curr_flow_changes

        # output = pd.DataFrame(flow_data)
        # change_sum = pd.Series(flow_changes_sum)

        return flow_data, flow_changes_sum
        # return output, change_sum

    def calc_distance(self, node1, node2):
        p1x, p1y = node1.coordinates
        p2x, p2y = node2.coordinates

        return math.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)

    def calc_age_diff(self, data, nodes_w_demand, threshold=130):
        out_dict = dict()
        for node, colData in data.items():
            out_data = colData.iloc[-1] / 3600
            if node in nodes_w_demand:
                out_dict[node] = out_data > threshold

        return out_dict

    def calc_industry_distance(self, wn):
        """
        Function to calculate the distance from each residential node
        to the nearest industrial node.
        """
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
        """
        Helper method for get_household. Collects data from n runs in folder
        """
        output = pd.DataFrame()
        files = os.listdir(folder)
        files.sort()
        for file in files:
            if param in file:
                curr_data = pd.read_pickle(os.path.join(folder, file))
                if param == "income":
                    output = pd.concat([output, curr_data])
                    print(param)
                    print(output)
                else:
                    output = pd.concat([output, curr_data.T])
                    print(param)
                    print(output)

        return output

    def get_household(self, folder, param, skip=[]):
        """
        Combine per household data from a group of n runs

        Parameters
        ----------
        folder : str
            data folder

        param : (list)
            params to combine
        """
        # output = dict()
        output = dict()
        # if isinstance(param, str):
        #     for i in range(30):
        #         if i not in skip:
        #             file = os.path.join(folder, param + '_' + str(i) + '.pkl')
        #             curr_data = pd.read_pickle(file)
        #             curr_data['i'] = i
        #             output = pd.concat([output, curr_data])
        # output[param] = self.collect_data(folder, param)
        # elif isinstance(param, list):
        for i in param:
            curr_param = pd.DataFrame()
            for j in range(30):
                if j not in skip:
                    file = os.path.join(folder, i + "_" + str(j) + ".pkl")
                    curr_data = pd.read_pickle(file)
                    if i != "income":
                        curr_data = curr_data.T
                    curr_data["i"] = j
                    """ Make a unique id that is a concat of hh_id and sim_id """
                    if i in [
                        "tw_cost",
                        "bw_cost",
                        "tw_demand",
                        "bw_demand",
                        "demo",
                        "drink",
                        "cook",
                        "hygiene",
                    ]:
                        curr_data["unique_id"] = curr_data.index.astype(str) + str(j)
                    elif i in ["income"]:
                        curr_data["unique_id"] = curr_data["id"].astype(str) + str(j)
                        curr_data.index.name = "wdn_node"
                        curr_data = curr_data.reset_index()

                    curr_data = curr_data.set_index("unique_id")
                    curr_param = pd.concat([curr_param, curr_data])

            output[i] = curr_param

        # print(output)
        # for item in param:
        #     output[item] = self.collect_data(folder, item)

        return output

    def convert_level(self, data):
        for i in range(30):
            perc_20 = data[data["i"] == i]["income"].quantile(0.20)

            data.loc[:, "level"] = np.where(data["income"] > perc_20, 1, 0)

    def package_household(self, data, dir, bw=True):
        """
        Package household data necessary for plotting later
        """
        # data interpolated from HUD extremely low income values using average
        # clinton household size of 2.56
        # extreme_income = 23452.8

        data["cost"] = self.get_household(
            dir + "/hh_results/", ["bw_cost", "tw_cost"] if bw else ["tw_cost"]
        )
        data["twa"] = self.get_household(
            dir + "/hh_results/", ["drink", "cook", "hygiene"]
        )
        data["income"] = self.get_household(dir + "/hh_results/", ["income"])["income"]

        data["demo"] = self.get_household(dir + "/hh_results/", ["demo"])

        data["demand"] = self.get_household(
            dir + "hh_results/", ["tw_demand", "bw_demand"] if bw else ["tw_demand"]
        )

        # for i in range(30):
        #     curr_income = ut.income_list(
        #         data=dt.clinton_income,
        #         n_house=len(data['income'][data['income']['i'] == i]),
        #         s=i
        #     )
        #     data['income'].loc[data['income']['i'] == i, 'income'] = (
        #         curr_income[0:len(data['income'][data['income']['i'] == i])]
        #     )

        #     # set the level as 1 if above 20 %-tile and 0 if below
        #     bot20 = np.percentile(
        #         data['income'].loc[data['income']['i'] == i, 'income'],
        #         20
        #     )
        #     data['income'].loc[data['income']['i'] == i, 'level'] = (
        #         np.where(
        #             data['income'].loc[data['income']['i'] == i, 'income'] > bot20,
        #             1, 0
        #         )
        #     )

        if not bw:
            data["cost"]["total"] = data["cost"]["tw_cost"]
        else:
            data["cost"]["total"] = (
                data["cost"]["bw_cost"].iloc[:, :-1]
                + data["cost"]["tw_cost"].iloc[:, :-1]
            )
            data["cost"]["total"]["i"] = data["cost"]["bw_cost"]["i"]
            # data["cost"]["total"]["unique_id"] = data["cost"]["bw_cost"]["unique_id"]

        self.convert_level(data["income"])

        data["cowpi"] = pd.concat(
            # "level": data["income"].loc[:, "level"].reset_index(drop=True),
            [
                data["income"].loc[:, "level"],
                (
                    data["cost"]["total"].iloc[:, -2]
                    / data["income"].loc[:, "income"]
                    * self.days
                    / 365
                ),
                data["cost"]["total"].iloc[:, -2],
                data["income"].loc[:, "income"],
                data["demo"]["demo"].loc[:, "white"],
                data["demo"]["demo"].loc[:, "hispanic"],
                data["demo"]["demo"].loc[:, "renter"],
                data["income"].loc[:, "i"],
                data["income"].loc[:, "wdn_node"],
            ],
            axis=1,
            keys=[
                "level",
                "cowpi",
                "cost",
                "income",
                "white",
                "hispanic",
                "renter",
                "i",
                "wdn_node",
            ],
        )

    def post_household(self):
        """
        Manipulate household data to be ready to plot
        """
        # get base data ready
        if "base" in self.scenarios:
            self.package_household(self.base, self.base_comp_dir, bw=False)

        # get base+bw data ready
        if "basebw" in self.scenarios:
            self.package_household(self.basebw, self.base_bw_comp_dir)

        # get pm data ready
        if "pm_nobw" in self.scenarios:
            self.package_household(self.pm_nobw, self.pm_nobw_comp_dir, bw=False)

        # get pm+bw data ready
        if "pm" in self.scenarios:
            self.package_household(self.pm, self.pm_comp_dir)

        if "sa" in self.scenarios:
            self.package_household(self.basebw_neg20, self.base_bw_neg20twa_comp_dir)
            self.package_household(self.basebw_20, self.base_bw_20twa_comp_dir)
            self.package_household(self.pm_neg20, self.pm_neg20twa_comp_dir)
            self.package_household(self.pm_20, self.pm_20twa_comp_dir)

        # get pm_nodi data ready
        # self.package_household(self.pm_nodi, self.pm_nodi_comp_dir)

        # get pm_perc data ready
        # self.package_household(self.pm_perc, self.pm_comp_perc_dir)

        # get pm_noD data ready
        # self.package_household(self.pm_noD, self.pm_comp_noD_dir)

        # get pm_noC data ready
        # self.package_household(self.pm_noC, self.pm_comp_noC_dir)

        # get pm_noH data ready
        # self.package_household(self.pm_noH, self.pm_comp_noH_dir)

        # get pm 25ind data ready
        # self.package_household(self.pm25ind, self.pm_25ind_comp_dir)

        # get pm 50ind data ready
        # self.package_household(self.pm50ind, self.pm_50ind_comp_dir)

        # get pm 75ind data ready
        # self.package_household(self.pm75ind, self.pm_75ind_comp_dir)

        # get pm 100ind data ready
        # self.package_household(self.pm100ind, self.pm_100ind_comp_dir)

    def make_avg_plot(
        self,
        ax,
        data,
        sd,
        cols,
        x_values,
        xlabel=None,
        ylabel=None,
        fig_name=None,
        show_labels=False,
        logx=False,
        sd_plot=True,
    ):
        """
        Function to plot data with error.

        Parameters:
            data (pd.DataFrame): data to be plotted
            sd (pd.DataFrame): error of data to be plotted
            xlabel (string): x label for figure
            ylabel (string): y label for figure
            fig_name (string): save name for figure
            x_values (list): x values for the plot
        """

        """ Plot a single figure with the input data """
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
            ax.plot(x_values, data[col], color="C" + str(i * m))

        if sd_plot:
            #  need to separate so that the legend fills correctly
            for i, col in enumerate(cols):
                ax.fill_between(
                    x_values,
                    data[col] - sd[col],
                    data[col] + sd[col],
                    color="C" + str(i * m),
                    alpha=0.5,
                )

        if show_labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(cols)

        if logx:
            ax.xscale("log")

        return ax

    def make_flow_plot(
        self,
        change_data,
        sum_data,
        percent,
        dir,
        legend_text,
        title,
        change_data2=None,
        sum_data2=None,
        days=90,
        ax=None,
    ):
        """
        Function to make a plot showing the flow direction change
        """

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
        x_values = np.array([x for x in np.arange(0, days, days / len(roll_change))])

        if dir == "top":
            y_1 = roll_change[sum_data[sum_data > percentiles].index]
            y_2 = roll_change2[sum_data2[sum_data2 > percentiles2].index]
        elif dir == "bottom":
            y_1 = roll_change[sum_data[sum_data < percentiles].index]
            y_2 = roll_change2[sum_data2[sum_data2 < percentiles2].index]
        elif dir == "middle":
            pipes = sum_data[sum_data > percentiles[percent[0]]]
            pipes = sum_data[sum_data < percentiles[percent[1]]]
            y_1 = roll_change[pipes.index]
            y_2 = roll_change2[pipes.index]

        if ax is None:
            plt.plot(x_values, y_1.mean(axis=1), color="C" + str(0))
            plt.plot(x_values, y_2.mean(axis=1), color="C" + str(2))
            plt.xlabel("Time (days)")
            plt.ylabel("Daily Average Flow Changes")
            plt.legend(legend_text)
            plt.savefig(
                self.pub_loc + title + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()
        else:
            ax.plot(x_values, y_1.mean(axis=1), color=self.prim_colors[0])
            ax.plot(x_values, y_2.mean(axis=1), color=self.prim_colors[2])

    def export_agent_loc(self, wn, output_loc, locations):
        res_loc = locations[self.res_nodes].sum(axis=1)
        ind_loc = locations[self.ind_nodes].sum(axis=1)
        com_loc = locations[self.com_nodes].sum(axis=1)
        rest_loc = locations[self.rest_nodes].sum(axis=1)
        output = pd.DataFrame(
            {"res": res_loc, "ind": ind_loc, "com": com_loc, "rest": rest_loc}
        )
        output.to_csv(output_loc + "locations.csv")

    def make_seir_plot(self, days):
        """Function to make the seir plot with the input columns"""
        base_data = dcp(self.base["avg_seir_data"])
        pm_data = dcp(self.pm["avg_seir_data"])
        base_sd = dcp(ut.calc_error(self.base["var_seir_data"], self.error))
        pm_sd = dcp(ut.calc_error(self.pm["var_seir_data"], self.error))
        base_data = base_data * 100
        pm_data = pm_data * 100
        base_sd = base_sd * 100
        pm_sd = pm_sd * 100

        input = ["S", "E", "I", "R", "wfh"]
        leg_text = ["Susceptible", "Exposed", "Infected", "Recovered", "WFH"]

        x_values = np.array([x for x in np.arange(0, days, days / len(base_data))])

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        axes[0] = self.make_avg_plot(axes[0], base_data, base_sd, input, x_values)
        axes[1] = self.make_avg_plot(axes[1], pm_data, pm_sd, input, x_values)

        plt.ylim(0, 100)
        plt.gcf().set_size_inches(7, 3.5)
        fig.supxlabel("Time (days)", y=-0.03)
        fig.supylabel("Percent Population", x=0.04)
        axes[0].legend(leg_text, loc="upper left")
        axes[0].text(
            0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )

        plt.savefig(
            self.pub_loc + "/" + "seir_" + self.error + "." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def make_distance_plot(self, x, y1, y2, sd1, sd2, xlabel, ylabel, name, data_names):
        """
        Make scatter plot plus binned levels of input data. Accepts one x
        vector and two y vectors.
        """
        mean_y1 = binned_statistic(
            x, y1, statistic="mean", bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
        )
        mean_y2 = binned_statistic(
            x, y2, statistic="mean", bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
        )

        sd_y1 = binned_statistic(
            x, sd1, statistic="mean", bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
        )
        sd_y2 = binned_statistic(
            x, sd2, statistic="mean", bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
        )
        sd_y1 = ut.calc_error(sd_y1.statistic, self.error) / 3600
        sd_y2 = ut.calc_error(sd_y2.statistic, self.error) / 3600
        # print(mean_y1.bin_edges)

        bin_names = [
            "0-500",
            "500-1000",
            "1000-1500",
            "1500-2000",
            "2000-2500",
            "2500-3000",
            "3000-3500",
        ]

        data_dict = {data_names[0]: mean_y1.statistic, data_names[1]: mean_y2.statistic}

        sd_dict = {data_names[0]: sd_y1, data_names[1]: sd_y2}

        bar_x = np.arange(len(bin_names))
        width = 0.25  # width of the bars
        multiplier = 0  # iterator

        fig, ax = plt.subplots(layout="constrained")

        for attribute, measurement in data_dict.items():
            offset = width * multiplier
            ax.bar(
                bar_x + offset,
                measurement,
                width,
                label=attribute,
                color="C" + str(multiplier * 2),
                yerr=sd_dict[attribute],
                error_kw=dict(lw=0.5, capsize=2, capthick=0.5),
            )
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.legend()
        ax.set_xticks(bar_x + width, bin_names, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.savefig(
            self.pub_loc + name + "." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def make_heatmap(self, data, xlabel, ylabel, name, aspect):
        """heatmap plot of all agents"""
        fig, ax = plt.subplots()
        im = ax.imshow(
            data, aspect=aspect
        )  # for 100 agents: 0.03, for 1000 agents: 0.003
        ax.figure.colorbar(im, ax=ax)
        # plt.xlim(1, data.shape[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # x_tick_labels = [0, 20, 40, 60, 80]
        # ax.set_xticks([i * 24 for i in x_tick_labels])
        # ax.set_xticklabels(x_tick_labels)

        plt.savefig(name + "." + self.format, format=self.format, bbox_inches="tight")
        plt.close()

    def calc_model_stats(self, wn, seir, age):
        """
        Function for calculating comparison stats for decision models:
            - average water age at the end of the simluation
            - peak infection rate
            - final susceptible count
            - peak exposure rate
        """
        age_data = getattr(age[self.nodes], "mean")(axis=1)
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

    def make_income_comp_plot(
        self,
        data,
        name="",
        xlabel=None,
        ylabel="%HI",
        box=2,
        outliers=None,
        means=True,
        income_line=4.6,
        legend_kwds={},
        ax=None,
    ):
        if box == 2:
            fig, axes = plt.subplots(1, 2, sharey=True)

            axes[0].boxplot(data["low"], sym=outliers, showmeans=means)
            axes[1].boxplot(data["high"], sym=outliers, showmeans=means)

            # set the x ticks for each subplot
            for ax in axes:
                ax.set_xticklabels(xlabel, rotation=45)

            # add a red dashed line at 4.6%
            if income_line:
                for ax in axes:
                    ax.axhline(y=income_line, color="r", linestyle="dashed")

            # set the ylabel
            axes[0].set_ylabel(ylabel)

            # add the subplot labels
            axes[0].text(
                0.5, -0.24, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.24, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )

            plt.gcf().set_size_inches(7, 3.5)
            plt.savefig(
                self.pub_loc + name + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()
        elif box == 1:
            if ax is None:
                ax = plt.subplot()
                print_fig = True
            else:
                print_fig = False

            xlocations = range(len(data[0]))
            width = (1 - 0.2) / len(data)
            # print(xlocations)

            axes = list()
            for i, block in enumerate(data):
                pos = [x + (width * i) + 0.01 for x in xlocations]
                # print(f"Median for first block: {block[0].median()}")
                axes.append(
                    ax.boxplot(
                        block,
                        sym=outliers,
                        showmeans=means,
                        widths=width,
                        positions=pos,
                        manage_ticks=False,
                        patch_artist=True,
                        boxprops={"facecolor": "C" + str(i)},
                        medianprops={"color": "black"},
                    )
                )

            # add a red dashed line at 4.6%
            if income_line:
                ax.axhline(y=income_line, color="r", linestyle="dashed")

            # ax.set_xmargin(-0.04)

            # set the x tick location and labels
            ax.set_xticks([x + len(data) / 2 * width - (width / 2) for x in xlocations])
            ax.set_xticklabels(xlabel, rotation=0)
            ax.legend(
                [a["boxes"][0] for a in axes],
                legend_kwds["labels"],
                loc=legend_kwds["loc"],
            )

            ax.set_ylabel(ylabel)

            if print_fig:
                plt.gcf().set_size_inches(3.5, 3.5)
                plt.savefig(
                    self.pub_loc + name + "." + self.format,
                    format=self.format,
                    bbox_inches="tight",
                    transparent=self.transparent,
                )
                plt.close()
            else:
                return ax

        else:
            """Make barchart of cowpi"""
            data.plot(
                kind="bar",
                log=True,
                # yerr=err, capsize=3,
                ylabel=ylabel,
                rot=0,
            )
            plt.gcf().set_size_inches(3.5, 3.5)
            plt.savefig(
                self.pub_loc + name + "_cow_comparison." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            # plot without the extremely low income households
            data = data.iloc[1:4, :]
            # print(cost_comp)
            data.plot(
                kind="bar",
                ylabel=ylabel,
                # yerr=err, capsize=3,
                rot=0,
            )
            plt.gcf().set_size_inches(3.5, 3.5)
            plt.savefig(
                self.pub_loc + name + "_cow_comparison_no_low_in." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

    def linear_regression(self, data, xname=None, yname=None, norm_x=True):
        """
        Wrapper function to perform a linear regression.
        """
        x = data[[xname]]
        y = data[yname]

        if norm_x:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))

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
        results_ind = res_nodes.groupby(["group", "bg"]).mean().loc[:, "min"]

        # read in income distribution data
        income = np.genfromtxt(
            "Input Files/clinton_income_data.csv",
            delimiter=",",
        )

        print(income)
        print(pd.Series(results_ind.values))

        data = pd.concat(
            [pd.Series(income[:, 1]), pd.Series(results_ind.values)],
            axis=1,
            keys=["Income", "Distance"],
        )

        print(data)

        # x = results_ind.values
        # y = income[:, 1]

        model, mod_x = self.linear_regression(data, "Distance", "Income")

        ax.scatter(mod_x, data["Income"])
        ax.plot(mod_x, model.predict(mod_x))

        return ax

    def scatter_plot(self, ind_dist, income, ax):
        x = ind_dist
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        x_with_intercept = np.empty(shape=(len(x_norm), 2), dtype=np.float64)
        x_with_intercept[:, 0] = 1
        x_with_intercept[:, 1] = x_norm

        y = income

        ols_model = sm.OLS(y, x_with_intercept).fit()
        sse = np.sum((ols_model.fittedvalues - y) ** 2)
        ssr = np.sum((ols_model.fittedvalues - y.mean()) ** 2)
        sst = ssr + sse
        print(f"R2 = {ssr/sst}")
        print(len(x_norm))
        print(np.sqrt(sse / (len(x_norm) - 2)))
        print(ols_model.summary())

        lr_model = LinearRegression()
        x = x_norm[:, np.newaxis]
        lr_model.fit(x, y)
        print(lr_model.score(x, y))
        print(lr_model.coef_)
        print(lr_model.intercept_)

        ax.plot(x, lr_model.predict(x))
        ax.scatter(x, y)
        ax.set_xlabel("Normalized Industrial Distance")
        ax.set_ylabel("Household Income")

        return ax

    def bg_map(
        self,
        ax,
        display_demo=None,
        node_data=None,
        wn_nodes=False,
        label_map="",
        label_nodes="",
        lg_fmt="{:.0f}",
        label_bg=False,
        plot_wn=True,
        pipes=False,
        node_cmap=None,
        pipe_cmap=None,
        vmin_inp=None,
        vmax_inp=None,
        legend_bool=False,
    ):
        """
        Create a map showing wn nodes and pipes layered on top of the block
        groups. Block groups are colored based on the display_demo param
        and nodes are colored based on the node_data.

        Parameters:
        -----------
            ax  :   matplotlib.axes
                axes object to plot the data to

            display_demo  :  str
                demographic to be displayed on the block groups
                options: ['median_income', 'perc_w', 'perc_nh']

            node_data  :  pd.DataFrame
                node data to be displayed
        """
        # directory with clinton data
        dir = "Input Files/cities/clinton/"

        # convert the wn to an object with various gdfs
        if wn_nodes:
            wn_gis = wntr.network.to_gis(self.wn)
            wn_gis.junctions = wn_gis.junctions.set_crs("epsg:4326")
            wn_gis.pipes = wn_gis.pipes.set_crs("epsg:4326")
            node_buildings = wn_gis.junctions
        else:
            node_buildings = ci.make_building_list(self.wn, "clinton", dir)
            node_buildings = node_buildings.query('type == "res"')

        # import the block groups for sampson county
        gdf = geopandas.read_file(dir + "sampson_bg_clinton/tl_2023_37_bg.shp")
        gdf["bg"] = gdf["TRACTCE"] + gdf["BLKGRPCE"]
        gdf.set_index("bg", inplace=True)
        gdf.index = gdf.index.astype("int64")

        # import demographic data using pandas
        demo = pd.read_csv(dir + "demographics_bg.csv")
        demo.set_index("bg", inplace=True)
        demo[["perc_w", "perc_nh", "perc_renter"]] = (
            demo[["perc_w", "perc_nh", "perc_renter"]] * 100
        )
        # filter the bgs for clinton
        bg = ["970802", "970600", "970801", "970702", "970701"]
        gdf = gdf[gdf["TRACTCE"].isin(bg)]

        gdf = gdf.join(demo)
        # print(gdf)

        if node_data is not None:
            # apply the bg crs to the wn data
            # wn_gis.junctions.to_crs(gdf.crs, inplace=True)
            node_buildings.to_crs(gdf.crs, inplace=True)

            # add node_data to wn_gis.junctions
            node_data.index = node_data.index.astype("int64")
            node_buildings.index = node_buildings.index.astype("int64")
            # wn_gis.junctions.index = wn_gis.junctions.index.astype('int64')
            # wn_gis.junctions.insert(0, 'data', node_data)
            # print(wn_gis.junctions)
            # print(node_buildings)
            # print(node_data)
            node_buildings["data"] = node_data

            # clip the bg layer to the extent of the wn layer
            # gdf = geopandas.clip(gdf, mask=wn_gis.junctions.total_bounds)
            gdf = geopandas.clip(gdf, mask=node_buildings.total_bounds)
        else:
            # wn_gis = wntr.network.to_gis(self.wn)
            # wn_gis.junctions = wn_gis.junctions.set_crs("epsg:4326")
            # wn_gis.junctions.to_crs(gdf.crs, inplace=True)
            node_buildings.to_crs(gdf.crs, inplace=True)
            gdf = geopandas.clip(gdf, mask=node_buildings.total_bounds)

        if wn_nodes:
            wn_gis.pipes.to_crs(gdf.crs, inplace=True)

        # plot the bg layer
        if display_demo is not None:
            ax = gdf.plot(
                ax=ax,
                column=display_demo,
                cmap="Blues",
                legend=True,
                zorder=1,
                legend_kwds={
                    "label": label_map,
                    "fraction": 0.04,
                    "pad": 0.04,
                    # "fmt": lg_fmt
                },
                vmin=vmin_inp,
                vmax=vmax_inp,
            )
            # if lg_fmt is not None:
            #     cbar = ax.get_figure().get_axes()[-1]
            #     cbar.yaxis.set_major_formatter(
            #         mticker.FuncFormatter(lg_fmt)
            #     )
        else:
            # print(gdf)
            ax = gdf.plot(ax=ax, color="white", edgecolor="black", lw=0.4)

        # add the junctions and pipes or the buildings
        if node_data is not None:
            print(node_buildings.dtypes)
            if legend_bool:
                legend_k = {
                    "label": label_nodes,
                    "fraction": 0.04,
                    "pad": 0.04,
                    # "fmt": lg_fmt
                }
            else:
                legend_k = None

            ax = node_buildings.plot(
                ax=ax,
                marker=".",
                markersize=3,
                # markersize=node_buildings["data"] * 2 if node_cmap is None else 3,
                zorder=3,
                column="data",
                legend=legend_bool,
                cmap=(
                    ListedColormap(["white", "red"]) if node_cmap is None else node_cmap
                ),
                vmin=0,
                vmax=vmax_inp,
                legend_kwds=(
                    {"labels": label_nodes} if node_data.dtype == bool else legend_k
                ),
                # legend_kwds={"labels": label_nodes}
            )
        else:
            if plot_wn:
                ax = node_buildings.plot(
                    ax=ax,
                    marker=".",
                    markersize=3,
                    color="darkorange",
                    zorder=3,
                )

        # add pipes if the wdn nodes are being plotted
        if wn_nodes:
            # print(wn_gis.pipes)
            if pipe_cmap is not None:
                legend_k = {
                    "label": label_nodes,
                    "fraction": 0.04,
                    "pad": 0.04,
                }
            else:
                legend_k = None

            ax = wn_gis.pipes.plot(
                ax=ax,
                color="black" if pipe_cmap is None else None,
                linewidth=wn_gis.pipes["diameter"] * 10 if pipes else 0.3,
                cmap=pipe_cmap,
                column="diameter" if pipe_cmap is not None else None,
                legend=True if pipes else False,
                legend_kwds=legend_k,
                # linewidth=0.5,
                zorder=2,
            )

        # label the block groups
        if label_bg:
            gdf.apply(
                lambda x: ax.annotate(
                    text=x.name, xy=x.geometry.centroid.coords[0], ha="center"
                ),
                axis=1,
            )

        ax.set_axis_off()

        # calculate how the average node data for each bg
        # node_join = node_buildings.sjoin(gdf, how='inner')
        # if node_data is not None:
        # print(node_buildings.loc[:, ["data", "bg"]].groupby(["bg"]).mean())

        return ax


class Graphics(BaseGraphics):
    """
    Main class for making graphics. The boilerplate methods are in
    BaseGraphics.

    parameters:
        publication : bool
            whether to plot pngs or pdfs
        error : str
            the type of error to use
            options include: se, ci95, and sd
    """

    def __init__(
        self,
        publication,
        error,
        days,
        scenario_ls=None,
        inp_file=None,
        skeletonized=False,
        single=False,
        remove_bg=False,
    ):
        self.days = days
        self.x_len = days * 24
        self.skeletonized = skeletonized
        self.transparent = remove_bg
        self.comp_list = [
            "seir_data",
            "demand",
            # "demo",
            "age",
            "flow",
            "cov_ff",
            "cov_pers",
            "wfh",
            "dine",
            "groc",
            "ppe",
        ]

        """ List that will be truncated based on the number of days the
        simulation was run. Does not include the warmup period """
        self.truncate_list = ["seir_data", "demand", "age", "flow"]

        if scenario_ls is not None:
            self.scenarios = scenario_ls
        else:
            self.scenarios = ["base", "basebw", "pm", "pm_nobw"]

        if not single:
            self.init_data()

        """ Set figure parameters """
        plt.rcParams["figure.figsize"] = [3.5, 3.5]
        self.error = error
        if publication:
            self.pub_loc = "Output Files/publication_figures/"
            plt.rcParams["figure.dpi"] = 800
            self.format = "pdf"
        else:
            self.pub_loc = "Output Files/png_figures/"
            self.format = "png"
            plt.rcParams["figure.dpi"] = 500

        self.prim_colors = ["#253494", "#2c7fb8", "#41b6c4", "#a1dab4", "#f1f174"]
        # sec_colors = ['#454545', '#929292', '#D8D8D8']

        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.prim_colors)
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.family"] = ["serif"]
        plt.rcParams["xtick.top"] = True
        plt.rcParams["xtick.bottom"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.left"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.width"] = 0.7
        plt.rcParams["ytick.major.width"] = 0.7
        plt.rcParams["xtick.major.size"] = 3.0
        plt.rcParams["ytick.major.size"] = 3.0

        """ Import water network and data """
        if not inp_file:
            inp_file = "Input Files/micropolis/MICROPOLIS_v1_inc_rest_consumers.inp"
            """ Set the various node lists """
            self.get_nodes(self.wn)
            """ Get industrial distances for each residential node in the network """
            self.ind_distances, ind_closest = self.calc_industry_distance(self.wn)

        self.wn = wntr.network.WaterNetworkModel(inp_file)

        # self.dist_values = [v for k, v in ind_distances.items() if k in self.res_nodes]

    def init_data(self):
        """Define data directories"""
        # self.base_comp_dir = 'Output Files/30_no_pm/'
        # self.pm_comp_dir = 'Output Files/30_all_pm/'
        self.base_comp_dir = "Output Files/30_base/"
        self.base_bw_comp_dir = "Output Files/30_basebw/"
        self.pm_comp_dir = "Output Files/30_pm/"
        self.pm_nobw_comp_dir = "Output Files/30_pm_nobw/"
        self.base_bw_neg20twa_comp_dir = "Output Files/30_basebw_-20twa/"
        self.base_bw_20twa_comp_dir = "Output Files/30_basebw_20twa/"
        self.pm_neg20twa_comp_dir = "Output Files/30_pm_-20twa/"
        self.pm_20twa_comp_dir = "Output Files/30_pm_20twa/"
        # self.pm_comp_perc_dir = 'Output Files/1_Distance Based Income/30_pmbw_di_perc/'
        # self.pm_25ind_comp_dir = 'Output Files/3_Sensitivity Analysis/30_all_pm_25ind_equity/'
        # self.pm_50ind_comp_dir = 'Output Files/3_Sensitivity Analysis/30_all_pm_50ind_equity/'
        # self.pm_75ind_comp_dir = 'Output Files/3_Sensitivity Analysis/30_all_pm_75ind_equity/'
        # self.pm_100ind_comp_dir = 'Output Files/3_Sensitivity Analysis/30_all_pm_100ind_equity/'
        # self.pm_comp_noD_dir = 'Output Files/1_Distance Based Income/30_pmbw_di_noD/'
        # self.pm_comp_noC_dir = 'Output Files/1_Distance Based Income/30_pmbw_di_noC/'
        # self.pm_comp_noH_dir = 'Output Files/1_Distance Based Income/30_pmbw_di_noH/'
        # self.pm_nodi_comp_dir = 'Output Files/2_Non-distance Based Income/30_pmbw/'

        """ Read in data from data directories """
        if "base" in self.scenarios:
            self.base = ut.read_comp_data(
                self.base_comp_dir, self.comp_list, self.days, self.truncate_list
            )
        if "basebw" in self.scenarios:
            self.basebw = ut.read_comp_data(
                self.base_bw_comp_dir, self.comp_list, self.days, self.truncate_list
            )
        if "pm" in self.scenarios:
            self.pm = ut.read_comp_data(
                self.pm_comp_dir, self.comp_list, self.days, self.truncate_list
            )
        if "pm_nobw" in self.scenarios:
            self.pm_nobw = ut.read_comp_data(
                self.pm_nobw_comp_dir, self.comp_list, self.days, self.truncate_list
            )
        if "sa" in self.scenarios:
            self.pm_neg20 = ut.read_comp_data(
                self.pm_neg20twa_comp_dir, self.comp_list, self.days, self.truncate_list
            )
            self.pm_20 = ut.read_comp_data(
                self.pm_20twa_comp_dir, self.comp_list, self.days, self.truncate_list
            )
            self.basebw_neg20 = ut.read_comp_data(
                self.base_bw_neg20twa_comp_dir,
                self.comp_list,
                self.days,
                self.truncate_list,
            )
            self.basebw_20 = ut.read_comp_data(
                self.base_bw_20twa_comp_dir,
                self.comp_list,
                self.days,
                self.truncate_list,
            )

        # self.pm_noD = ut.read_comp_data(
        #     self.pm_comp_noD_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm_noC = ut.read_comp_data(
        #     self.pm_comp_noC_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm_noH = ut.read_comp_data(
        #     self.pm_comp_noH_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm_nodi = ut.read_comp_data(
        #     self.pm_nodi_comp_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm25ind = ut.read_comp_data(
        #     self.pm_25ind_comp_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm50ind = ut.read_comp_data(
        #     self.pm_50ind_comp_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm75ind = ut.read_comp_data(
        #     self.pm_75ind_comp_dir, self.comp_list, self.days, self.truncate_list
        # )
        # self.pm100ind = ut.read_comp_data(
        #     self.pm_100ind_comp_dir, self.comp_list, self.days, self.truncate_list
        # )

        """ Read and distill household level data """
        self.post_household()

        # day200_loc = 'Output Files/2022-12-12_14-33_ppe_200Days_results/'
        # day400_loc = 'Output Files/2022-12-14_10-08_no_PM_400Days_results/'
        # days_200 = read_data(day200_loc, ['seir', 'demand', 'age'])
        # days_400 = read_data(day400_loc, ['seir', 'demand', 'age'])

        """ set x values """
        self.x_values_hour = np.array(
            [
                x
                for x in np.arange(
                    0, self.days, self.days / len(self.base["avg_demand"])
                )
            ]
        )
        self.x_values_day = np.array([x for x in range(self.days)])

        """ Get times list: first time is max wfh, 75% wfh, 50% wfh, 25% wfh """
        # self.get_times(self.pm)

    def flow_plots(self):
        """Make the flow direction changes plot"""
        pm_flow_change, pm_flow_sum = self.calc_flow_diff(
            self.pm["avg_flow"], self.times[len(self.times) - 1]
        )
        base_flow_change, base_flow_sum = self.calc_flow_diff(
            self.base["avg_flow"], self.times[len(self.times) - 1]
        )

        # print(pm_flow_sum['MA728'])
        # print(base_flow_sum['MA728'])

        ax = wntr.graphics.plot_network(
            self.wn,
            link_attribute=pm_flow_sum,
            link_colorbar_label="Flow Changes",
            node_size=0,
            link_width=2,
        )
        plt.savefig(
            self.pub_loc + "flow_network_pm." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        ax = wntr.graphics.plot_network(
            self.wn,
            link_attribute=base_flow_sum,
            link_colorbar_label="Flow Changes",
            node_size=0,
            link_width=2,
        )
        plt.savefig(
            self.pub_loc + "flow_network_base." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def demand_plots(self, sa=False):
        """Make demand plots by sector with PM data"""
        # define the columns of the input data and the x_values
        cols = ["Residential", "Commercial", "Industrial"]

        if sa:
            # sort data for 25ind case
            sector_dem50 = self.calc_sec_averages(self.pm25ind["avg_demand"], op="sum")
            sector_dem_var50 = self.calc_sec_averages(
                self.pm25ind["var_demand"], op="sum"
            )
            sector_dem_err50 = ut.calc_error(sector_dem_var50, self.error)

            # sort data for 50ind case
            sector_dem50 = self.calc_sec_averages(self.pm50ind["avg_demand"], op="sum")
            sector_dem_var50 = self.calc_sec_averages(
                self.pm50ind["var_demand"], op="sum"
            )
            sector_dem_err50 = ut.calc_error(sector_dem_var50, self.error)

            # sort data for 75ind case
            sector_dem75 = self.calc_sec_averages(self.pm75ind["avg_demand"], op="sum")
            sector_dem_var75 = self.calc_sec_averages(
                self.pm75ind["var_demand"], op="sum"
            )
            sector_dem_err75 = ut.calc_error(sector_dem_var75, self.error)

            # sort data for 100ind case
            sector_dem100 = self.calc_sec_averages(
                self.pm100ind["avg_demand"], op="sum"
            )
            sector_dem_var100 = self.calc_sec_averages(
                self.pm100ind["var_demand"], op="sum"
            )
            sector_dem_err100 = ut.calc_error(sector_dem_var100, self.error)

            # plot demand by sector for 50ind
            ax = plt.subplot()
            self.make_avg_plot(
                ax, sector_dem50, sector_dem_err50, cols, self.x_values_hour
            )
            # ax1 = ax.twinx()
            # ax1.plot(self.x_values_day, self.pm['avg_wfh'].mean(axis=1) * 100, color='k')
            ax.legend(["Residential", "Commercial", "Industrial"])
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Demand (L)")
            # ax1.set_ylabel('Percent of Population WFH')
            plt.savefig(
                self.pub_loc + "sector_demand_50ind" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

        if not self.skeletonized:
            # sort data
            sector_dem = self.calc_sec_averages(self.pm["avg_demand"], op="sum")
            sector_dem_var = self.calc_sec_averages(self.pm["var_demand"], op="sum")
            sector_dem_err = ut.calc_error(sector_dem_var, self.error)

            # plot demand by sector
            ax = plt.subplot()
            self.make_avg_plot(ax, sector_dem, sector_dem_err, cols, self.x_values_hour)
            # ax1 = ax.twinx()
            # ax1.plot(self.x_values_day, self.pm['avg_wfh'].mean(axis=1) * 100, color='k')
            ax.legend(["Residential", "Commercial", "Industrial"])
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Demand (L)")
            # ax1.set_ylabel('Percent of Population WFH')
            plt.savefig(
                self.pub_loc + "sector_demand" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            """ Make plots of aggregate demand data """
            demand_base = self.base["avg_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            demand_basebw = self.basebw["avg_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            demand_pm = self.pm["avg_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            demand_pm_nobw = self.pm_nobw["avg_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            demand = pd.concat(
                [
                    demand_base.sum(axis=1).rolling(24).mean(),
                    demand_basebw.sum(axis=1).rolling(24).mean(),
                    demand_pm_nobw.sum(axis=1).rolling(24).mean(),
                    demand_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            var_base = self.base["var_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            var_basebw = self.basebw["var_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            var_pm = self.pm["var_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            var_pm_nobw = self.pm_nobw["var_demand"][
                self.res_nodes + self.com_nodes + self.ind_nodes
            ]
            demand_var = pd.concat(
                [
                    var_base.sum(axis=1).rolling(24).mean(),
                    var_basebw.sum(axis=1).rolling(24).mean(),
                    var_pm_nobw.sum(axis=1).rolling(24).mean(),
                    var_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "Base+BW", "PM", "PM+BW"],
            )

            demand_err = ut.calc_error(demand_var, self.error)

            fig, axes = plt.subplots(1, 2)
            # format the y axis ticks to have a dollar sign and thousands commas
            fmt = "{x:,.0f}"
            tick = mtick.StrMethodFormatter(fmt)
            axes[0].yaxis.set_major_formatter(tick)
            axes[1].yaxis.set_major_formatter(tick)

            axes[0] = self.make_avg_plot(
                axes[0],
                demand,
                demand_err,
                ["Base", "Base+BW", "PM", "PM+BW"],
                self.x_values_hour,
                show_labels=False,
            )

            """ Plot residential demand in a subfigure """
            demand_base = self.base["avg_demand"][self.res_nodes]
            demand_basebw = self.basebw["avg_demand"][self.res_nodes]
            demand_pm = self.pm["avg_demand"][self.res_nodes]
            demand_pm_nobw = self.pm_nobw["avg_demand"][self.res_nodes]
            demand = pd.concat(
                [
                    demand_base.sum(axis=1).rolling(24).mean(),
                    demand_basebw.sum(axis=1).rolling(24).mean(),
                    demand_pm_nobw.sum(axis=1).rolling(24).mean(),
                    demand_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "Base+BW", "PM", "PM+BW"],
            )

            var_base = self.base["var_demand"][self.res_nodes]
            var_basebw = self.basebw["var_demand"][self.res_nodes]
            var_pm = self.pm["var_demand"][self.res_nodes]
            var_pm_nobw = self.pm_nobw["var_demand"][self.res_nodes]
            demand_var = pd.concat(
                [
                    var_base.sum(axis=1).rolling(24).mean(),
                    var_basebw.sum(axis=1).rolling(24).mean(),
                    var_pm_nobw.sum(axis=1).rolling(24).mean(),
                    var_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "Base+BW", "PM", "PM+BW"],
            )

            demand_err = ut.calc_error(demand_var, self.error)

            axes[1] = self.make_avg_plot(
                axes[1],
                demand,
                demand_err,
                ["Base", "Base+BW", "PM", "PM+BW"],
                self.x_values_hour,
            )

            axes[0].legend(["Base", "Base+BW", "PM", "PM+BW"])
            axes[0].text(
                0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )
            fig.supxlabel("Time (days)", y=-0.06)
            fig.supylabel("Demand (L)", x=0.04)
            plt.gcf().set_size_inches(7, 3.5)

            plt.savefig(
                self.pub_loc + "sum_demand_aggregate" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()
        else:  # if skeletonized
            nodes_w_demand = [
                name
                for name, node in self.wn.junctions()
                if node.demand_timeseries_list[0].base_value > 0
            ]

            print(
                (self.base["avg_demand"][nodes_w_demand].sum(axis=1) * 3600).sum()
                / 3.875
                / self.days
            )

            """ Make plots of aggregate demand data """
            demand_base = self.base["avg_demand"][nodes_w_demand]
            demand_basebw = self.basebw["avg_demand"][nodes_w_demand]
            demand_pm = self.pm["avg_demand"][nodes_w_demand]
            demand_pm_nobw = self.pm_nobw["avg_demand"][nodes_w_demand]
            demand = pd.concat(
                [
                    demand_base.sum(axis=1).rolling(24).mean(),
                    demand_basebw.sum(axis=1).rolling(24).mean(),
                    demand_pm_nobw.sum(axis=1).rolling(24).mean(),
                    demand_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            var_base = self.base["var_demand"][nodes_w_demand]
            var_basebw = self.basebw["var_demand"][nodes_w_demand]
            var_pm = self.pm["var_demand"][nodes_w_demand]
            var_pm_nobw = self.pm_nobw["var_demand"][nodes_w_demand]
            demand_var = pd.concat(
                [
                    var_base.sum(axis=1).rolling(24).mean(),
                    var_basebw.sum(axis=1).rolling(24).mean(),
                    var_pm_nobw.sum(axis=1).rolling(24).mean(),
                    var_pm.sum(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            demand_err = ut.calc_error(demand_var, self.error)

            ax = plt.subplot()
            # format the y axis ticks to have a dollar sign and thousands commas
            # fmt = "{x:,.0f}"
            # tick = mtick.StrMethodFormatter(fmt)
            # ax.yaxis.set_major_formatter(tick)

            ax = self.make_avg_plot(
                ax,
                demand,
                demand_err,
                ["Base", "TWA", "PM", "TWA+PM"],
                self.x_values_hour,
                xlabel="Time (days)",
                ylabel="Demand (L/s)",
                show_labels=True,
            )
            plt.savefig(
                self.pub_loc + "sum_demand_aggregate" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            """ Make plots of aggregate demand data """
            node_buildings = pd.read_pickle("buildings.pkl")
            print(node_buildings)
            print(node_buildings.groupby("type").size())
            counts = node_buildings.value_counts(["wdn_node", "type"]).unstack()
            perc_counts = counts.divide(counts.sum(axis=1) / 100, axis=0)

            perc_counts["type"] = perc_counts.apply(lambda x: x.idxmax(), axis=1)
            # res_perc_nodes = perc_counts[(perc_counts["mfh"] + perc_counts["res"]) > res_t].index
            # print(self.base["avg_demand"].loc[:, res_perc_nodes])
            print(perc_counts)

            # fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
            # top_plots = ["res", "mfh"]
            # mid_plots = ["ind", "com"]
            # bot_plots = ["caf", "gro"]
            # for i, type in enumerate(top_plots):
            #     axes[0, i] = self.plot_demand_by_node(
            #         axes[0, i], perc_counts[perc_counts["type"] == type].index
            #     )
            # for i, type in enumerate(mid_plots):
            #     axes[1, i] = self.plot_demand_by_node(
            #         axes[1, i], perc_counts[perc_counts["type"] == type].index
            #     )
            # for i, type in enumerate(bot_plots):
            #     axes[2, i] = self.plot_demand_by_node(
            #         axes[2, i], perc_counts[perc_counts["type"] == type].index
            #     )

            # plt.gcf().set_size_inches(3.5, 7)
            # plt.savefig(
            #     self.pub_loc + "sum_demand" + "." + self.format,
            #     format=self.format,
            #     bbox_inches="tight",
            # )
            # plt.close()

            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            axes[0, 0] = self.plot_demand_by_case(
                axes[0, 0], self.base["avg_demand"], perc_counts, legend_bool=True
            )
            axes[0, 1] = self.plot_demand_by_case(
                axes[0, 1], self.basebw["avg_demand"], perc_counts
            )
            axes[1, 0] = self.plot_demand_by_case(
                axes[1, 0], self.pm_nobw["avg_demand"], perc_counts
            )
            axes[1, 1] = self.plot_demand_by_case(
                axes[1, 1], self.pm["avg_demand"], perc_counts
            )
            axes[0, 0].text(
                0.5, -0.1, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            )
            axes[0, 1].text(
                0.5, -0.1, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            )
            axes[1, 0].text(
                0.5, -0.2, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            )
            axes[1, 1].text(
                0.5, -0.2, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            )
            fig.supxlabel("Time (days)", y=0)
            fig.supylabel("Demand (L/s)", x=0.04)
            plt.gcf().set_size_inches(6, 6)

            plt.savefig(
                self.pub_loc + "sum_demand_stacked" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            # res_data = self.base["demand"]["tw_demand"].groupby("i").sum()
            # for i in range(int(len(data.columns) / 30)):
            #     res_data.insert(30 * i, str(i), 0)
            # print(res_data)
            # data = res_data.iloc[:, ::-1].apply(
            #     lambda x: x - x.shift(-1, axis=0), axis=1
            # )
            # for i in range(int(len(res_data.columns) / 30)):
            #     res_data = res_data.drop(str(i), axis=1)
            # print(res_data.iloc[:, ::-1].mean(axis=0))

            # nonres_data = self.base["avg_demand"] - res_data

            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            axes[0, 0] = self.plot_demand_res_nonres(
                axes[0, 0], self.base, nodes_w_demand, legend_bool=True
            )
            axes[0, 1] = self.plot_demand_res_nonres(
                axes[0, 1], self.basebw, nodes_w_demand
            )
            axes[1, 0] = self.plot_demand_res_nonres(
                axes[1, 0], self.pm_nobw, nodes_w_demand
            )
            axes[1, 1] = self.plot_demand_res_nonres(
                axes[1, 1], self.pm, nodes_w_demand
            )
            axes[0, 0].text(
                0.5, -0.1, "Base", size=12, ha="center", transform=axes[0, 0].transAxes
            )
            axes[0, 1].text(
                0.5, -0.1, "TWA", size=12, ha="center", transform=axes[0, 1].transAxes
            )
            axes[1, 0].text(
                0.5, -0.2, "PM", size=12, ha="center", transform=axes[1, 0].transAxes
            )
            axes[1, 1].text(
                0.5,
                -0.2,
                "TWA+PM",
                size=12,
                ha="center",
                transform=axes[1, 1].transAxes,
            )
            fig.supxlabel("Time (days)", y=0)
            fig.supylabel("Demand (ML/day)", x=0.04)
            plt.gcf().set_size_inches(6, 6)

            plt.savefig(
                self.pub_loc + "sum_demand_res_nonres" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            ax = plt.subplot()

            ax = self.plot_demand_res_nonres(ax, self.pm, nodes_w_demand)

            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Demand (ML/day)")
            plt.savefig(
                self.pub_loc + "sum_demand_pm_" + "." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()
            # ax = plt.subplot()
            # ind_perc_nodes = perc_counts[perc_counts["ind"] > res_t].index
            # ax = self.plot_demand_by_node(ax, ind_perc_nodes)

            # plt.savefig(
            #     self.pub_loc + "sum_demand_ind" + "." + self.format,
            #     format=self.format,
            #     bbox_inches="tight",
            # )
            # plt.close()

            # ax = plt.subplot()
            # com_perc_nodes = perc_counts[perc_counts["com"] > res_t].index
            # ax = self.plot_demand_by_node(ax, com_perc_nodes)

            # plt.savefig(
            #     self.pub_loc + "sum_demand_com" + "." + self.format,
            #     format=self.format,
            #     bbox_inches="tight",
            # )
            # plt.close()

            # print(len(res_perc_nodes))
            # print(len(ind_perc_nodes))
            # print(len(com_perc_nodes))
            print(perc_counts.groupby("type").size())

    def age_plots(
        self, data=None, name="", thres_n=130, sa=False, map=False, threshold=False
    ):
        if not self.skeletonized:
            """Make age plot by sector for both base and PM"""
            cols = ["Residential", "Commercial", "Industrial"]

            age_pm = self.calc_sec_averages(self.pm["avg_age"])
            age_sd_pm = self.calc_sec_averages(self.pm["var_age"])
            age_pm_err = ut.calc_error(age_sd_pm, self.error)

            age_pm_nobw = self.calc_sec_averages(self.pm_nobw["avg_age"])
            age_sd_pm_nobw = self.calc_sec_averages(self.pm_nobw["var_age"])
            age_pm_nobw_err = ut.calc_error(age_sd_pm_nobw, self.error)

            age_base = self.calc_sec_averages(self.base["avg_age"])
            age_sd_base = self.calc_sec_averages(self.base["var_age"])
            age_base_err = ut.calc_error(age_sd_base, self.error)

            age_basebw = self.calc_sec_averages(self.basebw["avg_age"])
            age_sd_basebw = self.calc_sec_averages(self.basebw["var_age"])
            age_basebw_err = ut.calc_error(age_sd_basebw, self.error)

            age_pm50 = self.calc_sec_averages(self.pm50ind["avg_age"])
            age_sd_pm50 = self.calc_sec_averages(self.pm50ind["var_age"])
            age_pm50_err = ut.calc_error(age_sd_pm50, self.error)

            age_pm75 = self.calc_sec_averages(self.pm75ind["avg_age"])
            age_sd_pm75 = self.calc_sec_averages(self.pm75ind["var_age"])
            age_pm75_err = ut.calc_error(age_sd_pm75, self.error)

            fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
            axes[0, 0] = self.make_avg_plot(
                axes[0, 0],
                age_base / 3600,
                age_base_err / 3600,
                cols,
                self.x_values_hour,
            )
            axes[0, 1] = self.make_avg_plot(
                axes[0, 1],
                age_basebw / 3600,
                age_basebw_err / 3600,
                cols,
                self.x_values_hour,
            )
            axes[1, 0] = self.make_avg_plot(
                axes[1, 0],
                age_pm_nobw / 3600,
                age_pm_nobw_err / 3600,
                cols,
                self.x_values_hour,
            )
            axes[1, 1] = self.make_avg_plot(
                axes[1, 1], age_pm / 3600, age_pm_err / 3600, cols, self.x_values_hour
            )

            axes[0, 0].legend(cols)
            axes[0, 0].text(
                0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            )
            axes[0, 1].text(
                0.5, -0.14, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            )
            axes[1, 0].text(
                0.5, -0.23, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            )
            axes[1, 1].text(
                0.5, -0.23, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            )
            fig.supxlabel("Time (days)", y=0)
            fig.supylabel("Age (hrs)", x=0.04)
            plt.gcf().set_size_inches(4, 4)

            plt.savefig(
                self.pub_loc + "mean_age_sector." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()
        else:
            if data is None:
                age_base = self.base["avg_age"]
                age_basebw = self.basebw["avg_age"]
                age_pm = self.pm["avg_age"]
                age_pm_nobw = self.pm_nobw["avg_age"]
                var_base = self.base["var_age"]
                var_basebw = self.basebw["var_age"]
                var_pm = self.pm["var_age"]
                var_pm_nobw = self.pm_nobw["var_age"]
            else:
                age_base = data[0]["avg_age"]
                age_basebw = data[1]["avg_age"]
                age_pm = data[2]["avg_age"]
                age_pm_nobw = data[3]["avg_age"]
                var_base = data[0]["var_age"]
                var_basebw = data[1]["var_age"]
                var_pm = data[2]["var_age"]
                var_pm_nobw = data[3]["var_age"]

            age = pd.concat(
                [
                    age_base.mean(axis=1).rolling(24).mean(),
                    age_basebw.mean(axis=1).rolling(24).mean(),
                    age_pm_nobw.mean(axis=1).rolling(24).mean(),
                    age_pm.mean(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            age_var = pd.concat(
                [
                    var_base.mean(axis=1).rolling(24).mean(),
                    var_basebw.mean(axis=1).rolling(24).mean(),
                    var_pm_nobw.mean(axis=1).rolling(24).mean(),
                    var_pm.mean(axis=1).rolling(24).mean(),
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            age_err = ut.calc_error(age_var, self.error)

            ax = plt.subplot()
            # format the y axis ticks to have a dollar sign and thousands commas
            # fmt = '{x:,.0f}'
            # tick = mtick.StrMethodFormatter(fmt)
            # ax.yaxis.set_major_formatter(tick)

            ax = self.make_avg_plot(
                ax,
                age / 3600,
                age_err / 3600,
                ["Base", "TWA", "PM", "TWA+PM"],
                self.x_values_hour,
                xlabel="Time (days)",
                ylabel="Water Age (hours)",
                show_labels=True,
            )
            plt.savefig(
                self.pub_loc + name + "mean_age_aggregate." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

        if sa:
            """Make plot of industrial demand SA"""
            fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
            axes[0] = self.make_avg_plot(
                axes[0], age_pm / 3600, age_pm_err / 3600, cols, self.x_values_hour
            )
            axes[1] = self.make_avg_plot(
                axes[1], age_pm50 / 3600, age_pm50_err / 3600, cols, self.x_values_hour
            )
            axes[2] = self.make_avg_plot(
                axes[2], age_pm75 / 3600, age_pm75_err / 3600, cols, self.x_values_hour
            )

            axes[0].legend(cols)
            axes[0].text(
                0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )
            axes[2].text(
                0.5, -0.14, "(c)", size=12, ha="center", transform=axes[2].transAxes
            )
            fig.supxlabel("Time (days)", y=-0.06)
            fig.supylabel("Age (hrs)", x=0.04)
            plt.gcf().set_size_inches(8, 3.5)

            plt.savefig(
                self.pub_loc + "mean_age_sector_indSA." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

        if map:
            """Make age plot comparing base and PM"""
            nodes_w_demand = [
                name
                for name, node in self.wn.junctions()
                if node.demand_timeseries_list[0].base_value > 0
            ]
            # make_sector_plot(self.wn, no_wfh['avg_age'] / 3600, 'Age (hr)', 'mean',
            #                  'mean_age_aggregate_' + error, wfh['avg_age'] / 3600,
            #                  sd=ut.calc_error(no_wfh['var_age'], error)/3600,
            #                  sd2=ut.calc_error(wfh['var_age'], error)/3600, type='all')

            if data is None:
                base_age = self.calc_age_diff(
                    self.base["avg_age"], nodes_w_demand, thres_n
                )
                basebw_age = self.calc_age_diff(
                    self.basebw["avg_age"], nodes_w_demand, thres_n
                )
                pm_nobw_age = self.calc_age_diff(
                    self.pm_nobw["avg_age"], nodes_w_demand, thres_n
                )
                pm_age = self.calc_age_diff(self.pm["avg_age"], nodes_w_demand, thres_n)
            else:
                base_age = self.calc_age_diff(
                    data[0]["avg_age"], nodes_w_demand, thres_n
                )
                basebw_age = self.calc_age_diff(
                    data[1]["avg_age"], nodes_w_demand, thres_n
                )
                pm_nobw_age = self.calc_age_diff(
                    data[3]["avg_age"], nodes_w_demand, thres_n
                )
                pm_age = self.calc_age_diff(data[2]["avg_age"], nodes_w_demand, thres_n)
            print(pm_age)
            # basebw_age = self.calc_age_diff(self.basebw["avg_age"])

            """ Plot intersectional map with water age and block group """
            # diff_age = dict()
            # for k, v in base_age.items():
            #     diff_age[k] = pm_age[k] > v

            fig, axes = plt.subplots(2, 2)

            axes[0, 0] = self.bg_map(
                axes[0, 0],
                # "median_income",
                node_data=pd.Series(base_age),
                wn_nodes=True,
                # label="Median Income",
                # lg_fmt=custom_format,
                node_cmap=ListedColormap(["blue", "red"]),
            )
            axes[0, 1] = self.bg_map(
                axes[0, 1],
                # "perc_renter",
                node_data=pd.Series(basebw_age),
                wn_nodes=True,
                # label="% Renter",
                # vmin_inp=0,
                node_cmap=ListedColormap(["blue", "red"]),
            )
            axes[1, 0] = self.bg_map(
                axes[1, 0],
                # "perc_w",
                node_data=pd.Series(pm_nobw_age),
                wn_nodes=True,
                # label="% White",
                # vmin_inp=0,
                node_cmap=ListedColormap(["blue", "red"]),
            )
            axes[1, 1] = self.bg_map(
                axes[1, 1],
                # "perc_nh",
                node_data=pd.Series(pm_age),
                wn_nodes=True,
                # label="% non-Hispanic",
                # vmin_inp=0,
                node_cmap=ListedColormap(["blue", "red"]),
            )

            axes[0, 0].text(
                0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            )
            axes[0, 1].text(
                0.5, -0.14, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            )
            axes[1, 0].text(
                0.5, -0.14, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            )
            axes[1, 1].text(
                0.5, -0.14, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            )
            # fig.supxlabel("Time (days)", y=0)
            # fig.supylabel("Age (hrs)", x=0.04)
            orange_dot = mlines.Line2D(
                [0], [0], color="red", linewidth=0, marker=".", markersize=10
            )
            blue_dot = mlines.Line2D(
                [0], [0], color="blue", linewidth=0, marker=".", markersize=10
            )
            plt.gcf().set_size_inches(7, 6)
            fig.legend(
                handles=[blue_dot, orange_dot],
                labels=["<" + str(thres_n) + " hours", ">" + str(thres_n) + " hours"],
                loc="outside lower center",
            )

            plt.savefig(
                self.pub_loc + name + "intersection_age." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

        if threshold:
            # first30_age = age_pm.head(720).mean() / 3600
            last30_age = pd.concat(
                [
                    age_base.tail(720).mean() / 3600,
                    age_basebw.tail(720).mean() / 3600,
                    age_pm_nobw.tail(720).mean() / 3600,
                    age_pm.tail(720).mean() / 3600,
                ],
                axis=1,
                keys=["Base", "TWA", "PM", "TWA+PM"],
            )

            ax = plt.subplot()
            ax.boxplot(last30_age, showmeans=True)

            # add a line at 150 hrs
            ax.axhline(y=150, color="r", linestyle="dashed", lw=1)

            ax.set_xticklabels(["Base", "TWA", "PM", "TWA+PM"])

            ax.set_ylabel("Water Age (hours)")

            plt.savefig(
                self.pub_loc + name + "water_age_boxplot." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

    def ind_dist_plots(self):
        """Calculate the distance to the closest industrial node"""
        ind_distances, ind_closest = self.calc_industry_distance(self.wn)

        """ Make lists of the age values and age error values to plot """
        pm_age_values = list()
        base_age_values = list()
        pm_age_sd = list()
        base_age_sd = list()
        pm_curr_age_values = self.pm["avg_age"].iloc[len(self.pm["avg_age"]) - 1] / 3600
        base_curr_age_values = (
            self.base["avg_age"].iloc[len(self.base["avg_age"]) - 1] / 3600
        )
        pm_curr_age_sd = self.pm["var_age"].iloc[len(self.pm["var_age"]) - 1]
        base_curr_age_sd = self.base["var_age"].iloc[len(self.base["var_age"]) - 1]
        # print(pm_curr_age_values)
        """ Collect the age for each residential node """
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
        self.make_distance_plot(
            dist_values,
            base_age_values,
            pm_age_values,
            base_age_sd,
            pm_age_sd,
            "Distance (m)",
            "Age (hr)",
            "pm_age_ind_distance",
            ["Base", "PM"],
        )

    def sv_heatmap_plots(self):
        """Make agent state variable plots"""
        base_sv = ut.read_data(
            "Output Files/30_all_pm/2023-05-30_15-29_0_results/",
            ["cov_pers", "cov_ff", "media"],
        )
        no_pm_sv = ut.read_data(
            "Output Files/30_no_pm/2023-05-26_08-33_0_results/",
            ["cov_pers", "cov_ff", "media"],
        )

        agent = "124"
        cols = ["Personal", "Friends-Family", "Media"]
        data = pd.concat(
            [
                base_sv["cov_pers"][agent],
                base_sv["cov_ff"][agent],
                base_sv["media"][agent],
            ],
            axis=1,
            keys=cols,
        )
        plt.plot(self.x_values_day, data)
        plt.xlabel("Time (day)")
        plt.ylabel("Value")

        plt.savefig(
            self.pub_loc + "state_variable_plot." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        self.make_heatmap(
            base_sv["cov_ff"].T, "Time (day)", "Agent", "ff_heatmap_all_pm", 6
        )
        self.make_heatmap(
            no_pm_sv["cov_ff"].T, "Time (day)", "Agent", "ff_heatmap_no_pm", 6
        )

    def sv_comp_plots(self):
        """State variable scenario comparisons"""
        ff = pd.concat(
            [self.base["avg_cov_ff"].mean(axis=1), self.pm["avg_cov_ff"].mean(axis=1)],
            axis=1,
            keys=["Base", "PM"],
        )
        ff_var = pd.concat(
            [self.base["var_cov_ff"].mean(axis=1), self.pm["var_cov_ff"].mean(axis=1)],
            axis=1,
            keys=["Base", "PM"],
        )
        ff_err = ut.calc_error(ff_var, self.error)

        # ax = plt.subplot()
        # ax = self.make_avg_plot(ax, data, err, ['Base', 'PM'],
        #                         np.delete(self.x_values, 0),
        #                         'Time (day)', 'Average Value',
        #                         show_labels=True)
        # plt.savefig(self.pub_loc + 'ff_avg' + '.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        pers = pd.concat(
            [
                self.base["avg_cov_pers"].mean(axis=1),
                self.pm["avg_cov_pers"].mean(axis=1),
            ],
            axis=1,
            keys=["Base", "PM"],
        )
        pers_var = pd.concat(
            [
                self.base["var_cov_pers"].mean(axis=1),
                self.pm["var_cov_pers"].mean(axis=1),
            ],
            axis=1,
            keys=["Base", "PM"],
        )
        pers_err = ut.calc_error(pers_var, self.error)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False)
        axes[0] = self.make_avg_plot(
            axes[0], pers, pers_err, ["Base", "PM"], np.delete(self.x_values_day, 0)
        )
        axes[1] = self.make_avg_plot(
            axes[1], ff, ff_err, ["Base", "PM"], np.delete(self.x_values_day, 0)
        )

        axes[0].legend(["Base", "PM"])
        axes[0].text(
            0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )
        fig.supxlabel("Time (days)", y=-0.03)
        fig.supylabel("Average Values", x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(
            self.pub_loc + "sv_comparison." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
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
        """BBN decisions scenario comparisons"""
        cols = ["Dine out less", "Grocery shop less", "WFH", "Wear PPE"]
        data = pd.concat(
            [
                self.pm["avg_dine"].mean(axis=1),
                self.pm["avg_groc"].mean(axis=1),
                self.pm["avg_wfh"].mean(axis=1),
                self.pm["avg_ppe"].mean(axis=1),
            ],
            axis=1,
            keys=cols,
        )
        var = pd.concat(
            [
                self.pm["var_dine"].mean(axis=1),
                self.pm["var_groc"].mean(axis=1),
                self.pm["var_wfh"].mean(axis=1),
                self.pm["var_ppe"].mean(axis=1),
            ],
            axis=1,
            keys=cols,
        )
        err = ut.calc_error(var, self.error)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True
        )
        ax1.plot(np.delete(self.x_values_day, 0), data[cols[0]])
        ax1.fill_between(
            np.delete(self.x_values_day, 0),
            data[cols[0]] - err[cols[0]],
            data[cols[0]] + err[cols[0]],
            alpha=0.5,
        )
        ax2.plot(np.delete(self.x_values_day, 0), data[cols[1]])
        ax2.fill_between(
            np.delete(self.x_values_day, 0),
            data[cols[1]] - err[cols[1]],
            data[cols[1]] + err[cols[1]],
            alpha=0.5,
        )
        ax3.plot(np.delete(self.x_values_day, 0), data[cols[2]])
        ax3.fill_between(
            np.delete(self.x_values_day, 0),
            data[cols[2]] - err[cols[2]],
            data[cols[2]] + err[cols[2]],
            alpha=0.5,
        )
        ax4.plot(np.delete(self.x_values_day, 0), data[cols[3]])
        ax4.fill_between(
            np.delete(self.x_values_day, 0),
            data[cols[3]] - err[cols[3]],
            data[cols[3]] + err[cols[3]],
            alpha=0.5,
        )
        ax1.text(0.5, -0.14, "(a)", size=12, ha="center", transform=ax1.transAxes)
        ax2.text(0.5, -0.14, "(b)", size=12, ha="center", transform=ax2.transAxes)
        ax3.text(0.5, -0.24, "(c)", size=12, ha="center", transform=ax3.transAxes)
        ax4.text(0.5, -0.24, "(d)", size=12, ha="center", transform=ax4.transAxes)
        fig.supxlabel("Time (days)", y=-0.02)
        fig.supylabel("Percent Adoption", x=-0.03)

        plt.savefig(
            self.pub_loc + "bbn_decision_all_pm." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def make_equity_plots(self):
        metrics = pd.concat(
            [self.base["avg_burden"].iloc[:, 0], self.pm["avg_burden"].iloc[:, 0]],
            axis=1,
            keys=["Base", "PM"],
        )
        metrics_var = pd.concat(
            [self.base["var_burden"].iloc[:, 0], self.pm["var_burden"].iloc[:, 0]],
            axis=1,
            keys=["Base", "PM"],
        )
        err = ut.calc_error(metrics_var, self.error)

        warmup = metrics.index[-1] - self.x_len

        ax = plt.subplot()
        self.make_avg_plot(
            ax,
            metrics * 100,
            err * 100,
            ["Base", "PM"],
            (metrics.index - warmup) / 24,
            "Time (days)",
            "% of Income",
            show_labels=True,
        )

        plt.savefig(
            self.pub_loc + "equity_metrics." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def make_cost_plots(self, in_data=None, name="", map=False):
        """Make total cost plots showing tap, bottle, and total cost"""
        if in_data is None:
            cost_base = self.base
            cost_basebw = self.basebw
            cost_pm = self.pm
            cost_pm_nobw = self.pm_nobw
        else:
            cost_base = in_data[0]
            cost_basebw = in_data[1]
            cost_pm_nobw = in_data[2]
            cost_pm = in_data[3]

        # print(cost_basebw["cost"]["bw_cost"].iloc[:, :-2].mean(axis=0))

        cost_li = [
            cost_basebw["cost"]["bw_cost"][(cost_basebw["cowpi"]["level"] == 0)].iloc[
                :, -2
            ],
            cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 0)].iloc[:, -2],
        ]
        cost_hi = [
            cost_basebw["cost"]["bw_cost"][(cost_basebw["cowpi"]["level"] == 1)].iloc[
                :, -2
            ],
            cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 1)].iloc[:, -2],
        ]

        data_cost = pd.concat(
            [
                cost_basebw["cost"]["bw_cost"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].mean(),
                cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 0)].mean(),
                cost_basebw["cost"]["bw_cost"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].mean(),
                cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 1)].mean(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )
        var_cost = pd.concat(
            [
                cost_basebw["cost"]["bw_cost"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].var(),
                cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 0)].var(),
                cost_basebw["cost"]["bw_cost"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].var(),
                cost_pm["cost"]["bw_cost"][(cost_pm["cowpi"]["level"] == 1)].var(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )

        # print(data_cost)
        data_cost = data_cost.drop("i", axis=0)
        data_cost.index = data_cost.index.astype("int64")
        var_cost = var_cost.drop("i", axis=0)

        err_cost = ut.calc_error(var_cost, self.error)

        # print(data_cost)
        # print(data_cost.index / 24)
        # print(err_cost)

        ax = plt.subplot()
        self.make_avg_plot(
            ax=ax,
            data=data_cost,
            sd=err_cost,
            cols=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
            x_values=(data_cost.index / 24) - 30,
            ylabel="Cost ($)",
            xlabel="Time (days)",
            show_labels=True,
        )

        plt.savefig(
            self.pub_loc + name + "bw_cost_income." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        data_cost = pd.concat(
            [
                cost_basebw["cost"]["tw_cost"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].mean(),
                cost_pm["cost"]["tw_cost"][(cost_pm["cowpi"]["level"] == 0)].mean(),
                cost_basebw["cost"]["tw_cost"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].mean(),
                cost_pm["cost"]["tw_cost"][(cost_pm["cowpi"]["level"] == 1)].mean(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )
        var_cost = pd.concat(
            [
                cost_basebw["cost"]["tw_cost"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].var(),
                cost_pm["cost"]["tw_cost"][(cost_pm["cowpi"]["level"] == 0)].var(),
                cost_basebw["cost"]["tw_cost"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].var(),
                cost_pm["cost"]["tw_cost"][(cost_pm["cowpi"]["level"] == 1)].var(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )

        data_cost = data_cost.drop("i", axis=0)
        data_cost.index = data_cost.index.astype("int64")
        var_cost = var_cost.drop("i", axis=0)

        err_cost = ut.calc_error(var_cost, self.error)

        # print(data_cost)
        # print(data_cost.index / 24)
        # print(err_cost)

        ax = plt.subplot()
        self.make_avg_plot(
            ax=ax,
            data=data_cost,
            sd=err_cost,
            cols=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
            x_values=(data_cost.index / 24) - 30,
            ylabel="Cost ($)",
            xlabel="Time (days)",
            show_labels=True,
        )

        plt.savefig(
            self.pub_loc + name + "tw_cost_income." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        data_cost = pd.concat(
            [
                cost_basebw["cost"]["total"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].mean(),
                cost_pm["cost"]["total"][(cost_pm["cowpi"]["level"] == 0)].mean(),
                cost_basebw["cost"]["total"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].mean(),
                cost_pm["cost"]["total"][(cost_pm["cowpi"]["level"] == 1)].mean(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )
        var_cost = pd.concat(
            [
                cost_basebw["cost"]["total"][
                    (cost_basebw["cowpi"]["level"] == 0)
                ].var(),
                cost_pm["cost"]["total"][(cost_pm["cowpi"]["level"] == 0)].var(),
                cost_basebw["cost"]["total"][
                    (cost_basebw["cowpi"]["level"] == 1)
                ].var(),
                cost_pm["cost"]["total"][(cost_pm["cowpi"]["level"] == 1)].var(),
            ],
            axis=1,
            keys=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
        )

        data_cost = data_cost.drop("i", axis=0)
        data_cost.index = data_cost.index.astype("int64")
        var_cost = var_cost.drop("i", axis=0)

        err_cost = ut.calc_error(var_cost, self.error)

        # print(data_cost)
        # print(self.basebw["cowpi"][self.basebw["cowpi"]["level"] == 0]["cost"].mean())
        # print(self.basebw["cowpi"]["level"])
        # print(err_cost)

        ax = plt.subplot()
        self.make_avg_plot(
            ax=ax,
            data=data_cost,
            sd=err_cost,
            cols=[
                "Low-income TWA",
                "Low-income TWA+PM",
                "High-income TWA",
                "High-income TWA+PM",
            ],
            x_values=(data_cost.index / 24) - 30,
            ylabel="Cost ($)",
            xlabel="Time (days)",
            show_labels=True,
        )

        plt.savefig(
            self.pub_loc + name + "total_cost_income." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        # cost_leg = ['Bottled Water', 'Tap Water', 'Total']
        # cost_b = pd.concat(
        #     [self.basebw['cost']['bw_cost'].mean(axis=0),
        #      self.basebw['cost']['tw_cost'].mean(axis=0),
        #      self.basebw['cost']['total'].mean(axis=0)],
        #     axis=1,
        #     keys=cost_leg
        # )
        # cost_p = pd.concat(
        #     [self.pm['cost']['bw_cost'].mean(axis=0),
        #      self.pm['cost']['tw_cost'].mean(axis=0),
        #      self.pm['cost']['total'].mean(axis=0)],
        #     axis=1,
        #     keys=cost_leg
        # )

        # fig, axes = plt.subplots(1, 2, sharey=True)
        # axes[0] = self.make_avg_plot(
        #     axes[0], cost_b, None, cost_leg,
        #     cost_b.index / 24, sd_plot=False
        # )
        # axes[1] = self.make_avg_plot(
        #     axes[1], cost_p, None, cost_leg,
        #     cost_b.index / 24, sd_plot=False
        # )

        # fmt = '${x:,.0f}'
        # tick = mtick.StrMethodFormatter(fmt)
        # axes[0].yaxis.set_major_formatter(tick)

        # plt.gcf().set_size_inches(7, 3.5)
        # fig.supxlabel('Time (days)', y=-0.03)
        # fig.supylabel('Cost', x=0.04)
        # axes[0].legend(cost_leg, loc='upper left')
        # axes[0].text(0.5, -0.14, "(a)", size=12, ha="center",
        #              transform=axes[0].transAxes)
        # axes[1].text(0.5, -0.14, "(b)", size=12, ha="center",
        #              transform=axes[1].transAxes)

        # plt.savefig(self.pub_loc + 'cost.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        """ Make income based cost plots """
        cost_low = [
            cost_base["cowpi"][cost_base["cowpi"]["level"] == 0]["cost"],
            cost_basebw["cowpi"][cost_basebw["cowpi"]["level"] == 0]["cost"],
            cost_pm_nobw["cowpi"][cost_pm_nobw["cowpi"]["level"] == 0]["cost"],
            cost_pm["cowpi"][cost_pm["cowpi"]["level"] == 0]["cost"],
        ]

        cost_high = [
            cost_base["cowpi"][cost_base["cowpi"]["level"] == 1]["cost"],
            cost_basebw["cowpi"][cost_basebw["cowpi"]["level"] == 1]["cost"],
            cost_pm_nobw["cowpi"][cost_pm_nobw["cowpi"]["level"] == 1]["cost"],
            cost_pm["cowpi"][cost_pm["cowpi"]["level"] == 1]["cost"],
        ]

        self.make_income_comp_plot(
            [cost_low, cost_high],
            name + "cost_boxplot",
            ["Base", "TWA", "PM", "TWA+PM"],
            ylabel="Cost ($)",
            box=1,
            means=True,
            income_line=None,
            outliers="",
            legend_kwds={"labels": ["Low-income", "High-income"], "loc": "upper left"},
        )

        print("low-income cost values")
        for i in cost_low:
            print(i.median())
            print(i.max())
        print("high-income cost values")
        for i in cost_high:
            print(i.median())
            print(i.max())

        """ Make demographic cost plots """
        fix, axes = plt.subplots(3, 1)

        cost_low_race = self.filter_demo(0, "white", "cost", data=[self.base, self.pm])
        cost_high_race = self.filter_demo(1, "white", "cost", data=[self.base, self.pm])

        print("low-income cost values")
        for _, i in cost_low_race.items():
            for j in i:
                print(j.median())
        print("high-income cost values")
        for _, i in cost_high_race.items():
            for j in i:
                print(j.median())

        axes[0] = self.make_income_comp_plot(
            [
                cost_low_race["white"],
                cost_low_race["nonwhite"],
                cost_high_race["white"],
                cost_high_race["nonwhite"],
            ],
            xlabel=["Base", "TWA+PM"],
            ylabel="Cost ($)",
            box=1,
            means=False,
            income_line=None,
            outliers="",
            legend_kwds={
                "labels": [
                    "Low-income White",
                    "Low-income Non-white",
                    "High-income White",
                    "High-income Non-white",
                ],
                "loc": "upper left",
            },
            ax=axes[0],
        )

        # hispanic cost boxplot
        cost_low_hispanic = self.filter_demo(
            0, "hispanic", "cost", data=[self.base, self.pm]
        )
        cost_high_hispanic = self.filter_demo(
            1, "hispanic", "cost", data=[self.base, self.pm]
        )
        print("low-income cost values")
        for _, i in cost_low_hispanic.items():
            for j in i:
                print(j.median())
        print("high-income cost values")
        for _, i in cost_high_hispanic.items():
            for j in i:
                print(j.median())

        axes[1] = self.make_income_comp_plot(
            [
                cost_low_hispanic["hispanic"],
                cost_low_hispanic["nonhispanic"],
                cost_high_hispanic["hispanic"],
                cost_high_hispanic["nonhispanic"],
            ],
            name + "cost_boxplot_hispanic",
            ["Base", "TWA+PM"],
            ylabel="Cost ($)",
            box=1,
            means=False,
            income_line=None,
            outliers="",
            legend_kwds={
                "labels": [
                    "Low-income Hispanic",
                    "Low-income Non-Hispanic",
                    "High-income Hispanic",
                    "High-income Non-Hispanic",
                ],
                "loc": "upper left",
            },
            ax=axes[1],
        )

        # renter cost boxplot
        cost_low_renter = self.filter_demo(
            0, "renter", "cost", data=[self.base, self.pm]
        )
        cost_high_renter = self.filter_demo(
            1, "renter", "cost", data=[self.base, self.pm]
        )

        print("low-income cost values")
        for _, i in cost_low_renter.items():
            for j in i:
                print(j.median())
        print("high-income cost values")
        for _, i in cost_high_renter.items():
            for j in i:
                print(j.median())

        axes[2] = self.make_income_comp_plot(
            [
                cost_low_renter["renter"],
                cost_low_renter["nonrenter"],
                cost_high_renter["renter"],
                cost_high_renter["nonrenter"],
            ],
            name + "cost_boxplot_renter",
            ["Base", "TWA+PM"],
            ylabel="Cost ($)",
            box=1,
            means=False,
            income_line=None,
            outliers="",
            legend_kwds={
                "labels": [
                    "Low-income Renter",
                    "Low-income Non-renter",
                    "High-income Renter",
                    "High-income Non-renter",
                ],
                "loc": "upper left",
            },
            ax=axes[2],
        )

        axes[0].text(
            0.5, -0.1, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.1, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )
        axes[2].text(
            0.5, -0.1, "(c)", size=12, ha="center", transform=axes[2].transAxes
        )

        plt.gcf().set_size_inches(3.5, 7)

        plt.savefig(
            self.pub_loc + name + "cost_demo_boxplots." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        if map:
            fig, axes = plt.subplots(1, 2)
            axes[0] = self.bg_map(
                ax=axes[0],
                node_data=cost_base["cowpi"][["cost", "wdn_node"]]
                .groupby("wdn_node")
                .mean()["cost"],
                wn_nodes=True,
                node_cmap="viridis",
                vmax_inp=700,
            )
            axes[1] = self.bg_map(
                ax=axes[1],
                node_data=cost_pm["cowpi"][["cost", "wdn_node"]]
                .groupby("wdn_node")
                .mean()["cost"],
                wn_nodes=True,
                node_cmap="viridis",
                vmax_inp=700,
            )
            axes[0].text(
                0.5, -0.04, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.04, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )

            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0] = self.bg_map(
            #     ax=axes[0, 0],
            #     node_data=cost_base["cowpi"][["cost", "wdn_node"]]
            #     .groupby("wdn_node")
            #     .mean()["cost"],
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=700,
            # )
            # axes[0, 1] = self.bg_map(
            #     ax=axes[0, 1],
            #     node_data=cost_basebw["cowpi"][["cost", "wdn_node"]]
            #     .groupby("wdn_node")
            #     .mean()["cost"],
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=700,
            # )
            # axes[1, 0] = self.bg_map(
            #     ax=axes[1, 0],
            #     node_data=cost_pm_nobw["cowpi"][["cost", "wdn_node"]]
            #     .groupby("wdn_node")
            #     .mean()["cost"],
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=700,
            # )
            # axes[1, 1] = self.bg_map(
            #     ax=axes[1, 1],
            #     node_data=cost_pm["cowpi"][["cost", "wdn_node"]]
            #     .groupby("wdn_node")
            #     .mean()["cost"],
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=700,
            # )

            # axes[0, 0].text(
            #     0.5, -0.08, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            # )
            # axes[0, 1].text(
            #     0.5, -0.08, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            # )
            # axes[1, 0].text(
            #     0.5, -0.08, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            # )
            # axes[1, 1].text(
            #     0.5, -0.08, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            # )

            plt.gcf().set_size_inches(7, 3.5)

            fig.subplots_adjust(bottom=0.12, wspace=0.1, hspace=0.08)
            cbar_ax = fig.add_axes(rect=(0.165, 0.05, 0.7, 0.02))

            norm = mpl.colors.Normalize(vmin=0, vmax=700)
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
                cax=cbar_ax,
                orientation="horizontal",
                label="Cost ($)",
            )
            plt.savefig(
                self.pub_loc + name + "cost_network." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

    def cowpi_boxplot(
        self,
        in_data=None,
        name="",
        demographics=False,
        di=False,
        perc=False,
        sa=False,
        map=False,
    ):
        if in_data is None:
            base_data = self.base
            basebw_data = self.basebw
            pm_nobw_data = self.pm_nobw
            pm_data = self.pm
        else:
            base_data = in_data[0]
            basebw_data = in_data[1]
            pm_nobw_data = in_data[2]
            pm_data = in_data[3]

        print(base_data["cowpi"][["income", "i"]].groupby("i").median().mean())
        print(base_data["cowpi"][["income", "i"]].groupby("i").median().std() / math.sqrt(30))
        print(base_data["cowpi"][["income", "i"]].groupby("i").quantile(0.2))
        print(base_data["cowpi"]["income"].median())
        print(base_data["cowpi"]["income"].mean())
        print(base_data["cowpi"]["income"].quantile(0.2))

        """Make cowpi boxplots"""
        cowpi_bot20 = [
            base_data["cowpi"][base_data["cowpi"]["level"] == 0]["cowpi"] * 100,
            basebw_data["cowpi"][basebw_data["cowpi"]["level"] == 0]["cowpi"] * 100,
            pm_nobw_data["cowpi"][pm_nobw_data["cowpi"]["level"] == 0]["cowpi"] * 100,
            pm_data["cowpi"][pm_data["cowpi"]["level"] == 0]["cowpi"] * 100,
        ]

        cowpi_top80 = [
            base_data["cowpi"][base_data["cowpi"]["level"] == 1]["cowpi"] * 100,
            basebw_data["cowpi"][basebw_data["cowpi"]["level"] == 1]["cowpi"] * 100,
            pm_nobw_data["cowpi"][pm_nobw_data["cowpi"]["level"] == 1]["cowpi"] * 100,
            pm_data["cowpi"][pm_data["cowpi"]["level"] == 1]["cowpi"] * 100,
        ]

        """ Print some stats about low-income households """
        for i in cowpi_bot20:
            print((i > 4.5).sum() / len(i))
            # print((i > 4.5).sum() / (i > 0).sum())
        for i in cowpi_top80:
            print((i > 4.5).sum() / len(i))
            # print((i > 4.5).sum() / (i > 0).sum())

        # print total population data
        print("Total population data")
        for i in range(len(cowpi_bot20)):
            print(
                ((cowpi_bot20[i] > 4.5).sum() + (cowpi_top80[i] > 4.5).sum())
                / (len(cowpi_bot20[i]) + len(cowpi_top80[i]))
            )

        print("%HI median values:")
        print([a.median() for a in cowpi_bot20])
        print([a.median() for a in cowpi_top80])

        # self.make_income_comp_plot(
        #     data,
        #     "cow_boxplot",
        #     ["Base", "Base+BW", "PM", "PM+BW"],
        #     # ['Base', 'Base+BW', 'SD+BW'],
        #     box=True,
        # )

        self.make_income_comp_plot(
            [cowpi_bot20, cowpi_top80],
            name + "cow_boxplot_income",
            ["Base", "TWA", "PM", "TWA+PM"],
            # ['Base', 'Base+BW', 'SD+BW'],
            ylabel="%HI",
            box=1,
            means=False,
            outliers="",
            income_line=None,
            legend_kwds={"labels": ["Low-income", "High-income"], "loc": "upper left"},
        )

        """ Make plot of intersection between income and whether node exceeds
        threshold """
        nodes_w_demand = [
            name
            for name, node in self.wn.junctions()
            if node.demand_timeseries_list[0].base_value > 0
        ]
        print(self.calc_age_diff(base_data["avg_age"], nodes_w_demand))
        base_age = pd.DataFrame(
            self.calc_age_diff(base_data["avg_age"], nodes_w_demand),
            index=["data"]
        ).T
        pm_age = pd.DataFrame(
            self.calc_age_diff(pm_data["avg_age"], nodes_w_demand),
            index=["data"]
        ).T

        base_nodes = base_age[base_age["data"]].index.to_list()
        pm_nodes = pm_age[pm_age["data"]].index.to_list()

        # for row in base_data["cowpi"].iterrows():
        #     print(row)
        # print(base_data["cowpi"])

        cowpi_lowI = [
            base_data["cowpi"][base_data["cowpi"]["level"] == 0][["cowpi", "wdn_node"]],
            pm_data["cowpi"][pm_data["cowpi"]["level"] == 0][["cowpi", "wdn_node"]],
        ]

        cowpi_highI = [
            base_data["cowpi"][base_data["cowpi"]["level"] == 1][["cowpi", "wdn_node"]],
            pm_data["cowpi"][pm_data["cowpi"]["level"] == 1][["cowpi", "wdn_node"]],
        ]

        print(cowpi_lowI[0])

        base_below = [
            cowpi_lowI[0][~cowpi_lowI[0]["wdn_node"].isin(base_nodes)]["cowpi"] * 100,
            cowpi_highI[0][~cowpi_highI[0]["wdn_node"].isin(base_nodes)]["cowpi"] * 100,
        ]

        base_above = [
            cowpi_lowI[0][cowpi_lowI[0]["wdn_node"].isin(base_nodes)]["cowpi"] * 100,
            cowpi_highI[0][cowpi_highI[0]["wdn_node"].isin(base_nodes)]["cowpi"] * 100,
        ]

        pm_below = [
            cowpi_lowI[1][~cowpi_lowI[1]["wdn_node"].isin(pm_nodes)]["cowpi"] * 100,
            cowpi_highI[1][~cowpi_highI[1]["wdn_node"].isin(pm_nodes)]["cowpi"] * 100,
        ]

        pm_above = [
            cowpi_lowI[1][cowpi_lowI[1]["wdn_node"].isin(pm_nodes)]["cowpi"] * 100,
            cowpi_highI[1][cowpi_highI[1]["wdn_node"].isin(pm_nodes)]["cowpi"] * 100,
        ]

        fig, axes = plt.subplots(1, 2, sharey=True)
        axes[0] = self.make_income_comp_plot(
            [base_below, base_above],
            name + "cow_boxplot_income",
            ["Low-income", "High-income"],
            # ylabel="%HI",
            box=1,
            means=False,
            outliers="",
            income_line=None,
            legend_kwds={"labels": ["<130 hours", ">130 hours"], "loc": "best"},
            ax=axes[0]
        )
        axes[1] = self.make_income_comp_plot(
            [pm_below, pm_above],
            name + "cow_boxplot_income",
            ["Low-income", "High-income"],
            # ylabel="%HI",
            box=1,
            means=False,
            outliers="",
            income_line=None,
            legend_kwds={"labels": ["<130 hours", ">130 hours"], "loc": "best"},
            ax=axes[1]
        )

        for i in base_below:
            print(i.median())
        for i in base_above:
            print(i.median())
        for i in pm_below:
            print(i.median())
        for i in pm_above:
            print(i.median())

        axes[0].text(
            0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )

        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(
            self.pub_loc + name + "cow_threshold_boxplot." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        if demographics:
            """Make race cross low-income plot"""
            low_race = self.filter_demo(
                0, "white", "cowpi", 100, data=[self.base, self.pm]
            )
            high_race = self.filter_demo(
                1, "white", "cowpi", 100, data=[self.base, self.pm]
            )

            for i in low_race["white"]:
                print((i > 4.5).sum() / len(i))
            for i in low_race["nonwhite"]:
                print((i > 4.5).sum() / len(i))
            for i in high_race["white"]:
                print((i > 4.5).sum() / len(i))
            for i in high_race["nonwhite"]:
                print((i > 4.5).sum() / len(i))

            print("Race %HI median values:")
            print([a.median() for a in low_race["white"]])
            print([a.median() for a in low_race["nonwhite"]])
            print([a.median() for a in high_race["white"]])
            print([a.median() for a in high_race["nonwhite"]])

            fix, axes = plt.subplots(3, 1)

            axes[0] = self.make_income_comp_plot(
                [
                    low_race["white"],
                    low_race["nonwhite"],
                    high_race["white"],
                    high_race["nonwhite"],
                ],
                "cow_boxplot_race",
                ["Base", "TWA+PM"],
                box=1,
                means=False,
                outliers="",
                income_line=None,
                legend_kwds={
                    "labels": [
                        "Low-income White",
                        "Low-income Non-white",
                        "High-income White",
                        "High-income Non-white",
                    ],
                    "loc": "best",
                },
                ax=axes[0],
            )

            """ Make hispanic low-income plot """
            low_hispanic = self.filter_demo(
                0, "hispanic", "cowpi", 100, data=[self.base, self.pm]
            )
            high_hispanic = self.filter_demo(
                1, "hispanic", "cowpi", 100, data=[self.base, self.pm]
            )

            print("Hispanic %HI median values:")
            print([a.median() for a in low_hispanic["hispanic"]])
            print([a.median() for a in low_hispanic["nonhispanic"]])
            print([a.median() for a in high_hispanic["hispanic"]])
            print([a.median() for a in high_hispanic["nonhispanic"]])

            axes[1] = self.make_income_comp_plot(
                [
                    low_hispanic["hispanic"],
                    low_hispanic["nonhispanic"],
                    high_hispanic["hispanic"],
                    high_hispanic["nonhispanic"],
                ],
                "cow_boxplot_hispanic",
                ["Base", "TWA+PM"],
                box=1,
                means=False,
                outliers="",
                income_line=None,
                legend_kwds={
                    "labels": [
                        "Low-income Hispanic",
                        "Low-income Non-Hispanic",
                        "High-income Hispanic",
                        "High-income Non-Hispanic",
                    ],
                    "loc": "best",
                },
                ax=axes[1],
            )

            """ Make hispanic low-income plot """
            low_renter = self.filter_demo(
                0, "renter", "cowpi", 100, data=[self.base, self.pm]
            )
            high_renter = self.filter_demo(
                1, "renter", "cowpi", 100, data=[self.base, self.pm]
            )

            print("Renter %HI median values:")
            print([a.median() for a in low_renter["renter"]])
            print([a.median() for a in low_renter["nonrenter"]])
            print([a.median() for a in high_renter["renter"]])
            print([a.median() for a in high_renter["nonrenter"]])

            axes[2] = self.make_income_comp_plot(
                [
                    low_renter["renter"],
                    low_renter["nonrenter"],
                    high_renter["renter"],
                    high_renter["nonrenter"],
                ],
                "cow_boxplot_renter",
                ["Base", "TWA+PM"],
                box=1,
                means=False,
                outliers="",
                income_line=None,
                legend_kwds={
                    "labels": [
                        "Low-income Renter",
                        "Low-income Non-renter",
                        "High-income Renter",
                        "High-income Non-renter",
                    ],
                    "loc": "best",
                },
                ax=axes[2],
            )
            axes[0].text(
                0.5, -0.1, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.1, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )
            axes[2].text(
                0.5, -0.1, "(c)", size=12, ha="center", transform=axes[2].transAxes
            )

            plt.gcf().set_size_inches(3.5, 7)

            plt.savefig(
                self.pub_loc + name + "cow_demo_boxplots." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            """ Plot or export data that shows how much more likely
            a household is to have unaffordable water based on demographics """
            # first find the households that exceed threhold and are in
            # demo groups
            # race_unaffordable = self.threshold_demo("white")
            # race_all = self.threshold_demo("white", threshold=0)

            # white_unaffordable = np.array([len(d) for d in race_unaffordable["low"]])
            # white_all = np.array([len(d) for d in race_all["low"]])
            # hoc_unaffordable = np.array([len(d) for d in race_unaffordable["high"]])
            # hoc_all = np.array([len(d) for d in race_all["high"]])
            # # hoc = [d.mean() for d in data["high"]]
            # print(white_unaffordable)
            # print(white_all)
            # print(hoc_unaffordable)
            # print(hoc_all)

            # rr = self.calc_risk(
            #     hoc_unaffordable, hoc_all, white_unaffordable, white_all
            # )

            # print(rr)

            # fig, axes = plt.subplots(2, 2)

            # axes[0, 0] = self.bg_map(
            #      axes[0, 0], "median_income", , wn_nodes=True
            # )
            # axes[0, 1] = self.bg_map(
            #      axes[0, 1], "perc_renter", pd.Series(diff_age), wn_nodes=True
            # )
            # axes[1, 0] = self.bg_map(
            #      axes[1, 0], "perc_w", race_unaffordable, wn_nodes=True
            # )
            # axes[1, 1] = self.bg_map(
            #      axes[1, 1], "perc_nh", pd.Series(diff_age), wn_nodes=True
            # )

            # axes[0, 0].text(
            #     0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            # )
            # axes[0, 1].text(
            #     0.5, -0.14, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            # )
            # axes[1, 0].text(
            #     0.5, -0.23, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            # )
            # axes[1, 1].text(
            #     0.5, -0.23, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            # )
            # # fig.supxlabel("Time (days)", y=0)
            # # fig.supylabel("Age (hrs)", x=0.04)
            # plt.gcf().set_size_inches(7, 7)

            # plt.savefig(
            #     self.pub_loc + "intersection_age." + self.format,
            #     format=self.format,
            #     bbox_inches="tight",
            # )
            # plt.close()

        """ Make plots comparing income distance scenarios """
        if di:
            # need to change 0 to 1 here if reverting back low medium and high
            cowpi_low = [
                self.pm_nodi["cowpi"][self.pm_nodi["cowpi"]["level"] == 0]["cowpi"]
                * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 0]["cowpi"] * 100,
            ]

            cowpi_high = [
                self.pm_nodi["cowpi"][self.pm_nodi["cowpi"]["level"] == 1]["cowpi"]
                * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 1]["cowpi"] * 100,
            ]

            # cowpi_med = [
            #     self.pm_nodi['cowpi'][self.pm_nodi['cowpi']['level'] == 2]['cowpi']*100,
            #     self.pm['cowpi'][self.pm['cowpi']['level'] == 2]['cowpi']*100
            # ]

            # cowpi_high = [
            #     self.pm_nodi['cowpi'][self.pm_nodi['cowpi']['level'] == 3]['cowpi']*100,
            #     self.pm['cowpi'][self.pm['cowpi']['level'] == 3]['cowpi']*100
            # ]

            # cowpi_lower20 = [
            #     self.pm_nodi['cowpi'].quantile(0.2)['income'],
            #     self.pm_nodi['cowpi'].quantile(0.5)['income'],
            #     self.pm_nodi['cowpi'].quantile(0.9)['income']
            # ]
            # print(cowpi_lower20)

            # data = {
            #     'low': cowpi_low,
            #     'med': cowpi_med,
            #     'high': cowpi_high
            # }

            data = {"low": cowpi_low, "high": cowpi_high}

            self.make_income_comp_plot(
                data,
                "cow_boxplot_di",
                ["No DI", "DI"],
                box=True,
                means=False,
                outliers="",
            )

        """ Make plots comparing percentage vs absolute twa scenarios """
        if perc:
            cowpi_low = [
                self.pm_perc["cowpi"][self.pm_perc["cowpi"]["level"] == 0]["cowpi"]
                * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 0]["cowpi"] * 100,
            ]

            cowpi_high = [
                self.pm_perc["cowpi"][self.pm_perc["cowpi"]["level"] == 1]["cowpi"]
                * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 1]["cowpi"] * 100,
            ]

            # cowpi_med = [
            #     self.pm_perc['cowpi'][self.pm_perc['cowpi']['level'] == 2]['cowpi'].groupby(level=0).mean()*100,
            #     self.pm['cowpi'][self.pm['cowpi']['level'] == 2]['cowpi'].groupby(level=0).mean()*100
            # ]

            # cowpi_high = [
            #     self.pm_perc['cowpi'][self.pm_perc['cowpi']['level'] == 3]['cowpi'].groupby(level=0).mean()*100,
            #     self.pm['cowpi'][self.pm['cowpi']['level'] == 3]['cowpi'].groupby(level=0).mean()*100
            # ]

            # data = {
            #     'low': cowpi_low,
            #     'med': cowpi_med,
            #     'high': cowpi_high
            # }
            data = {"low": cowpi_low, "high": cowpi_high}

            self.make_income_comp_plot(
                data,
                "cow_boxplot_perc",
                ["Percentage", "Absolute"],
                box=True,
                outliers="",
            )

        """ Make plots for each SA scenario, no drinking, no cooking, and no
        hygiene """
        if sa:
            cowpi_low = [
                self.pm_noD["cowpi"][self.pm_noD["cowpi"]["level"] == 0]["cowpi"] * 100,
                self.pm_noC["cowpi"][self.pm_noC["cowpi"]["level"] == 0]["cowpi"] * 100,
                self.pm_noH["cowpi"][self.pm_noH["cowpi"]["level"] == 0]["cowpi"] * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 0]["cowpi"] * 100,
            ]

            cowpi_high = [
                self.pm_noD["cowpi"][self.pm_noD["cowpi"]["level"] == 1]["cowpi"] * 100,
                self.pm_noC["cowpi"][self.pm_noC["cowpi"]["level"] == 1]["cowpi"] * 100,
                self.pm_noH["cowpi"][self.pm_noH["cowpi"]["level"] == 1]["cowpi"] * 100,
                self.pm["cowpi"][self.pm["cowpi"]["level"] == 1]["cowpi"] * 100,
            ]

            data = {"low": cowpi_low, "high": cowpi_high}

            self.make_income_comp_plot(
                data,
                "cow_boxplot_exclusion",
                [
                    "Excluding Drink",
                    "Excluding Cook",
                    "Excluding Hygience",
                    "No Exclusions",
                ],
                box=True,
                means=False,
                outliers="",
            )

        if map:
            fig, axes = plt.subplots(1, 2)
            axes[0] = self.bg_map(
                ax=axes[0],
                node_data=base_data["cowpi"][base_data["cowpi"]["level"] == 0][
                    ["cowpi", "wdn_node"]
                ]
                .groupby("wdn_node")
                .mean()["cowpi"]
                * 100,
                wn_nodes=True,
                node_cmap="viridis",
                vmax_inp=10,
            )
            axes[1] = self.bg_map(
                ax=axes[1],
                node_data=pm_data["cowpi"][pm_data["cowpi"]["level"] == 0][
                    ["cowpi", "wdn_node"]
                ]
                .groupby("wdn_node")
                .mean()["cowpi"]
                * 100,
                wn_nodes=True,
                node_cmap="viridis",
                vmax_inp=10,
            )
            axes[0].text(
                0.5, -0.04, "(a)", size=12, ha="center", transform=axes[0].transAxes
            )
            axes[1].text(
                0.5, -0.04, "(b)", size=12, ha="center", transform=axes[1].transAxes
            )

            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0] = self.bg_map(
            #     ax=axes[0, 0],
            #     node_data=base_data["cowpi"][base_data["cowpi"]["level"] == 0][
            #         ["cowpi", "wdn_node"]
            #     ]
            #     .groupby("wdn_node")
            #     .mean()["cowpi"]
            #     * 100,
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=10,
            #     # legend_bool=True,
            #     # label="%HI",
            # )
            # axes[0, 1] = self.bg_map(
            #     ax=axes[0, 1],
            #     node_data=basebw_data["cowpi"][basebw_data["cowpi"]["level"] == 0][
            #         ["cowpi", "wdn_node"]
            #     ]
            #     .groupby("wdn_node")
            #     .mean()["cowpi"]
            #     * 100,
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=10,
            # )
            # axes[1, 0] = self.bg_map(
            #     ax=axes[1, 0],
            #     node_data=pm_nobw_data["cowpi"][pm_nobw_data["cowpi"]["level"] == 0][
            #         ["cowpi", "wdn_node"]
            #     ]
            #     .groupby("wdn_node")
            #     .mean()["cowpi"]
            #     * 100,
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=10,
            # )
            # axes[1, 1] = self.bg_map(
            #     ax=axes[1, 1],
            #     node_data=pm_data["cowpi"][pm_data["cowpi"]["level"] == 0][
            #         ["cowpi", "wdn_node"]
            #     ]
            #     .groupby("wdn_node")
            #     .mean()["cowpi"]
            #     * 100,
            #     wn_nodes=True,
            #     node_cmap="viridis",
            #     vmax_inp=10,
            # )

            # axes[0, 0].text(
            #     0.5, -0.08, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
            # )
            # axes[0, 1].text(
            #     0.5, -0.08, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
            # )
            # axes[1, 0].text(
            #     0.5, -0.08, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
            # )
            # axes[1, 1].text(
            #     0.5, -0.08, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
            # )

            plt.gcf().set_size_inches(7, 3.5)

            fig.subplots_adjust(bottom=0.12, wspace=0.1, hspace=0.08)
            cbar_ax = fig.add_axes(rect=(0.165, 0.05, 0.7, 0.02))

            norm = mpl.colors.Normalize(vmin=0, vmax=10)
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
                cax=cbar_ax,
                orientation="horizontal",
                label="%HI",
            )
            plt.savefig(
                self.pub_loc + name + "cowpi_network." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            # fig, axes = plt.subplots(1, 1)
            # axes = self.bg_map(
            #     ax=axes,
            #     node_data=pm_data["cowpi"][pm_data["cowpi"]["level"] == 0][
            #         ["cowpi", "wdn_node"]
            #     ]
            #     .groupby("wdn_node")
            #     .mean()
            #     * 100,
            #     wn_nodes=True,
            #     node_cmap="Oranges",
            #     vmax_inp=10,
            #     # legend_bool=True,
            #     # label="Cost of water / household income (%)",
            # )
            # plt.savefig(
            #     self.pub_loc + name + "cowpi_network_3mt." + self.format,
            #     format=self.format,
            #     bbox_inches="tight",
            #     transparent=self.transparent,
            # )
            # plt.close()

    def make_city_map(self):
        """Plot the block groups of clinton"""
        fig, axes = plt.subplots(1, 2)
        axes[0] = self.bg_map(axes[0], wn_nodes=False, label_bg=True, plot_wn=False)
        # plt.gcf().set_size_inches(4, 4)
        # plt.savefig(
        #     self.pub_loc + "clinton-bg." + self.format,
        #     format=self.format,
        #     bbox_inches="tight",
        #     transparent=self.transparent,
        # )
        # plt.close()

        """ Plot the block groups with the wdn """
        # ax = plt.subplot()
        axes[1] = self.bg_map(
            axes[1],
            label_nodes="Diameter (mm)",
            wn_nodes=True,
            pipes=True,
            pipe_cmap="viridis",
        )
        axes[0].text(
            0.5, -0.1, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )
        plt.gcf().set_size_inches(7, 3.5)
        plt.savefig(
            self.pub_loc + "clinton-wdn_and_bg." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        """ Plot the block groups with the wdn, but without pipe diameters """
        ax = plt.subplot()
        ax = self.bg_map(ax, wn_nodes=True)
        plt.gcf().set_size_inches(4, 4)
        plt.savefig(
            self.pub_loc + "clinton-wdn_nopipeD." + self.format,
            format=self.format,
            bbox_inches="tight",
        )
        plt.close()

        """ Plot intersection with demographics """
        fig, axes = plt.subplots(2, 2)
        print(self.base["cowpi"])

        axes[0, 0] = self.bg_map(
            ax=axes[0, 0],
            display_demo="median_income",
            node_data=(
                self.base["cowpi"]
                .loc[self.base["cowpi"]["i"] == 0][["level", "wdn_node"]]
                .groupby("wdn_node")
                .median()
                .astype(bool)["level"]
            ),
            wn_nodes=True,
            label_map="Median Income",
            # node_cmap="Oranges",
            # lg_fmt=custom_format,
            legend_bool=True,
            label_nodes=["Low-income", "High-income"],
        )
        axes[1, 1] = self.bg_map(
            ax=axes[1, 1],
            display_demo="perc_renter",
            node_data=(
                self.base["cowpi"]
                .loc[self.base["cowpi"]["i"] == 0][["renter", "wdn_node"]]
                .groupby("wdn_node")
                .median()
                .astype(bool)["renter"]
            ),
            wn_nodes=True,
            label_map="% Renter",
            # node_cmap="Oranges",
            vmin_inp=0,
            legend_bool=True,
            label_nodes=["Non-renter", "Renter"],
        )
        axes[0, 1] = self.bg_map(
            ax=axes[0, 1],
            display_demo="perc_w",
            node_data=(
                self.base["cowpi"]
                .loc[self.base["cowpi"]["i"] == 0][["white", "wdn_node"]]
                .groupby("wdn_node")
                .median()
                .astype(bool)["white"]
            ),
            wn_nodes=True,
            label_map="% White",
            # node_cmap="Oranges",
            vmin_inp=0,
            legend_bool=True,
            label_nodes=["Non-white", "White"],
        )
        axes[1, 0] = self.bg_map(
            ax=axes[1, 0],
            display_demo="perc_nh",
            node_data=(
                self.base["cowpi"]
                .loc[self.base["cowpi"]["i"] == 0][["hispanic", "wdn_node"]]
                .groupby("wdn_node")
                .median()
                .astype(bool)["hispanic"]
            ),
            wn_nodes=True,
            label_map="% non-Hispanic",
            # node_cmap="Oranges",
            vmin_inp=0,
            legend_bool=True,
            label_nodes=["Non-Hispanic", "Hispanic"],
        )

        axes[0, 0].text(
            0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0, 0].transAxes
        )
        axes[0, 1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[0, 1].transAxes
        )
        axes[1, 0].text(
            0.5, -0.14, "(c)", size=12, ha="center", transform=axes[1, 0].transAxes
        )
        axes[1, 1].text(
            0.5, -0.14, "(d)", size=12, ha="center", transform=axes[1, 1].transAxes
        )
        # fig.supxlabel("Time (days)", y=0)
        # fig.supylabel("Age (hrs)", x=0.04)
        plt.gcf().set_size_inches(7, 6)

        plt.savefig(
            self.pub_loc + "clinton-intersection." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        """ Plot map with the percent residential """
        node_buildings = pd.read_pickle("buildings.pkl")
        counts = node_buildings.value_counts(["wdn_node", "type"]).unstack()
        perc_counts = counts.divide(counts.sum(axis=1) / 100, axis=0)
        # print(perc_counts)

        ax = plt.subplot()
        ax = self.bg_map(
            ax=ax,
            wn_nodes=True,
            node_data=perc_counts["res"],
            label_nodes="% Residential",
            node_cmap="viridis",
            legend_bool=True,
        )
        plt.savefig(
            self.pub_loc + "perc_res_map." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def cowpi_barchart(self):
        level_cowpi_b = self.base["cowpi"].groupby("level").mean()["cowpi"]
        std_cowpi_b = self.base["cowpi"].groupby("level").std()["cowpi"]
        # print(level_cowpi_b)

        level_cowpi_bbw = self.basebw["cowpi"].groupby("level").mean()["cowpi"]
        std_cowpi_bbw = self.basebw["cowpi"].groupby("level").std()["cowpi"]
        # print(level_cowpi_bbw)

        level_cowpi_p = self.pm["cowpi"].groupby("level").mean()["cowpi"]
        std_cowpi_p = self.pm["cowpi"].groupby("level").std()["cowpi"]
        # print(level_cowpi_p)

        level_cowpi_p25 = self.pm25ind["cowpi"].groupby("level").mean()["cowpi"]
        level_cowpi_p50 = self.pm50ind["cowpi"].groupby("level").mean()["cowpi"]
        level_cowpi_p75 = self.pm75ind["cowpi"].groupby("level").mean()["cowpi"]
        level_cowpi_p100 = self.pm100ind["cowpi"].groupby("level").mean()["cowpi"]

        std_cowpi_p25 = self.pm25ind["cowpi"].groupby("level").std()["cowpi"]
        std_cowpi_p50 = self.pm50ind["cowpi"].groupby("level").std()["cowpi"]
        std_cowpi_p75 = self.pm75ind["cowpi"].groupby("level").std()["cowpi"]
        std_cowpi_p100 = self.pm100ind["cowpi"].groupby("level").std()["cowpi"]

        cost_comp_basepm = pd.DataFrame(
            {
                "Base": level_cowpi_b,
                "Base+BW": level_cowpi_bbw,
                "Social Distancing+BW": level_cowpi_p,
            },
            index=[0, 1, 2, 3],
        )

        cost_std_basepm = pd.DataFrame(
            {
                "Base": std_cowpi_b,
                "Base+BW": std_cowpi_bbw,
                "Social Distancing+BW": std_cowpi_p,
            },
            index=[0, 1, 2, 3],
        )

        # convert to percentages
        cost_comp_basepm = cost_comp_basepm * 100
        cost_std_basepm = cost_std_basepm
        print(cost_comp_basepm)
        print(cost_std_basepm)

        cost_comp_basepm = cost_comp_basepm.rename(
            {0: "Extremely Low", 1: "Low", 2: "Medium", 3: "High"}
        )
        cost_std_basepm = cost_std_basepm.rename(
            {0: "Extremely Low", 1: "Low", 2: "Medium", 3: "High"}
        )

        # make the barchart
        self.make_income_comp_plot(cost_comp_basepm, "basepm")

        cost_comp_sa = pd.DataFrame(
            {
                "No Minimum": level_cowpi_p,
                # '25%': level_cowpi_p25,
                "50%": level_cowpi_p50,
                # '75%': level_cowpi_p75,
                "100%": level_cowpi_p100,
                "Base": level_cowpi_b,
            },
            index=[0, 1, 2, 3],
        )

        # cost_std_sa = pd.DataFrame(
        #     {'No Minimum': std_cowpi_p,
        #      '25%': std_cowpi_p25,
        #      '50%': std_cowpi_p50,
        #      '75%': std_cowpi_p75,
        #      '100%': std_cowpi_p100,
        #      'Base': std_cowpi_b},
        #     index=[0, 1, 2, 3]
        # )

        # convert to percentages
        cost_comp_sa = cost_comp_sa * 100
        # cost_std_sa = cost_std_sa

        cost_comp_sa = cost_comp_sa.rename(
            {0: "Extremely Low", 1: "Low", 2: "Medium", 3: "High"}
        )
        # cost_std_sa = cost_std_sa.rename({0: 'Extremely Low', 1: 'Low', 2: 'Medium', 3: 'High'})

        # make barchart
        self.make_income_comp_plot(cost_comp_sa, "sa")

    def make_twa_plots(self):
        """
        Tap water avoidance adoption plots
        """
        # print(self.pm['twa'])
        twa_keys = ["drink", "cook", "hygiene"]
        twas = ["Drink", "Cook", "Hygiene"]
        # print(self.basebw['twa'])
        twa_basebw = self.calc_twa_averages(self.basebw["twa"], twa_keys)
        for i in range(len(twa_basebw)):
            twa_basebw[i].index = twa_basebw[i].index - 719
            twa_basebw[i].loc[0] = [0, 0, 0]
            twa_basebw[i].sort_index(inplace=True)

        twa_pm = self.calc_twa_averages(self.pm["twa"], twa_keys)
        for i in range(len(twa_pm)):
            twa_pm[i].index = twa_pm[i].index - 719
            twa_pm[i].loc[0] = [0, 0, 0]
            twa_pm[i].sort_index(inplace=True)

        # households = len(self.pm['twa']['drink'].index)
        # print(twa_pm)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        axes[0] = self.make_avg_plot(
            axes[0],
            twa_basebw[0] * 100,
            ut.calc_error(twa_basebw[1], "ci95") * 100,
            twa_keys,
            twa_basebw[0].index / 24,
        )
        axes[1] = self.make_avg_plot(
            axes[1],
            twa_pm[0] * 100,
            ut.calc_error(twa_pm[1], "ci95") * 100,
            twa_keys,
            twa_pm[0].index / 24,
        )
        # axes[0] = self.make_avg_plot(
        #     axes[0], twa_basebw / households * 100, None,
        #     twas, twa_basebw.index / 24,
        #     sd_plot=False
        # )
        # axes[1] = self.make_avg_plot(
        #     axes[1], twa_pm / households * 100, None,
        #     twas, twa_pm.index / 24,
        #     sd_plot=False
        # )
        axes[0].legend(twas)

        axes[0].text(
            0.5, -0.14, "(a)", size=12, ha="center", transform=axes[0].transAxes
        )
        axes[1].text(
            0.5, -0.14, "(b)", size=12, ha="center", transform=axes[1].transAxes
        )
        fig.supxlabel("Time (days)", y=-0.03)
        fig.supylabel("Percent of Households", x=0.04)
        plt.gcf().set_size_inches(7, 3.5)

        plt.savefig(
            self.pub_loc + "twa_comp." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def sa_plots(self, age=True, cost=True, cowpi=True, map=True):
        nodes_w_demand = [
            name
            for name, node in self.wn.junctions()
            if node.demand_timeseries_list[0].base_value > 0
        ]

        if age:
            self.age_plots(
                data=[self.base, self.basebw_neg20, self.pm_neg20, self.pm_nobw],
                name="sa-20_",
                thres_n=110,
                map=True,
                threshold=True,
            )
            self.age_plots(
                [self.base, self.basebw_20, self.pm_20, self.pm_nobw],
                name="sa20_",
                thres_n=150,
                map=True,
                threshold=True,
            )

        if cost:
            self.make_cost_plots(
                in_data=[self.base, self.basebw_neg20, self.pm_nobw, self.pm_neg20],
                name="sa-20",
            )
            self.make_cost_plots(
                in_data=[self.base, self.basebw_20, self.pm_nobw, self.pm_20],
                name="sa20",
            )

        if cowpi:
            self.cowpi_boxplot(
                in_data=[self.base, self.basebw_neg20, self.pm_nobw, self.pm_neg20],
                name="sa-20",
                demographics=True,
            )
            self.cowpi_boxplot(
                in_data=[self.base, self.basebw_20, self.pm_nobw, self.pm_20],
                name="sa20",
                demographics=True,
            )

        if map:
            fig, axes = plt.subplots(1, 2, sharey=True)

            print(self.basebw_neg20["cowpi"])

            # axes[0] = self.bg_map(
            #     axes[0],
            #     display_demo=None,
            #     node_data=pd.Series(diff_neg20_age),
            #     wn_nodes=True,
            #     label="Cost ($)",
            #     # lg_fmt=custom_format,
            # )
            # axes[0, 1] = self.bg_map(
            #     axes[0, 1],
            #     "perc_renter",
            #     pd.Series(diff_neg20_age),
            #     wn_nodes=True,
            #     label="% Renter",
            # )

    def income_plots(self):
        """
        Make plot of Clinton, NC income by BG and micropolis income
        """
        ax0 = plt.subplot()
        ax0 = self.clinton_bg_plot(ax0)

        income = self.base["income"].loc[:, "income"].groupby(level=0).median()
        print(income)
        dist_income = pd.concat(
            [income, pd.Series(self.ind_distances)], axis=1, keys=["Income", "Distance"]
        )
        model, x = self.linear_regression(dist_income, "Distance", "Income")

        ax0.plot(x, model.predict(x))
        ax0.set(
            xlabel="Minimum Industrial Distance (normalized)", ylabel="Median Income"
        )

        # format the y axis ticks to have a dollar sign and thousands commas
        fmt = "${x:,.0f}"
        tick = mtick.StrMethodFormatter(fmt)
        ax0.yaxis.set_major_formatter(tick)

        ax0.legend(["Clinton, NC Data", "Clinton, NC Regression", "Micropolis"])
        plt.savefig(
            self.pub_loc + "income_bg-micropolis." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        """ Income map """
        loc = "Output Files/1_Distance Based Income/30_basebw_di/0/"
        comp_list = ["income"]
        data = ut.read_data(loc, comp_list)

        income = data["income"].loc[:, "income"].groupby(level=0).mean()

        ax1 = plt.subplot()
        ax1 = wntr.graphics.plot_network(
            self.wn,
            node_attribute=income,
            node_size=5,
            node_range=[0, 200000],
            node_colorbar_label="Income ($)",
            ax=ax1,
        )
        plt.gcf().set_size_inches(4, 3.5)
        plt.savefig(
            self.pub_loc + "income_map." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

    def make_single_plots(self, file, days, twa_plot=True):
        """Set the warmup period"""
        x_len = days * 24

        nodes_w_demand = [
            name
            for name, node in self.wn.junctions()
            if node.demand_timeseries_list[0].base_value > 0
        ]

        """ Make SEIR plot without error """
        loc = "Output Files/" + file + "/"
        comp_list = self.comp_list + [
            "bw_cost",
            "tw_cost",
            "bw_demand",
            "tw_demand",
            "income",
            "drink",
            "cook",
            "hygiene",
        ]
        # comp_list.remove('agent_loc')
        data = ut.read_data(loc, comp_list)
        print(data["wfh"].sum(axis=1))
        print(data["cov_ff"].sum(axis=1))
        data["tot_cost"] = data["tw_cost"] + data["bw_cost"]
        data["tot_demand"] = data["tw_demand"] + data["bw_demand"]
        print(data["tw_demand"].sum(axis=1) / 3.875 / len(data["tw_demand"].columns))
        print(
            (data["demand"][nodes_w_demand].sum(axis=1) * 3600).sum()
            / 3.875
            / 1000000
            / days
        )
        # print(data['age'].loc[15559200, self.res_nodes].notna().sum())
        # for i, val in data['age'].items():
        #     if 'TN' in i:
        #         print(f"{i}: {val.iloc[-1] / 3600}")
        # for i in data['age'].loc[:, 'TN49']:
        #     print(i/3600)
        # for i in data['demand'].loc[:, 'TN49']:
        #     print(i)
        households = len(data["income"])
        leg_text = ["S", "E", "I", "R", "wfh"]
        ax = plt.subplot()
        x_values = np.array([x for x in np.arange(0, days, days / x_len)])
        self.make_avg_plot(
            ax,
            data["seir_data"].iloc[-x_len:] * 100,
            None,
            leg_text,
            x_values,
            "Time (days)",
            "Percent Population",
            show_labels=True,
            sd_plot=False,
        )
        plt.savefig(
            loc + "seir" + "." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        """ Make demand plot """
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
        tanks = ["749", "750", "751", "752"]
        data["demand"] = data["demand"]

        """ make demand plots for the tanks """
        # data['demand'].loc[:, tanks].plot()
        # plt.savefig(loc + 'demand_tanks.png', bbox_inches='tight')
        # plt.close()

        data["demand"].loc[:, "58"].plot()
        plt.savefig(loc + "demand_58.png", bbox_inches="tight")
        plt.close()

        demand = data["demand"][nodes_w_demand].sum(axis=1)
        x_values = np.array([x for x in np.arange(0, days, days / x_len)])
        plt.plot(x_values, demand.iloc[-x_len:])
        plt.savefig(
            loc + "aggregate_demand" + "." + self.format,
            format=self.format,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        plt.close()

        # cols = ['Residential', 'Commercial', 'Industrial']
        # demand_res = data['demand'].loc[:, self.res_nodes].sum(axis=1)
        # demand_ind = data['demand'].loc[:, self.ind_nodes].sum(axis=1)
        # demand_com = data['demand'].loc[:, self.com_nodes].sum(axis=1)
        # demand = pd.concat([demand_res.rolling(24).mean(),
        #                     demand_ind.rolling(24).mean(),
        #                     demand_com.rolling(24).mean()],
        #                    axis=1, keys=cols)
        # x_values = np.array([
        #     x for x in np.arange(0, days, days / x_len)
        # ])
        # ax = plt.subplot()
        # self.make_avg_plot(
        #     ax, demand.iloc[-x_len:], sd=None, cols=cols, x_values=x_values,
        #     sd_plot=False
        # )
        # plt.savefig(loc + 'mean_res_demand' + '.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        """ Make age plots """
        age = data["age"][nodes_w_demand].mean(axis=1)
        age_all = data["age"][nodes_w_demand] / 3600
        print(age_all)
        age_all.loc[:, "58"].plot()
        plt.savefig(loc + "age_58.png", bbox_inches="tight")
        plt.close()
        # print(data['age'].loc[8470800, self.com_nodes].sort_values() / 3600)
        # print(data['age'].loc[8470800, self.res_nodes].sort_values() / 3600)
        plt.plot(x_values, age.iloc[-x_len:] / 3600)
        plt.savefig(
            loc + "age" + "." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        # res_age_pm = data['age'][self.res_nodes].mean(axis=1)
        # com_age_pm = data['age'][self.com_nodes].mean(axis=1)
        # ind_age_pm = data['age'][self.ind_nodes].mean(axis=1)

        # # make input data and sd
        # pm_age = pd.concat([res_age_pm.rolling(24).mean(),
        #                     com_age_pm.rolling(24).mean(),
        #                     ind_age_pm.rolling(24).mean()],
        #                    axis=1, keys=cols)
        # pm_age_sd = pd.concat([res_age_pm.rolling(24).std(),
        #                        com_age_pm.rolling(24).std(),
        #                        ind_age_pm.rolling(24).std()],
        #                       axis=1, keys=cols)
        # ax = plt.subplot()
        # self.make_avg_plot(
        #     ax, pm_age.iloc[-x_len:] / 3600, pm_age_sd[-x_len:] / 3600, cols,
        #     x_values, 'Time (days)', 'Water Age (hr)', show_labels=True,
        #     sd_plot=True
        # )

        # plt.savefig(loc + '_sector_age.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        # plt.show()

        """ Plot the income by node """
        # fig, axes = plt.subplots(1, 2)
        # ind_distances, ind_closest = self.calc_industry_distance(self.wn)
        # dist_values = [v for k, v in ind_distances.items() if k in data['income'].index]
        # income = data['income'].loc[:, 'income'].groupby(level=0).mean()
        # print(len(dist_values))
        # print(len(income))

        # axes[0] = self.scatter_plot(dist_values, income, axes[0])

        # axes[1] = wntr.graphics.plot_network(
        #     self.wn,
        #     node_attribute=income,
        #     node_size=5, node_range=[0, 200000], node_colorbar_label='Income ($)',
        #     ax=axes[1]
        # )
        # plt.gcf().set_size_inches(7, 3.5)
        # plt.savefig(loc + 'income_map.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        # wntr.graphics.plot_network(self.wn, node_attribute='elevation')
        # plt.show()

        # print(data['tot_cost'].iloc[-1, :].groupby(level=0).mean())
        # ax = wntr.graphics.plot_network(
        #     self.wn,
        #     node_attribute=data['tot_cost'].iloc[-1, :].groupby(level=0).mean(),
        #     node_size=5
        # )
        # plt.savefig(loc + 'tot_cost_map.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()
        print(data["age"].iloc[-1, :])

        ax = wntr.graphics.plot_network(
            self.wn,
            node_attribute=data["age"].iloc[-1, :] / 3600,
            node_size=5,
            node_colorbar_label="Water Age (hr)",
            node_range=[0, 336],
        )
        plt.savefig(
            loc + "age_map." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        """ Make maps with water age and bg demographics """
        data["income"].set_index(keys="id", inplace=True)
        # cowpi = pd.concat(
        #     [data['income'],
        #      data['tot_cost'].iloc[-1, :] / (data['income'] * self.days / 365)],
        #     axis=1, keys=['Income', 'COWPI']
        # )
        cowpi = pd.concat(
            [
                data["tot_cost"].iloc[-1, :]
                / data["income"].loc[:, "income"]
                * self.days
                / 365
                * 100,
                data["income"].loc[:, "income"],
            ],
            axis=1,
            keys=["cowpi", "income"],
        )

        for i, row in cowpi.iterrows():
            if row["cowpi"] < 0:
                print(row)

        cowpi_grouped = cowpi.groupby(level=0).mean()
        ax = plt.subplot()
        # self.bg_map(ax, 'median_income', data['age'].iloc[-1, :] / 3600)
        self.bg_map(ax, "median_income", cowpi_grouped.loc[:, "cowpi"] > 4.6)
        plt.gcf().set_size_inches(7, 3.5)
        plt.savefig(
            loc + "income-x-age." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        ax = plt.subplot()
        self.bg_map(ax)
        plt.savefig(
            loc + "clinton-bg." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        cowpi_low20 = cowpi.loc[:, "income"].quantile(0.20)

        def f(x):
            if x["income"] > cowpi_low20:
                return 1
            elif x["income"] < cowpi_low20:
                return 0

        cowpi["level"] = cowpi.apply(f, axis=1)
        # print(cowpi)
        income_level = pd.concat(
            [
                cowpi[cowpi.loc[:, "level"] == 0]["cowpi"].reset_index(drop=True),
                cowpi[cowpi.loc[:, "level"] == 1]["cowpi"].reset_index(drop=True),
            ],
            axis=1,
            keys=["Lower 20%", "Upper 80%"],
        )

        ax = plt.subplot()
        income_level.plot(kind="box", sym="")
        plt.savefig(
            loc + "income_cowpi." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        # plot race boxplot
        print(data["demo"])
        race = pd.concat(
            [
                cowpi[data["demo"].loc["white", :]]
                .loc[:, "cowpi"]
                .reset_index(drop=True),
                cowpi[~data["demo"].loc["white", :]]
                .loc[:, "cowpi"]
                .reset_index(drop=True),
            ],
            axis=1,
            keys=["White", "POC"],
        )
        # print(race)
        for i, row in race.iterrows():
            print(row)

        ax = plt.subplot()
        race.plot(kind="box", sym="")
        plt.savefig(
            loc + "race_cowpi." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        # plot hispanic boxplot
        hispanic = pd.concat(
            [
                cowpi[~data["demo"].loc["hispanic", :]]["cowpi"].reset_index(drop=True),
                cowpi[data["demo"].loc["hispanic", :]]["cowpi"].reset_index(drop=True),
            ],
            axis=1,
            keys=["Non-Hispanic", "Hispanic"],
        )

        ax = plt.subplot()
        race.plot(kind="box", sym="")
        plt.savefig(
            loc + "race_cowpi." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        # plot renters boxplot
        hispanic = pd.concat(
            [
                cowpi[data["demo"].loc["renter", :]]["cowpi"].reset_index(drop=True),
                cowpi[~data["demo"].loc["renter", :]]["cowpi"].reset_index(drop=True),
            ],
            axis=1,
            keys=["Renter", "Non-renter"],
        )

        ax = plt.subplot()
        hispanic.plot(kind="box", sym="")
        plt.savefig(
            loc + "renter_cowpi." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        """ Plot the number of residences """
        node_buildings = pd.read_pickle("buildings.pkl")
        counts = node_buildings.value_counts(["wdn_node", "type"]).unstack()
        perc_counts = counts.divide(counts.sum(axis=1) / 100, axis=0)
        print(perc_counts)

        ax = wntr.graphics.plot_network(
            self.wn,
            node_attribute=perc_counts["res"],
            node_size=5,
            node_colorbar_label="% Residential",
        )
        plt.savefig(
            loc + "perc_res_map." + self.format, format=self.format, bbox_inches="tight"
        )
        plt.close()

        """ Plot of TWA parameters """
        if twa_plot:
            warmup = data["bw_cost"].index[-1] - x_len
            data["tot_cost"] = data["bw_cost"] + data["tw_cost"]
            twa = pd.concat(
                [
                    data["drink"].sum(axis=1),
                    data["cook"].sum(axis=1),
                    data["hygiene"].sum(axis=1),
                ],
                axis=1,
                keys=["Drink", "Cook", "Hygiene"],
            )

            ax = plt.subplot()
            self.make_avg_plot(
                ax,
                twa / households * 100,
                None,
                ["Drink", "Cook", "Hygiene"],
                (twa.index / 24) - 30,
                "Time (days)",
                "Percent of Households",
                show_labels=True,
                sd_plot=False,
            )

            plt.savefig(
                loc + "twa." + self.format, format=self.format, bbox_inches="tight"
            )
            plt.close()

            """ Heatmap and map of costs """
            # convert the annual income to an income that is specific to timeframe
            data["income"] = data["income"]
            self.make_heatmap(
                data["tot_cost"].T,
                "Time (weeks)",
                "Household",
                loc + "tot_cost_heatmap",
                0.01,
            )

            cols = ["Tap Water", "Bottled Water", "Total"]
            cost = pd.concat(
                [
                    data["tw_cost"].mean(axis=1),
                    data["bw_cost"].mean(axis=1),
                    data["tot_cost"].mean(axis=1),
                ],
                axis=1,
                keys=cols,
            )
            cost_max = pd.concat(
                [
                    data["tw_cost"].max(axis=1),
                    data["bw_cost"].max(axis=1),
                    data["tot_cost"].max(axis=1),
                ],
                axis=1,
                keys=cols,
            )

            # average cost plot
            ax = plt.subplot()
            self.make_avg_plot(
                ax,
                cost,
                None,
                cols,
                (cost.index - warmup) / 24,
                "Time (Days)",
                "Mean Water Cost ($)",
                show_labels=True,
                sd_plot=False,
            )

            plt.savefig(
                loc + "mean_water_cost." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

            # max cost plot
            ax = plt.subplot()
            self.make_avg_plot(
                ax,
                cost_max,
                None,
                cols,
                (cost_max.index - warmup) / 24,
                "Time (Days)",
                "Maximum Water Cost ($)",
                show_labels=True,
                sd_plot=False,
            )

            plt.savefig(
                loc + "max_water_cost." + self.format,
                format=self.format,
                bbox_inches="tight",
                transparent=self.transparent,
            )
            plt.close()

        """ Equity metric costs """
        # metrics = pd.concat([data['traditional'],
        #                      data['burden']],
        #                     axis=1, keys=['Traditional', 'Burden'])

        # ax = plt.subplot()
        # self.make_avg_plot(
        #     ax, metrics * 100, None, ['Traditional', 'Burden'],
        #     (metrics.index - warmup) / 24,
        #     'Time (days)', '% of Income', show_labels=True, sd_plot=False
        # )

        # plt.savefig(loc + 'equity_metrics.' + self.format,
        #             format=self.format, bbox_inches='tight')
        # plt.close()

        """ % of income figure by income level """
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
