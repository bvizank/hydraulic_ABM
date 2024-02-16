    def make_contour(self, graph, data, data_type, fig_name,
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

# print(wfh['avg_seir_data'])
# index_vals = wfh['avg_seir_data'].index
# for i, item in enumerate(wfh['avg_seir_data'].wfh):
#     print(index_vals[i])
#     print(item)

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

# make_sector_plot(wn, no_wfh['avg_age']/3600, 'Age (hr)', no_wfh_comp_dir,
#                  'mean', 'mean_age', sd=no_wfh['var_age']/3600)
# make_sector_plot(wn, days_200['age']/3600, 'Age (hr)', day200_loc, 'mean',
#                  'mean_age', days=200)
# make_sector_plot(wn, days_400['age']/3600, 'Age (hr)', day400_loc, 'mean',
#                  'mean_age', days=400)
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'max', 'max_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, no_wfh['demand'], 'Demand (L)', wfh_loc, 'mean', 'mean_demand_aggregate',
#                  wfh['demand'], type='all')
# make_sector_plot(wn, pressure, 'Pressure (m)', pressure_wfh, type='all')

# closest_distances = calc_closest_node(wn)
# age_values = list()
# curr_age_values = wfh['age'].iloc[len(wfh['age'])-1]/3600
# for age in curr_age_values.items():
#     if age[0] in closest_distances.keys():
#         age_values.append(age[1])
# make_distance_plot(closest_distances.values(), age_values, 'Distance (m)',
#                    'Age (hr)', wfh_loc, 'age_closest_node')

