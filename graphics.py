import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import networkx as nx

output_loc = 'Output Files/2022-10-24_10-30_all_pb_ppe_current_results/'
data_file = output_loc + 'datasheet.xlsx'

seir = pd.read_excel(data_file, sheet_name='seir_data', index_col=1)
demand = pd.read_excel(data_file, sheet_name='demand', index_col=0)
pressure = pd.read_excel(data_file, sheet_name='pressure', index_col=0)
age = pd.read_excel(data_file, sheet_name='age', index_col=0)
agent = pd.read_excel(data_file, sheet_name='agent locations', index_col=0)

'''Import water network'''
inp_file = 'Input Files/MICROPOLIS_v1_orig_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()


def calc_difference(data_time_1, data_time_2):
    '''Function to take the difference between two time points '''
    return (data_time_2 - data_time_1)


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
    plt.savefig(fig_name)
    plt.close()


max_wfh = seir.wfh.loc[int(seir.wfh.idxmax())]
times = []
times = times + [seir.wfh.searchsorted(max_wfh/4)+12]
times = times + [seir.wfh.searchsorted(max_wfh/2)+12]
# print(seir.wfh.searchsorted(max_wfh/2))
times = times + [seir.wfh.searchsorted(max_wfh*3/4)+12]
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

for i, time in enumerate(times):
    if time != times[0]:
        if time >= len(demand):
            time = time - 1
        make_contour(G, demand_diff[i], 'demand', output_loc + 'demand_' + str(time), True,
                     'Demand [ML]', vmin=demand_stats[1], vmax=demand_stats[0])
        make_contour(G, pressure_diff[i], 'pressure', output_loc + 'pressure_' + str(time), True,
                     'Pressure [m]', vmin=pressure_stats[1], vmax=pressure_stats[0])
        make_contour(G, age_diff[i], 'age', output_loc + 'age_' + str(time), True,
                     'Age [sec]', vmin=age_stats[1], vmax=age_stats[0])
        make_contour(G, agent_diff[i], 'agent', output_loc + 'locations_' + str(time), True,
                     '# of Agents', vmin=agent_stats[1], vmax=agent_stats[0])

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
plt.axvline(x=times[2]*3600, color='black')
plt.axvline(x=times[3]*3600, color='black')
plt.xlabel('Time (days)')
plt.ylabel('Percent Population')
plt.legend()
plt.savefig(output_loc + '/' + 'seir_wfh.png')
plt.close()

#              'Pressure [m]', vmin=0, vmax=85)
# make_contour(G, pressure.iloc[12+(24*45)], 'pressure', output_loc + 'pressure_' + str(12+(24*45)), True,
#              'Pressure [m]', vmin=0, vmax=85)

# make_contour(G, demand.iloc[12], 'demand', output_loc + 'demand_' + str(12), True,
#              'Demand [ML]', vmin=0, vmax=0.02)
# make_contour(G, demand.iloc[12+(24*45)], 'demand', output_loc + 'demand_' + str(12+(24*45)), True,
#              'Demand [ML]', vmin=0, vmax=0.02)
