import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import networkx as nx

output_loc = 'Output Files/2022-10-17_08-15_wfh_dine_current_results/'
data_file = output_loc + 'datasheet.xlsx'

demand = pd.read_excel(data_file, sheet_name='demand', index_col=0)
pressure = pd.read_excel(data_file, sheet_name='pressure', index_col=0)
age = pd.read_excel(data_file, sheet_name='age', index_col=0)
agent = pd.read_excel(data_file, sheet_name='agent locations', index_col=0)

'''Import water network'''
inp_file = 'Input Files/MICROPOLIS_v1_orig_consumers.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
G = wn.get_graph()

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

times = [12, (24*18)+12, (24*36)+12, (24*54)+12, (24*72)+12]

for time in times:
    make_contour(G, demand.iloc[time], 'demand', output_loc + 'demand_' + str(time), True,
                  'Demand [ML]', vmin=0, vmax=0.04)
    make_contour(G, pressure.iloc[time], 'pressure', output_loc + 'pressure_' + str(time), True,
                'Pressure [m]', vmin=0, vmax=90)
    make_contour(G, age.iloc[time], 'age', output_loc + 'age_' + str(time), True,
                 'Age [sec]', vmin=0)
    make_contour(G, agent.iloc[time], 'agent', output_loc + 'locations_' + str(time), True,
                 '# of Agents', vmin=0, vmax=750)    

# make_contour(G, pressure.iloc[12], 'pressure', output_loc + 'pressure_' + str(12), True,
#              'Pressure [m]', vmin=0, vmax=85)
# make_contour(G, pressure.iloc[12+(24*45)], 'pressure', output_loc + 'pressure_' + str(12+(24*45)), True,
#              'Pressure [m]', vmin=0, vmax=85)

# make_contour(G, demand.iloc[12], 'demand', output_loc + 'demand_' + str(12), True,
#              'Demand [ML]', vmin=0, vmax=0.02)
# make_contour(G, demand.iloc[12+(24*45)], 'demand', output_loc + 'demand_' + str(12+(24*45)), True,
#              'Demand [ML]', vmin=0, vmax=0.02)
