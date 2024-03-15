import pandas as pd
import numpy as np


''' Hourly sleep probabilities '''
# sleep_data = pd.read_excel(r'Input Files/sleep_data.xlsx')
# sleep = sleep_data['sleep_data'].tolist()
sleep_data = np.genfromtxt(r'Input Files/sleep_data.csv', delimiter=',')

''' Hourly Radio probabilities '''
# radio_data = pd.read_excel(r'Input Files/Radio_data.xlsx')
# radio = radio_data['radio_data'].tolist()
radio = np.genfromtxt(r'Input Files/Radio_data.csv', delimiter=',')

''' Hourly TV probabilities '''
# tv_data = pd.read_excel(r'Input Files/TV_data.xlsx')
# tv = tv_data['tv_data'].tolist()
tv = np.genfromtxt(r'Input Files/tv_data.csv', delimiter=',')

# media = {'sleep': sleep_distr,
#          'radio': radio_distr,
#          'tv': tv_distr}

''' Load agent parameters for BBN predictions '''
bbn_params = pd.read_csv(r'Input Files/all_bbn_data.csv')
bbn_param_list = bbn_params.columns.to_list()
bbn_params = np.array(bbn_params)

# bbn_params = np.genfromtxt(r'Input Files/all_bbn_data.csv', delimiter=',')

''' Load in the new residential patterns from Lakewood data '''
# wfh_patterns = pd.read_csv(
#     r'Input Files/res_patterns/normalized_res_patterns.csv'
# )
wfh_patterns = np.genfromtxt(
    r'Input Files/res_patterns/normalized_res_patterns.csv',
    delimiter=','
)

''' Currently reflects Lakewood data '''
income = {
    1: (1.83913478, 27649.41457),
    2: (3.929102832, 20760.71905),
    3: (7.346390908, 13674.60584),
    4: (3.519763579, 30890.71114),
    5: (11.97233919, 9709.881934),
    6: (12.76472073, 10398.50404)
}

# Set values for susceptibility based on age. From https://doi.org/10.1371/journal.pcbi.1009149
susDict = {1: [0.525, 0.001075, 0.000055, 0.00002],
           2: [0.6, 0.0072, 0.00036, 0.0001],
           3: [0.65, 0.0208, 0.00104, 0.00032],
           4: [0.7, 0.0343, 0.00216, 0.00098],
           5: [0.75, 0.07650, 0.00933, 0.00265],
           6: [0.8, 0.1328, 0.03639, 0.00766],
           7: [0.85, 0.20655, 0.08923, 0.02439],
           8: [0.9, 0.2457, 0.1742, 0.08292],
           9: [0.9, 0.2457, 0.1742, 0.1619]
}
