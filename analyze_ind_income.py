import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm


'''
    first make list of industrial node locations
    list = {1: (-294,  59),
            2: (-243, 04)}

    test the distance from each residential node to each ind node
    by iterating around the list made above and calculating distance.

    figure out the minimum distance for each res node.
'''
# see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy
pd.options.mode.copy_on_write = True
# import the data the includes residential and industrial nodes and spatial data
col_names = ['lon', 'lat', 'val', 'struct', 'sec', 'group', 'bg', 'city']
data = pd.read_csv(
    'Input Files/clinton_data.csv',
    delimiter=',',
    names=col_names
)

# convert lat and lon to radians
data['lat'] = data['lat'] * np.pi / 180
data['lon'] = data['lon'] * np.pi / 180

# ind_nodes = data[(data['sec'] == 3) & (data['city'] == 1)]
# res_nodes = data[(data['sec'] == 1) & (data['city'] == 1)]
ind_nodes = data[(data['sec'] == 3)]
res_nodes = data[(data['sec'] == 1)]
res_nodes = res_nodes[res_nodes.loc[:, 'struct'] == 1]

# make a dict of industrial parcel locations
ind_loc = dict()
for i, row in ind_nodes.iterrows():
    ind_loc[row.name] = (row['lat'], row['lon'])

for i, key in enumerate(ind_loc):
    '''
    Using haversine formula to calculate distance

    More information found here:
    https://www.movable-type.co.uk/scripts/latlong.html
    '''
    res_nodes = res_nodes.assign(del_lat=res_nodes['lat'] - ind_loc[key][0])
    res_nodes = res_nodes.assign(del_lon=res_nodes['lon'] - ind_loc[key][1])
    res_nodes = res_nodes.assign(a=(
        np.sin(res_nodes['del_lat']/2) ** 2 +
        np.cos(res_nodes['lat']) * np.cos(ind_loc[key][0]) *
        np.sin(res_nodes['del_lon']/2) ** 2
    ))
    res_nodes.loc[:, str(key)] = (
        2 * np.arctan2(
                np.sqrt(res_nodes['a']), np.sqrt(1 - res_nodes['a'])
            ) *
        6371000
    )

col_names.extend(['del_lat', 'del_lon', 'a'])
res_nodes.loc[:, 'min'] = res_nodes.loc[:, ~res_nodes.columns.isin(col_names)].min(axis=1)

model = LinearRegression()
x = res_nodes[['min']]
y = res_nodes[['val']]
model.fit(x, y)
print(model.score(x, y))
print(model.coef_)
print(model.intercept_)

plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel('Industrial Distance')
plt.ylabel('Parcel Value')
plt.savefig('clinton_parcel_val.png', format='png', bbox_inches='tight')
plt.close()
results_ind = res_nodes.groupby(['group', 'bg']).mean().loc[:, 'min']
# print(res_nodes.groupby(['group', 'bg']).count())

# read in income distribution data
income = np.genfromtxt(
    'Input Files/clinton_income_data.csv',
    delimiter=',',
)

y = income[:, 1]

print(results_ind)
x = results_ind.values
x_norm = (
    (x - np.min(x)) / (np.max(x) - np.min(x))
)
y_norm = (y)
#     (y - np.min(y)) / (np.max(y) - np.min(y))
# )
x_with_intercept = np.empty(shape=(len(x_norm), 2), dtype=np.float64)
x_with_intercept[:, 0] = 1
x_with_intercept[:, 1] = x_norm

print(x_with_intercept)
print(y_norm)

model = sm.OLS(y_norm, x_with_intercept).fit()
print(model.fittedvalues)
print(model.fittedvalues-y_norm)
sse = np.sum((model.fittedvalues - y_norm)**2)
ssr = np.sum((model.fittedvalues - y_norm.mean())**2)
sst = ssr + sse
print(f'R2 = {ssr/sst}')
print(len(x_norm))
print(np.sqrt(sse/(len(x_norm) - 2)))
print(model.summary())

model = LinearRegression()
x = x_norm[:, np.newaxis]
y = y_norm[:, np.newaxis]
model.fit(x, y)
print(model.score(x, y))
print(model.coef_)
print(model.intercept_)

plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel('Normalized Industrial Distance')
plt.ylabel('Normalized Median BG Income')
plt.savefig('clinton_bg_income.png', format='png', bbox_inches='tight')
plt.close()

# block_groups = [
#     970600.1,
#     970600.2,
#     970600.3,
#     970600.4,
#     970701.1,
#     970701.2,
#     970702.1,
#     970702.2,
#     970702.3,
#     970702.4,
#     970801.1,
#     970801.2,
#     970802.1,
#     970802.2,
#     970802.3
# ]

# for i, key in enumerate(ind_loc):
#     '''
#     Using haversine formula to calculate distance

#     More information found here:
#     https://www.movable-type.co.uk/scripts/latlong.html
#     '''
#     res_nodes.loc[:, 'del_lat'] = res_nodes.loc[:, 'lat'] - ind_loc[key][0]
#     res_nodes.loc[:, 'del_lon'] = res_nodes.loc[:, 'lon'] - ind_loc[key][1]
#     res_nodes.loc[:, 'a'] = (
#         np.sin(res_nodes.loc[:, 'del_lat']/2) ** 2 +
#         np.cos(res_nodes.loc[:, 'lat']) * np.cos(ind_loc[key][0]) *
#         np.sin(res_nodes.loc[:, 'del_lon']/2) ** 2
#     )
#     res_nodes.loc[:, str(key)] = (
#         2 * np.arctan2(
#                 np.sqrt(res_nodes.loc[:, 'a']), np.sqrt(1 - res_nodes.loc[:, 'a'])
#             ) *
#         6371000
#     )

# col_names.extend(['del_lat', 'del_lon', 'a'])
# res_nodes.loc[:, 'min'] = res_nodes.loc[:, ~res_nodes.columns.isin(col_names)].min(axis=1)
# results_ind = res_nodes.groupby(['group', 'bg']).mean().loc[:, 'min']

# # read in income distribution data
# income = np.genfromtxt(
#     'Input Files/clinton_income_data.csv',
#     delimiter=',',
#     dtype=np.int64
# )

# block_groups = [
#     970600.1,
#     970600.2,
#     970600.3,
#     970600.4,
#     970701.1,
#     970701.2,
#     970702.1,
#     970702.2,
#     970702.3,
#     970702.4,
#     970801.1,
#     970801.2,
#     970802.1,
#     970802.2,
#     970802.3
# ]

# # make repeated data
# brackets = pd.Series([
#     5000,
#     12500,
#     17500,
#     22500,
#     27500,
#     32500,
#     37500,
#     42500,
#     47500,
#     55000,
#     67500,
#     87500,
#     112500,
#     137500,
#     175000,
#     200000
# ])

# results = list()
# for col in income.T:
#     dataset = np.array(
#         [brackets[i] for i, v in enumerate(col) for j in range(v)]
#     )

#     # KernelDensity requires 2D array
#     dataset = dataset[:, np.newaxis]

#     # fit KDE to the dataset
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(dataset)

#     results.append(np.mean(kde.sample(10000)))

# print(results_ind)
# print(results)
# combined = pd.DataFrame({
#     'dist': results_ind.to_numpy(),
#     'income': np.array(results)}
# )

# combined = combined.sort_values(by='income')
# model = LinearRegression()
# x = combined[['dist']]
# y = combined[['income']]
# model.fit(x, y)
# print(model.score(x, y))
# print(model.coef_)
# print(model.intercept_)

# plt.scatter(x, y)
# plt.plot(x, model.predict(x))
# plt.xlabel('Industrial Distance')
# plt.ylabel('Income')
# plt.show()

''' Calculate kernel density functions for income distributions '''
# data = {
#     17500: 96,
#     22500: 25,
#     32500: 12,
#     37500: 51,
#     47500: 20,
#     55000: 59,
#     67500: 40,
#     87500: 67,
#     112500: 36,
#     137500: 2,
#     175000: 18,
#     200000: 63
# }
# dataset = np.array([i for i in data for j in range(data[i])])

# # KernelDensity requires 2D array
# dataset = dataset[:, np.newaxis]

# # fit KDE to the dataset
# kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(dataset)

# # x-value range for plotting KDE
# x_range = np.linspace(dataset.min()-0.3, dataset.max()+0.3, num=600)

# # compute the log-likelihood of each sample
# log_density = kde.score_samples(x_range[:, np.newaxis])

# plt.figure(figsize=(10, 4))
# # put labels over datapoints
# for i, xi in enumerate(dataset):
#     plt.annotate(r'$x_{}$'.format(i+1),
#                  xy=[xi, 0.07],
#                  horizontalalignment='center',
#                  fontsize=18)

#     # draw KDE curve
# plt.plot(x_range, np.exp(log_density), 
#          color='gray', linewidth=2.5)

# # draw boxes representing datapoints
# plt.plot(dataset, np.zeros_like(dataset), 's', 
#          markersize=8, color='black')

# plt.xscale('log')

# plt.xlabel('$x$', fontsize=22)
# plt.ylabel('$f(x)$', fontsize=22, rotation='horizontal', labelpad=24)
# # plt.show()

# print(np.median(kde.sample(10000)))
