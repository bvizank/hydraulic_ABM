import geopandas
import matplotlib.pyplot as plt
import wntr
import pandas as pd


# import the water netowrk model
dir = 'Input Files/cities/clinton/'
wn = wntr.network.WaterNetworkModel(dir + 'clinton.inp')

# convert the wn to an object with various gdfs
wn_gis = wntr.network.to_gis(wn)
wn_gis.junctions = wn_gis.junctions.set_crs('epsg:4326')
wn_gis.pipes = wn_gis.pipes.set_crs('epsg:4326')

# import the block groups for sampson county
gdf = geopandas.read_file(dir + 'sampson_bg_clinton/tl_2023_37_bg.shp')
gdf['bg'] = gdf['TRACTCE'] + gdf['BLKGRPCE']
gdf.set_index('bg', inplace=True)
gdf.index = gdf.index.astype('int64')

# import demographic data using pandas
demo = pd.read_csv(dir + 'demographics_bg.csv')
demo.set_index('bg', inplace=True)

# filter the bgs for clinton
bg = [
    '970802',
    '970600',
    '970801',
    '970702',
    '970701'
]
gdf = gdf[gdf['TRACTCE'].isin(bg)]

gdf = gdf.join(demo)
print(gdf)

# apply the bg crs to the wn data
wn_gis.junctions.to_crs(gdf.crs, inplace=True)
wn_gis.pipes.to_crs(gdf.crs, inplace=True)

# clip the bg layer to the extent of the wn layer
gdf = geopandas.clip(gdf, mask=wn_gis.junctions.total_bounds)

# plot the bg layer
base = gdf.plot(column='median_income', legend=True, zorder=1)

# add the junctions and pipes
wn_gis.pipes.plot(ax=base, linewidth=1, zorder=2)
wn_gis.junctions.plot(ax=base, marker='o', color='red', markersize=5, zorder=3);

base.set_axis_off()
plt.savefig('clinton_wn_x_bg.pdf', format='pdf', bbox_inches='tight')
