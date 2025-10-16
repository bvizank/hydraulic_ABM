import city_info as ci
import wntr
import osmnx as ox
import pandas as pd
import geopandas as gpd


wn = wntr.network.WaterNetworkModel("Input Files/cities/clinton/clinton.inp")
city = "clinton"
dir = "Input Files/cities/clinton/"

parcel_data = ci.make_building_list(wn, city, dir)
print(parcel_data.query("type == 'com'"))
print(parcel_data.query("type == 'ind'"))
print(parcel_data.query("type == 'res'"))

print(
    parcel_data.query('type == "com"')
    .loc[:, "parusedsc2"]
    .str.split("|")
    .str.get(0)
    .unique()
)
