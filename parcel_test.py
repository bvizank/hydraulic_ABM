import city_info as ci
import wntr
import osmnx as ox


wn = wntr.network.WaterNetworkModel('Input Files/cities/clinton/clinton.inp')
city = 'clinton'
dir = 'Input Files/cities/clinton/'

service_area = ci.get_water_utility_service_areas('clinton')

osm_buildings = ci.get_osm_buildings_within_area(service_area)
print(osm_buildings.loc['way', :])

parcel_data = ci.get_parcel_data_within_area(service_area)
print(parcel_data)
