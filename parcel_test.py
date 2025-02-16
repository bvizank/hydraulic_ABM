import city_info as ci
import wntr

wn = wntr.network.WaterNetworkModel('Input Files/cities/clinton/clinton.inp')
city = 'clinton'
dir = 'Input Files/cities/clinton/'

# node_buildings = ci.make_building_list(wn, city, dir)

service_area = ci.get_water_utility_service_areas('clinton')
print(service_area)
osm_buildings = ci.get_osm_buildings_within_area(service_area)

print(osm_buildings)
