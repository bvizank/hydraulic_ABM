import city_info as ci
import wntr

wn = wntr.network.WaterNetworkModel('Input Files/cities/clinton/clinton.inp')
city = 'clinton'
dir = 'Input Files/cities/clinton/'

node_buildings = ci.make_building_list(wn, city, dir)
