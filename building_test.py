import city_info as ci
import wntr
import os


dir = "Input Files/cities/clinton"
city = "clinton"
wn = wntr.network.WaterNetworkModel(os.path.join(dir, city + ".inp"))

buildings = ci.make_building_list(wn, city, dir)
