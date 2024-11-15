# import overpy


# api = overpy.Overpass()
# result = api.query("way(34.95573,-78.37715,35.04686,-78.27209);out;")

# print(result.ways[5])
# print(result.ways[5].tags)
# print(result.ways[5].attributes)

# for way in result.ways:
#     if way.id == 1149481940:
#         print(way.tags)
#         print(way.attributes)

# print(len(result.nodes))

import osmnx as ox
import requests


# Specify the location you're interested in
bbox = 34.95573, 35.04686, -78.27209, -78.37715

# get the water service boundaries for NC
url = "https://services.nconemap.gov/secure/rest/services/NC1Map_Water_Sewer_2004/MapServer/4/query?outFields=*&where=1%3D1&f=geojson"
payload = {
    
}

# Download building footprints for the specified location
# create network from that bounding box
# G = ox.graph_from_bbox(bbox=bbox, network_type="drive_service")
buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})

# Project the GeoDataFrame to a suitable coordinate system
buildings_proj = ox.project_gdf(buildings)

# Calculate the area of each building in square meters
buildings_proj['area'] = buildings_proj.area

print(buildings_proj.head)

# Print the total building area
print(buildings_proj['area'].sum())
