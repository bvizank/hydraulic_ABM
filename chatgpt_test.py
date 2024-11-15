import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point, mapping
import wntr
import utils as ut
import data as dt
import matplotlib.pyplot as plt


def query_nconemap(map_name, id, params, call):
    """
    Fetch data from NCOneMap from the given map_name
    """
    url = (
        "https://services.nconemap.gov/secure/rest/services/NC1" +
        map_name +
        "/MapServer/" +
        str(id) +
        "/query?where=1%3D1"
    )
    
    method = getattr(requests, call)
    response = method(url, params=params)
    print(response.url)
    data = response.json()
    
    return data


def convert_geojson_gdf(data, filter, filter_key=None, filter_val=None):
    # Extract geometry and create GeoDataFrame
    features = data['features']
    geometry = list()
    out_features = list()
    for feature in features:
        if filter:
            if feature['properties'][filter_key] == filter_val:
                geometry.append(shape(feature["geometry"]))
                out_features.append(feature)
        else:
            geomtry.append()
    df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")

    return df


def get_water_utility_service_areas(city_name):
    """
    Fetches water utility service areas from NCOneMap dataset.
    """
    
    map = "Map_Water_Sewer_2004"    
    params = {
        "outFields": "wasyname",
        "outSR": "4326",
        "f": "geojson",
    }
    
    data = query_nconemap(map, 4, params, 'get')

    features = data['features']
    geometry = list()
    out_features = list()
    for feature in features:
        if feature['properties']['wasyname'] == city_name:
            geometry.append(shape(feature["geometry"]))
            out_features.append(feature)
    df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")
    
    return df


def get_osm_buildings_within_area(service_area_geom):
    """
    Fetches building footprints from OSM within a given geometry (service area).
    """
    buildings = ox.features_from_polygon(service_area_geom, tags={"building": True})
    buildings = ox.project_gdf(buildings, to_crs="EPSG:4326")
    buildings_proj = ox.project_gdf(buildings)
    
    # calculate area of building footprint in meters and convert to feet
    buildings['area'] = buildings_proj.area * 10.764
    
    plt.hist(buildings['area'], bins=100)
    plt.show()

    return buildings
    

def get_parcel_data_within_area(service_area_geom):
    """
    Fetches parcel data from NCOneMap within the given geometry
    """
    envelope = mapping(service_area_geom.envelope)['coordinates']
    
    pt1 = envelope[0][0]
    pt2 = envelope[0][2]
    map = "Map_Parcels"    
    params = {
        "inSR": "4326",
        "geometryType": "esriGeometryEnvelope",
        # "geometryType": "esriGeometryPolygon",
        "geometry": f"{pt1[0]}, {pt1[1]}, {pt2[0]}, {pt2[1]}",
        # "geometry": {service_area_geom},
        "outSR": "4326",
        "returnIdsOnly": "true",
        "f": "geojson",
    }
    
    data = query_nconemap(map, 0, params, 'get')
    print(f"This envelope has {len(data['objectIds'])} object Ids.")

    df = gpd.GeoDataFrame()
    for i in range(0, len(data['objectIds']), 50):
        print(f"Getting data for objects {i}:{i+50}")
        ob_params = {
            "outFields": str(["parusedesc", "parusedsc2"]),
            "outSR": "4326",
            "f": "geojson",
            "objectIds": str(data['objectIds'][i:i+50]),
        }
        ob_data = query_nconemap(map, 0, ob_params, 'post')
        features = ob_data['features']
        geometry = list()
        out_features = list()
        for feature in features:
            geometry.append(Point(feature['geometry']['coordinates']))
            out_features.append(feature['properties'])
        new_df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")
        print(new_df)
        df = pd.concat([df, new_df])
        last_i = i

    ob_params = {
        "outFields": ["parusedesc", "parusedsc2"],
        "outSR": "4326",
        "f": "geojson",
        "objectIds": data['objectIds'][last_i:len(data['objectIds'])]
    }
    ob_data = query_nconemap(map, 0, ob_params, 'get')
    df = pd.concat([df, convert_geojson_gdf(ob_data, filter=False)])

    print(df)
    
    return df


def buildings_in_city(city_name):
    """
    Find the buildings in the water service boundary.
    
    Parameters:
    -----------
        city_name : str
            name of the utility (wasyname) from NCOneMap
            https://www.nconemap.gov/datasets/nconemap::type-a-current-public-water-systems-2004/explore
    """
    # Step 1: Fetch water utility service areas
    service_areas = get_water_utility_service_areas(city_name)

    # Step 2: Filter for areas in Clinton, NC
    if service_areas.empty:
        print(f"No water utility service areas found for {city_name}.")
        return
    print(f"Found {len(service_areas)} service areas for {city_name}.")

    # Step 3: Get buildings within each service area
    for idx, area in service_areas.iterrows():
        # area_name = area["Name"]
        service_area_geom = area.geometry
        
        print(f"Fetching buildings within service area: {idx}")
        # buildings = get_osm_buildings_within_area(service_area_geom)
        buildings = get_parcel_data_within_area(service_area_geom)
        
        if buildings.empty:
            print(f"No buildings found in service area: {idx}")
        else:
            print(f"Found {len(buildings)} buildings in service area: {idx}")
            print(buildings.head())  # Print first few buildings as a sample
    
    return buildings
    
    
def buildings_by_type(buildings):
    """
    Characterize each building as commercial, residential, or industrial
    
    Parameters:
        buildings : GeoDataFrame
            list of buildings to be sorted
    """
    buildings['type'] = np.where(
        buildings['building'].isin(dt.building_res), 'res',
        np.where(buildings['building'] == 'industrial', 'ind', 'com')
    )
    
    return buildings


def ckdnearest(gdfA, gdfB):
    """
    Find the nearest point in gdfB for each point in gdfA
    
    Parameters:
    -----------
        gdfA : GeoDataFrame
            network to find nearest neighbors
        
        gdfB : GeoDataFrame
            network check for nearest neighbors
    """

    nA = np.array(list(gdfA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdfB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    print(gdfA)
    print(gdfB)
    gdB_nearest = gdfB.iloc[idx].drop(columns="geometry").reset_index()
    print(gdB_nearest)
    gdf = pd.concat(
        [
            gdfA.reset_index(),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf


def make_node_groups(buildings, wn):
    """
    Map the buildings to each node of the wn.
    
    Parameters:
    -----------
        buildings : GeoDataFrame
            list of buildings within the water service boundary
        
        wn : WaterNetworkModel
            water network of the target city
    """
    wn_gis = wntr.network.to_gis(wn, crs="EPSG:4326")
    
    wn_gis.junctions.plot()
    print(buildings['geometry'].centroid.index)
    
    wn_nearest = ckdnearest(buildings['geometry'].centroid, wn_gis.junctions)
    print(wn_nearest)


buildings = buildings_in_city('City of Clinton')
buildings.to_csv('building_data.csv')
buildings = buildings_by_type(buildings)
print(buildings)


inp_file = 'Input Files/clinton/Final_Clinton.inp'
_, _, wn = ut.init_wntr(inp_file)

make_node_groups(buildings, wn)
