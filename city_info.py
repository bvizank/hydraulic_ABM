import requests
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point, mapping
from shapely.geometry.polygon import Polygon
import wntr
import matplotlib.pyplot as plt


def query_nconemap(map_name, id, params, call, url=None):
    """
    Fetch data from NCOneMap from the given map_name
    """
    if url is None:
        url = (
            "https://services.nconemap.gov/secure/rest/services/NC1" +
            map_name +
            "/MapServer/" +
            str(id) +
            "/query?where=1%3D1"
        )

    method = getattr(requests, call)
    response = method(url, params=params)
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
            geometry.append()
    df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")

    return df


def get_water_utility_service_areas(city_name):
    """
    Fetches water utility service areas from NCOneMap dataset.
    """

    map_name = "Map_Water_Sewer_2004"
    params = {
        "outFields": "wasyname",
        "outSR": "4326",
        "f": "geojson",
    }

    data = query_nconemap(map_name, 4, params, 'get')

    features = data['features']
    geometry = list()
    out_features = list()
    for feature in features:
        if city_name.capitalize() in feature['properties']['wasyname']:
            geometry.append(shape(feature["geometry"]))
            out_features.append(feature)
    df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")

    return df


def get_osm_buildings_within_area(service_area_geom):
    """
    Fetches building footprints from OSM within a given geometry (service area).
    """
    buildings = ox.features_from_polygon(service_area_geom.geometry[0], tags={"building": True})
    buildings = ox.project_gdf(buildings, to_crs="EPSG:4326")
    buildings_proj = ox.project_gdf(buildings)

    # calculate area of building footprint in meters and convert to feet
    buildings['area'] = buildings_proj.area * 10.764

    # plt.hist(buildings['area'], bins=100)
    # plt.show()

    return buildings


def assign_bg(data):
    '''
    Assign block group id to each parcel.
    '''
    dir = 'Input Files/cities/clinton/'

    # import the block group geometry
    gdf = gpd.read_file(dir + 'sampson_bg_clinton/tl_2023_37_bg.shp')
    gdf['bg'] = gdf['TRACTCE'] + gdf['BLKGRPCE']
    gdf.set_index('bg', inplace=True)
    gdf.index = gdf.index.astype('int64')

    # filter the bgs for clinton
    bg = [
        '970802',
        '970600',
        '970801',
        '970702',
        '970701'
    ]
    gdf = gdf[gdf['TRACTCE'].isin(bg)]

    # convert gdf crs to data
    gdf.to_crs(data.crs, inplace=True)

    # delete index_right from data
    data.drop('index_right', axis=1, inplace=True)

    # spatial join the parcels with the block groups
    data = data.sjoin(gdf, how='inner')
    # print(data.columns)

    return data


def get_parcel_data_within_area(service_area_geom):
    """
    Fetches parcel data from NCOneMap within the given geometry
    """
    envelope = mapping(service_area_geom.geometry.envelope.iloc[0])['coordinates']

    pt1 = envelope[0][0]
    pt2 = envelope[0][2]
    map_name = "Map_Parcels"
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

    data = query_nconemap(map_name, 0, params, 'get')
    print(f"This envelope has {len(data['objectIds'])} object Ids.")

    df = gpd.GeoDataFrame()
    batch_size = 50
    for i in range(0, len(data['objectIds']), batch_size):
        print(f"Getting data for objects {i}:{i+batch_size}")

        batch_ids = data['objectIds'][i: i+batch_size]
        batch_ids = ', '.join(map(str, batch_ids))
        ob_params = {
            "outFields": "parusedesc, parusedsc2",
            "outSR": "4326",
            "f": "geojson",
            "objectIds": batch_ids,
        }
        ob_data = query_nconemap(map_name, 1, ob_params, 'get')
        features = ob_data['features']
        geometry = list()
        out_features = list()
        for feature in features:
            geometry.append(shape(feature['geometry']))
            out_features.append(feature['properties'])
        new_df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")
        df = pd.concat([df, new_df])
        last_i = i

    ob_params = {
        "outFields": ["parusedesc", "parusedsc2"],
        "outSR": "4326",
        "f": "geojson",
        "objectIds": data['objectIds'][last_i:len(data['objectIds'])]
    }
    ob_data = query_nconemap(map_name, 1, ob_params, 'get')
    features = ob_data['features']
    geometry = list()
    out_features = list()
    for feature in features:
        geometry.append(shape(feature['geometry']))
        out_features.append(feature['properties'])
    new_df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")
    df = pd.concat([df, new_df])
    print(df)

    df = df.sjoin(service_area_geom, how='inner')

    return df


def buildings_in_city(city_name):
    """
    Find the buildings in the water service boundary.

    Parameters:
    -----------
        city_name : str
            name of the city with a service area in NC OneMap
            https://www.nconemap.gov/datasets/nconemap::type-a-current-public-water-systems-2004/explore
    """
    # Step 1: Fetch water utility service areas
    service_area = get_water_utility_service_areas(city_name)

    # Step 2: Filter for areas in city
    if service_area.empty:
        print(f"No water utility service areas found for {city_name}.")
        return
    print(f"Found {len(service_area)} service areas for {city_name}.")

    # Step 3: Get buildings within each service area
    # for idx, area in service_areas.iterrows():
    # area_name = area["Name"]
    # print(f"Fetching buildings within service area: {idx}")
    # buildings = get_osm_buildings_within_area(service_area_geom)
    buildings = get_parcel_data_within_area(service_area)
    buildings = buildings.reset_index(drop=True)

    if buildings.empty:
        print("No buildings found in service area")
    else:
        print(f"Found {len(buildings)} buildings in service area")
        print(buildings.head())  # Print first few buildings as a sample

    return buildings


def building_stats(buildings):
    '''
    Print the statistics of the buidings in the given city
    '''
    com_buildings = sum(buildings['type'] == 'com')
    res_buildings = sum(buildings['type'] == 'res')
    ind_buildings = sum(buildings['type'] == 'ind')

    print(f"Number of commercial buildings found: {com_buildings}")
    print(f"Number of residential buildings found: {res_buildings}")
    print(f"Number of industrial buildings found: {ind_buildings}")


def buildings_by_type(buildings):
    """
    Characterize each building as commercial, residential, or industrial

    Parameters:
    -----------
        buildings : GeoDataFrame
            list of buildings to be sorted
    """
    # buildings['type'] = np.where(
    #     buildings['building'].isin(dt.building_res), 'res',
    #     np.where(buildings['building'] == 'industrial', 'ind', 'com')
    # )
    com_mask = (
        (buildings['parusedesc'] == 'COMMERCIAL') |
        (buildings['parusedesc'] == 'EXEMPT') |
        (buildings['parusedesc'] == 'AGRICULTURE')
    )
    # print(sum(com_mask))
    res_mask = (buildings['parusedesc'] == 'RESIDENTIAL')
    ind_mask = (buildings['parusedsc2'].str.contains('INDUSTRIAL', regex=False))

    buildings['type'] = np.where(
        com_mask, 'com',
        np.where(res_mask, 'res', '')
    )
    buildings['type'] = np.where(ind_mask, 'ind', buildings['type'])

    buildings = buildings[buildings['type'] != '']

    # building_stats(buildings)
    buildings = assign_bg(buildings)

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
    # print(gdfA)
    # print(gdfB)
    gdB_nearest = gdfB.iloc[idx].drop(columns="geometry").rename_axis('wdn_node').reset_index()
    # print(gdB_nearest)
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
    # print(wn_gis.junctions.loc['1555', :])

    # wn_gis.junctions.plot()
    # print(buildings['geometry'].centroid.index)

    wn_nearest = ckdnearest(buildings, wn_gis.junctions)

    return wn_nearest


def make_building_list(wn, city, dir):
    if city + '_parcel.pkl' in os.listdir(dir):
        buildings = pd.read_pickle(os.path.join(dir, city + '_parcel.pkl'))
    else:
        buildings = buildings_in_city(city)
        buildings.to_pickle(os.path.join(dir, city + '_parcel.pkl'))

    buildings = buildings_by_type(buildings)
    buildings.to_csv(os.path.join(dir, 'building_data.csv'))
    # print(buildings)

    return make_node_groups(buildings, wn)
