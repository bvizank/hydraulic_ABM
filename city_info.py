import requests
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import shape, mapping
# from shapely.geometry.polygon import Polygon
import wntr
import matplotlib.pyplot as plt
import data as dt
import warnings


warnings.filterwarnings("ignore")


def query_nconemap(map_name, id, params, call, url=None):
    """
    Fetch data from NCOneMap from the given map_name
    """
    if url is None:
        url = (
            "https://services.nconemap.gov/secure/rest/services/NC1Map_"
            + map_name
            + "/MapServer/"
            + str(id)
            + "/query?where=1%3D1"
        )

    method = getattr(requests, call)
    response = method(url, params=params)
    data = response.json()

    return data


def convert_geojson_gdf(data, filter, filter_key=None, filter_val=None):
    # Extract geometry and create GeoDataFrame
    features = data["features"]
    geometry = list()
    out_features = list()
    for feature in features:
        if filter:
            if feature["properties"][filter_key] == filter_val:
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

    # map_name = "Water_Sewer_2004"
    # params = {
    #     "outFields": "wasyname",
    #     "outSR": "4326",
    #     "f": "geojson",
    # }

    # data = query_nconemap(map_name, 4, params, "get")

    # features = data["features"]

    features = gpd.read_file(
        os.path.join(
            "Input Files",
            "cities",
            "Type_A_Current_Public_Water_Systems_(2004).geojson",
        )
    )
    # geometry = list()
    # out_features = list()
    # for feature in features:
    #     if city_name.capitalize() in feature["properties"]["wasyname"]:
    #         geometry.append(shape(feature["geometry"]))
    #         out_features.append(feature)
    # df = gpd.GeoDataFrame(out_features, geometry=geometry, crs="EPSG:4326")

    # return df

    return features[
        features.loc[:, "wasyname"].str.contains(city_name.capitalize(), regex=False)
    ]


def get_osm_buildings_within_area(service_area_geom):
    """
    Fetches building footprints from OSM within a given geometry (service area).
    """
    buildings = ox.features_from_polygon(
        service_area_geom.geometry.iloc[0], tags={"building": True}
    )
    buildings = ox.project_gdf(buildings, to_crs="EPSG:4326")
    buildings_proj = ox.project_gdf(buildings)

    # calculate area of building footprint in meters and convert to feet
    buildings["area"] = buildings_proj.area * 10.764

    # plt.hist(buildings['area'], bins=100)
    # plt.show()

    return buildings


def assign_bg(data):
    """
    Assign block group id to each parcel.
    """
    dir = "Input Files/cities/clinton/"

    # import the block group geometry
    gdf = gpd.read_file(dir + "sampson_bg_clinton/tl_2023_37_bg.shp")
    gdf["bg"] = gdf["TRACTCE"] + gdf["BLKGRPCE"]
    gdf.set_index("bg", inplace=True)
    gdf.index = gdf.index.astype("int64")

    # filter the bgs for clinton
    bg = ["970802", "970600", "970801", "970702", "970701"]
    gdf = gdf[gdf["TRACTCE"].isin(bg)]

    # convert gdf crs to data
    gdf.to_crs(data.crs, inplace=True)

    # delete index_right from data
    data.drop("index_right", axis=1, inplace=True)

    # spatial join the parcels with the block groups
    data = data.sjoin(gdf, how="inner")
    data = data.rename(columns={"index_right": "bg"})
    # print(data.columns)

    return data


def get_parcel_data_within_area(service_area_geom):
    """
    Fetches parcel data from NCOneMap within the given geometry
    """
    envelope = mapping(service_area_geom.geometry.envelope.iloc[0])["coordinates"]

    pt1 = envelope[0][0]
    pt2 = envelope[0][2]
    map_name = "Parcels"
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

    data = query_nconemap(map_name, 1, params, "get")
    print(f"This envelope has {len(data['objectIds'])} object Ids.")

    df = gpd.GeoDataFrame()
    batch_size = 50
    for i in range(0, len(data["objectIds"]), batch_size):
        print(f"Getting data for objects {i}:{i+batch_size}")

        batch_ids = data["objectIds"][i : i + batch_size]
        batch_ids = ", ".join(map(str, batch_ids))
        ob_params = {
            "outFields": "parusedesc, parusedsc2",
            "outSR": "4326",
            "f": "geojson",
            "objectIds": batch_ids,
        }
        ob_data = query_nconemap(map_name, 1, ob_params, "get")
        features = ob_data["features"]
        geometry = list()
        out_features = list()
        for feature in features:
            # print(feature["geometry"])
            # print(shape(feature["geometry"]))
            geometry.append(shape(feature["geometry"]))
            out_features.append(feature["properties"])
        new_df = gpd.GeoDataFrame(
            out_features, geometry=geometry, crs=service_area_geom.crs
        )
        df = pd.concat([df, new_df])
        last_i = i

    ob_params = {
        "outFields": ["parusedesc", "parusedsc2"],
        "outSR": "4326",
        "f": "geojson",
        "objectIds": data["objectIds"][last_i : len(data["objectIds"])],
    }
    ob_data = query_nconemap(map_name, 1, ob_params, "get")
    features = ob_data["features"]
    geometry = list()
    out_features = list()
    for feature in features:
        geometry.append(shape(feature["geometry"]))
        out_features.append(feature["properties"])
    new_df = gpd.GeoDataFrame(
        out_features, geometry=geometry, crs=service_area_geom.crs
    )
    df = pd.concat([df, new_df])

    df = df.sjoin(service_area_geom, how="inner")

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
    """
    Print the statistics of the buidings in the given city
    """
    com_buildings = sum(buildings["type"] == "com")
    res_buildings = sum(buildings["type"] == "res")
    ind_buildings = sum(buildings["type"] == "ind")

    print(f"Number of commercial buildings found: {com_buildings}")
    print(f"Number of residential buildings found: {res_buildings}")
    print(f"Number of industrial buildings found: {ind_buildings}")


def type_helper(x):
    if x["ind"]:
        return "ind"
    if x["gro"]:
        return "gro"
    if x["caf"]:
        return "caf"
    if x["mfh"]:
        return "mfh"
    if x["com"]:
        return "com"
    if x["res"]:
        return "res"

    return ""


def buildings_by_type(buildings):
    """
    Characterize each building as commercial, residential, or industrial

    Parameters:
    -----------
        buildings : GeoDataFrame
            list of buildings to be sorted
    """
    buildings["parusedsc2"] = buildings.loc[:, "parusedsc2"].str.split("|")

    # filter out parcels without a secondary parcel description (parusedsc2)
    buildings = buildings[~buildings.loc[:, "parusedsc2"].isna()]
    print(buildings)

    com_target = set(list(dt.com_types.keys()))
    buildings["com"] = [
        not com_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    res_target = set(list(dt.res_types.keys()))
    buildings["res"] = [
        not res_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    ind_target = set(list(dt.ind_types.keys()))
    buildings["ind"] = [
        not ind_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    mfh_target = set(list(dt.mfh_types.keys()))
    buildings["mfh"] = [
        not mfh_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    caf_target = set(list(dt.caf_types.keys()))
    buildings["caf"] = [
        not caf_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    gro_target = set(list(dt.gro_types.keys()))
    buildings["gro"] = [
        not gro_target.isdisjoint(x) for x in buildings["parusedsc2"]
    ]

    buildings["type"] = buildings.apply(type_helper, axis=1)

    # buildings['type'] = np.where(
    #     buildings['building'].isin(dt.building_res), 'res',
    #     np.where(buildings['building'] == 'industrial', 'ind', 'com')
    # )
    # com_mask = (
    #     (
    #         (buildings["parusedesc"] == "COMMERCIAL")
    #         | (buildings["parusedesc"] == "EXEMPT")
    #         | (buildings["parusedesc"] == "AGRICULTURE")
    #     )
    #     & (buildings["parusedsc2"] != "")
    #     & ~(buildings["parusedsc2"].str.contains("CHURCH", regex=False, na=False))
    # )
    # print(sum(com_mask))
    # res_mask = (
    #     (buildings["parusedesc"] == "RESIDENTIAL")
    #     & ~(buildings["parusedsc2"].str.contains(list(dt.com_types.keys()), regex=False, na=False))
    # )
    # ind_mask = buildings["parusedsc2"].str.contains("INDUSTRIAL", regex=False)

    # buildings["type"] = np.where(com_mask, "com", np.where(res_mask, "res", ""))
    # buildings["type"] = np.where(ind_mask, "ind", buildings["type"])

    # buildings["type"] = buildings.apply(type_helper, axis=1)

    buildings = buildings[buildings["type"] != ""]

    # building_stats(buildings)
    buildings = assign_bg(buildings)

    return buildings


def get_building_areas(city_name, buildings, service_area):
    """
    Get the building areas from osmnx and match then with the corresponding
    polygon in the buildings gdf

    Parameters:
    -----------
        city_name : str
            name of the city, used to find water service boundary
        buildings : GeoDataFrame
            dataframe of parcel polygons, representing buildings in the city
    """
    # get the water service area

    # get the buildings areas from osmnx
    osm_buildings = get_osm_buildings_within_area(service_area)
    # extract only the "way" values
    osm_buildings = osm_buildings.loc["way", :]

    # need to remove type from buildings areas as it conclicts with the
    # parcel database
    osm_buildings = osm_buildings.loc[:, ("geometry", "area")]

    # make the geometry points instead of polygons
    osm_buildings.geometry = osm_buildings.geometry.centroid

    # sjoin the areas with the parcel polygons
    buildings_area = buildings.sjoin(osm_buildings, how="inner")
    buildings_area = buildings_area.groupby(level=0)["area"].sum()
    # print(buildings_area)
    buildings["area"] = buildings_area
    # print(buildings)

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
    gdB_nearest = (
        gdfB.iloc[idx].drop(columns="geometry").rename_axis("wdn_node").reset_index()
    )
    # print(gdB_nearest)
    gdf = pd.concat(
        [gdfA.reset_index(drop=True), gdB_nearest, pd.Series(dist, name="dist")], axis=1
    )

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

    buildings.geometry = buildings.geometry.centroid

    wn_nearest = ckdnearest(buildings, wn_gis.junctions)

    return wn_nearest


def make_building_list(wn, city, dir):
    # read parcel data from pickle or from NC OneMap
    if city + "_parcel.pkl" in os.listdir(dir):
        buildings = pd.read_pickle(os.path.join(dir, city + "_parcel.pkl"))
    else:
        buildings = buildings_in_city(city)
        buildings.to_pickle(os.path.join(dir, city + "_parcel.pkl"))

    # read water service area from pickle or from NC OneMap
    if city + "_service_area.pkl" in os.listdir(dir):
        service_area = pd.read_pickle(os.path.join(dir, city + "_service_area.pkl"))
    else:
        service_area = get_water_utility_service_areas(city)
        service_area.to_pickle(os.path.join(city + "_service_area.pkl"))

    buildings = buildings_by_type(buildings)
    buildings.to_csv(os.path.join(dir, "building_data.csv"))

    # get the area of each building
    data2keep = [
        "parusedesc",
        "parusedsc2",
        "geometry",
        "type",
        "bg",
        "Shape__Are",
        "area",
    ]
    buildings = get_building_areas(city, buildings, service_area)
    buildings = buildings.loc[:, data2keep]

    # remove parcels without a building
    buildings = buildings[~buildings["area"].isnull()]
    # print(buildings)

    return make_node_groups(buildings, wn)
