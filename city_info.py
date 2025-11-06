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


def get_bg_layer(tracts, ctfips, crs):
    """
    Fetch block group data from ESRI https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Block_Groups_v1/FeatureServer
    """
    url = (
        "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
        + "USA_Block_Groups_v1/FeatureServer/"
        + "0"
        + "/query?"
    )

    payload = {
        "where": "TRACT IN (" + ",".join(["'" + i + "'" for i in tracts]) + ") AND COUNTY=" + str(ctfips),
        "returnIdsOnly": "true",
        "f": "geojson",
    }

    response = requests.get(url, params=payload).json()

    object_ids = response["properties"]["objectIds"]

    payload = {
        "outFields": "TRACT,BLKGRP",
        "outSR": str(crs),
        "f": "geojson",
        "objectIds": ",".join(map(str, object_ids))
    }

    response = requests.get(url, params=payload).json()

    features = response["features"]
    geometry = list()
    out_features = list()
    for feature in features:
        geometry.append(shape(feature["geometry"]))
        out_features.append(feature["properties"])
    df = gpd.GeoDataFrame(
        out_features, geometry=geometry, crs="EPSG:"+str(crs)
    )

    return df


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


def query_fema(map_name, fips, layer_id=0, max_record_count=100, crs=3547):
    """
    Fetch data from FEMA's ESRI building data Feature Layer

    Parameters:
    -----------

    map_name (str): options are USA_Block_Groups_v1 or USA_Structures_View

    layer_name (str): 

    """

    # first get the object ids of the subset
    payload = {
        "where": "FIPS=" + str(fips),
        "returnIdsOnly": "true",
        "f": "geojson"
    }
    url = (
        "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
        + map_name
        + "/FeatureServer/"
        + str(layer_id)
        + "/query?"
    )

    response = requests.get(url, params=payload).json()

    object_ids = response["properties"]["objectIds"]

    print(f"The location you have chosen has {len(object_ids)} buildings.")

    # get objects in groups of the max record count
    if len(object_ids) > max_record_count:
        df = gpd.GeoDataFrame()
        for i in range(0, len(object_ids), max_record_count):
            print(f"Fetching data for objects {i} to {i + max_record_count}")
            curr_ids = object_ids[i:(i + max_record_count)]
            curr_ids = ",".join(map(str, curr_ids))
            payload = {
                "outFields": "OCC_CLS,PRIM_OCC,SEC_OCC,PROP_ADDR,SQFEET,CENSUSCODE",
                "outSR": str(crs),
                "f": "geojson",
                "objectIds": curr_ids
            }

            response = requests.get(url, params=payload)

            print(response)

            features = response.json()["features"]
            geometry = list()
            out_features = list()
            for feature in features:
                geometry.append(shape(feature["geometry"]))
                out_features.append(feature["properties"])
            new_df = gpd.GeoDataFrame(
                out_features, geometry=geometry, crs="EPSG:"+str(crs)
            )
            df = pd.concat([df, new_df])
            last_i = i

        # print(f"Fetching the final records: {last_i} to {len(object_ids)}")
        # curr_ids = object_ids[last_i:len(object_ids)]
        # curr_ids = ",".join(map(str, curr_ids))
        # payload = {
        #     "outFields": "OCC_CLS,PRIM_OCC,SEC_OCC,PROP_ADDR,SQFEET,CENSUSCODE",
        #     "outSR": str(crs),
        #     "f": "geojson",
        #     "objectIds": curr_ids
        # }

        # response = requests.get(url, params=payload)

        # features = response.json()["features"]
        # geometry = list()
        # out_features = list()
        # for feature in features:
        #     geometry.append(shape(feature["geometry"]))
        #     out_features.append(feature["properties"])
        # new_df = gpd.GeoDataFrame(
        #     out_features, geometry=geometry, crs="EPSG:"+str(crs)
        # )
        # df = pd.concat([df, new_df])

        # print(df)

    return df


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
    buildings = ox.projection.project_gdf(buildings, to_crs="EPSG:4326")
    buildings_proj = ox.projection.project_gdf(buildings)

    # calculate area of building footprint in meters and convert to feet
    buildings["area"] = buildings_proj.area * 10.764

    # plt.hist(buildings['area'], bins=100)
    # plt.show()

    return buildings


def assign_bg_api(buildings, bg_layer):
    """
    Assign the block group to each building using data from census api
    """
    buildings = buildings.sjoin(bg_layer, how="inner")

    print(buildings)

    buildings.loc[:, "bg"] = buildings.loc[:, "bg"] + buildings.loc[:, "BLKGRP"]
    buildings.loc[:, "bg"] = pd.to_numeric(buildings.loc[:, "bg"])

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


def buildings_by_area(fips):
    """
    Find the buildings in the county given the FIPS code
    """
    return query_fema("USA_Structures_View", fips, 0)


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


def fema_building_types(buildings, perc_caf=0.1, perc_gro=0.01):
    """
    Characterize the building types based on the FEMA building database
    """
    # if OCC_CLS is "Residential" and PRIM_OCC is "Single Family Dwelling"
    # then type is res
    buildings["res"] = (
        (buildings["OCC_CLS"] == "Residential") &
        (buildings["PRIM_OCC"].isin(["Single Family Dwelling", "Manufactured Home", "Unclassified"]))
    )

    buildings["mfh"] = (
        (buildings["OCC_CLS"] == "Residential") &
        (buildings["PRIM_OCC"].isin(["Institutional Dormitory", "Multi - Family Dwelling"]))
    )

    buildings["ind"] = (
        buildings["OCC_CLS"] == "Industrial"
    )

    buildings["com"] = (
        (buildings["OCC_CLS"].isin(
            [
                "Commercial", "Utility & Misc.", "Education", "Government",
                "Unclassified", "Agriculture"
            ]
        )) |
        (buildings["PRIM_OCC"] == "Temporary Lodging")
    )

    """
    Make a certain number of commercial buildings, cafe and grocery buildings
    because FEMA doesn't include either distinction in it's building classification
    system
    """
    # total number of buildings
    tot_buildings = len(buildings)
    com_buildings = np.where(buildings.com == True)

    caf_buildings = tot_buildings * perc_caf
    gro_buildings = tot_buildings * perc_gro

    buildings["caf"] = False
    buildings["caf"].iloc[np.random.choice(com_buildings[0], int(caf_buildings))] = True

    # convert PRIM_OCC to "Retail Trade" if cafe
    buildings.PRIM_OCC[buildings.caf == True] = "Retail Trade"

    buildings["gro"] = False
    buildings["gro"].iloc[np.random.choice(com_buildings[0], int(gro_buildings))] = True

    # convert PRIM_OCC to "Retail Trade" if cafe
    buildings.PRIM_OCC[buildings.gro == True] = "Retail Trade"

    # make all com buildings that were turned into caf or gro nodes, false
    buildings["com"].iloc[np.where(buildings.caf == True)] = False
    buildings["com"].iloc[np.where(buildings.gro == True)] = False

    buildings["type"] = buildings.apply(type_helper, axis=1)

    # make PRIM_OCC a list to conform to previous standards
    buildings["PRIM_OCC"] = buildings["PRIM_OCC"]

    # make PRIM_OCC a list to conform to previous standards
    buildings["PRIM_OCC"] = buildings["PRIM_OCC"].apply(lambda x: [x])

    return buildings


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
    # print(buildings)

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


def make_node_groups(buildings, wn, crs=4326):
    """
    Map the buildings to each node of the wn.

    Parameters:
    -----------
        buildings : GeoDataFrame
            list of buildings within the water service boundary

        wn : WaterNetworkModel
            water network of the target city
    """
    wn_gis = wntr.network.to_gis(wn, crs="EPSG:"+str(crs))
    # print(wn_gis.junctions.loc['1555', :])

    # wn_gis.junctions.plot()
    # print(buildings['geometry'].centroid.index)

    # buildings.geometry = buildings.geometry.centroid

    wn_nearest = ckdnearest(buildings, wn_gis.junctions)

    return wn_nearest


def make_building_list(wn, city, dir, crs=3547, fips=21159):
    # read parcel data from pickle or from NC OneMap
    # UNCOMMENT IF USING NC CITY
    # if city + "_parcel.pkl" in os.listdir(dir):
    #     buildings = pd.read_pickle(os.path.join(dir, city + "_parcel.pkl"))
    # else:
    #     buildings = buildings_in_city(city)
    #     buildings.to_pickle(os.path.join(dir, city + "_parcel.pkl"))

    # read water service area from pickle or from NC OneMap
    # UNCOMMENT IF USING NC CITY
    # if city + "_service_area.pkl" in os.listdir(dir):
    #     service_area = pd.read_pickle(os.path.join(dir, city + "_service_area.pkl"))
    # else:
    #     service_area = get_water_utility_service_areas(city)
    #     service_area.to_pickle(os.path.join(city + "_service_area.pkl"))

    # buildings = buildings_by_type(buildings)
    if city + "_buildings.pkl" in os.listdir(dir):
        buildings = pd.read_pickle(os.path.join(dir, city + "_buildings.pkl"))
    else:
        buildings = buildings_by_area(fips)
        buildings.to_pickle(os.path.join(dir, city + "_buildings.pkl"))

    # clean data from FEMA
    # convert polygons to points
    buildings.geometry = buildings.geometry.centroid

    # remove any buildings that do not have an address
    buildings = buildings[~buildings.PROP_ADDR.isnull()]

    # group by address to consolidate multi-building lots
    buildings = (
        buildings
        .groupby("PROP_ADDR")
        .agg(
            {
                "OCC_CLS": "first",
                "PRIM_OCC": "first",
                "SEC_OCC": "first",
                "SQFEET": "sum",
                "CENSUSCODE": "first",
                "geometry": "first"
            }
        )
    )

    print(buildings)

    # add the prison back
    buildings.loc["628 FEDERAL DRIVE", "OCC_CLS"] = "Residential"
    buildings.loc["628 FEDERAL DRIVE", "PRIM_OCC"] = "Institutional Dormitory"

    buildings = fema_building_types(buildings)

    # reinitialize geodataframe after groupby
    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs=crs)

    # rename columns to match rest of model
    buildings = buildings.rename(columns={"SQFEET": "area", "CENSUSCODE": "bg"})

    # update the bg column
    buildings.loc[:, "bg"] = buildings.loc[:, "bg"].str[5:]

    # assign bg to each building

    if city + "_bg_layer.pkl" in os.listdir(dir):
        bg_layer = pd.read_pickle(os.path.join(dir, city + "_bg_layer.pkl"))
    else:
        # get the tracts from building data
        tracts = buildings["bg"].unique()

        # get the block group gis layer
        bg_layer = get_bg_layer(tracts, 159, crs)

    print(bg_layer)

    buildings = assign_bg_api(buildings, bg_layer)

    # save building data to a csv
    buildings.to_csv(os.path.join(dir, city + "_building_data.csv"))

    print(buildings)
    print(buildings.crs)

    # get the area of each building
    # UNCOMMENT IF USING NC CITY
    # data2keep =
    #     "parusedesc",
    #     "parusedsc2",
    #     "geometry",
    #     "type",
    #     "bg",
    #     "Shape__Are",
    #     "area",
    # ]
    # buildings = get_building_areas(city, buildings, service_area)
    # buildings = buildings.loc[:, data2keep]

    # remove parcels without a building
    # UNCOMMENT IF USING NC CITY
    # buildings = buildings[~buildings["area"].isnull()]
    # print(buildings)

    return make_node_groups(buildings, wn, crs=crs), bg_layer
