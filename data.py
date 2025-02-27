import pandas as pd
import numpy as np


""" Hourly sleep probabilities """
# sleep_data = pd.read_excel(r'Input Files/sleep_data.xlsx')
# sleep = sleep_data['sleep_data'].tolist()
sleep_data = np.genfromtxt(r"Input Files/sleep_data.csv", delimiter=",")

""" Hourly Radio probabilities """
# radio_data = pd.read_excel(r'Input Files/Radio_data.xlsx')
# radio = radio_data['radio_data'].tolist()
radio = np.genfromtxt(r"Input Files/Radio_data.csv", delimiter=",")

""" Hourly TV probabilities """
# tv_data = pd.read_excel(r'Input Files/TV_data.xlsx')
# tv = tv_data['tv_data'].tolist()
tv = np.genfromtxt(r"Input Files/tv_data.csv", delimiter=",")

# media = {'sleep': sleep_distr,
#          'radio': radio_distr,
#          'tv': tv_distr}

""" Load agent parameters for BBN predictions """
bbn_params = pd.read_csv(r"Input Files/all_bbn_data.csv")
bbn_param_list = bbn_params.columns.to_list()
bbn_params = np.array(bbn_params)

# bbn_params = np.genfromtxt(r'Input Files/all_bbn_data.csv', delimiter=',')

""" Load in the new residential patterns from Lakewood data """
# wfh_patterns = pd.read_csv(
#     r'Input Files/res_patterns/normalized_res_patterns.csv'
# )
wfh_patterns = np.genfromtxt(
    r"Input Files/res_patterns/normalized_res_patterns.csv", delimiter=","
)

""" Low income values by household size """
low_income = {1: 41100, 2: 46950, 3: 52800, 4: 58650, 5: 63350, 6: 68050}

""" Extremely low income values by household size """
ex_low_income = {1: 15400, 2: 20440, 3: 25820, 4: 31200, 5: 36580, 6: 41960}

""" Gamma parameters for income based on size for Clinton, NC """
size_income = {
    1: (0.22849, 96694.59),
    2: (0.063517253, 790761.5278),
    3: (0.667215195, 128128.0771),
    4: (0.016212716, 2261619.808),
    5: (0.063113445, 1094267.64),
    6: (1.303040116, 50842.64038),
}

""" Values for susceptibility based on age.
From https://doi.org/10.1371/journal.pcbi.1009149 """
susDict = {
    1: [0.525, 0.001075, 0.000055, 0.00002],
    2: [0.6, 0.0072, 0.00036, 0.0001],
    3: [0.65, 0.0208, 0.00104, 0.00032],
    4: [0.7, 0.0343, 0.00216, 0.00098],
    5: [0.75, 0.07650, 0.00933, 0.00265],
    6: [0.8, 0.1328, 0.03639, 0.00766],
    7: [0.85, 0.20655, 0.08923, 0.02439],
    8: [0.9, 0.2457, 0.1742, 0.08292],
    9: [0.9, 0.2457, 0.1742, 0.1619],
}

""" Data for data synthesis """
read_list = [
    "age",
    "bw_cost",
    "bw_demand",
    "cook",
    "cov_ff",
    "cov_pers",
    "demo",
    "demand",
    "dine",
    "drink",
    "flow",
    "groc",
    "hygiene",
    "income",
    "media",
    "ppe",
    "pressure",
    "seir_data",
    "tw_cost",
    "wfh",
]

avg_list = [
    "income",
    "bw_cost",
    "bw_demand",
    "tw_cost",
    "cook",
    "hygiene",
    "drink",
    "demo",
]


clinton_income = {
    0: 76,
    10000: 119,
    15000: 133,
    25000: 146,
    35000: 123,
    50000: 143,
    75000: 93,
    100000: 111,
    150000: 26,
    200000: 30,
}

com_water_coef = {
    "stores": 0.0979,
    "shopping": 0.0960,
    "offices": 0.1289,
    "medical": 0.1562,
    "restaurant": 0.7417,
    "fast_food": 0.6369,
    "finance": 0.3705,
    "auto": 0.1238,
    "hotel": 0.2286,
    "other_com": 0.0981,
    "gen_com": 0.1304,
    "manufacturing": 0.0545,
    "warehouse": 0.0335,
    "storage": 0.1520,
    "other_ind": 0.1196,
    "gen_ind": 0.0496,
    "school": 0.0684,
    "gen_inst": 0.0781,
    "other_inst": 0.1053,
}

res_types = {
    "RANCH": com_water_coef["other_com"],
    "DOUBLE WIDE MOHO": com_water_coef["other_com"],
    "CONTEMPORARY": com_water_coef["gen_com"],
    "CONVENTIONAL": com_water_coef["gen_com"],
}

mfh_types = {
    "WALK-UP APARTMENT": com_water_coef["gen_inst"],
    "GARDEN APARTMENT": com_water_coef["gen_inst"],
    "MULTI-FAMILY": com_water_coef["gen_inst"],
    "APARTMENT ELEVATOR": com_water_coef["gen_inst"],
}

com_types = {
    "GENERAL": com_water_coef["gen_com"],
    "TYPICAL OFFICE": com_water_coef["offices"],
    # "STORAGE": com_water_coef["storage"],
    "MEDICAL": com_water_coef["medical"],
    "LAUNDROMAT": com_water_coef["gen_com"],
    "WAREHOUSE MINI STOR": com_water_coef["warehouse"],
    "FIRE STATION": com_water_coef["other_inst"],
    "BANK": com_water_coef["finance"],
    "SPECIAL RETAIL": com_water_coef["stores"],
    "RETAIL STORE": com_water_coef["stores"],
    "SERVICE SHOP": com_water_coef["auto"],
    "IMPLEMENT SHED": com_water_coef["gen_com"],
    "HOSPITAL": com_water_coef["medical"],
    "BEAUTY SHOP": com_water_coef["shopping"],
    "WAREHOUSE STORAGE": com_water_coef["warehouse"],
    "GYMNASIUM": com_water_coef["gen_com"],
    "GOVERNMENT BUILDING": com_water_coef["gen_inst"],
    "FRATERNAL BUILDING": com_water_coef["gen_inst"],
    "SERVICE GARAGE": com_water_coef["auto"],
    "AUDITORIUM": com_water_coef["gen_com"],
    "SERVICE STATION": com_water_coef["auto"],
    "SCHOOL": com_water_coef["school"],
    "LIBRARY": com_water_coef["school"],
    "VETERINARY HOSPITAL": com_water_coef["medical"],
    "FEED MILL BUILDING": com_water_coef["other_ind"],
    "RURAL RETAIL": com_water_coef["stores"],
    "AUTOMOTIVE CENTER": com_water_coef["auto"],
    "CAR WASH": com_water_coef["gen_com"],
    # "SHED": com_water_coef["storage"],
    "MOTEL": com_water_coef["hotel"],
    "AUTOMOTIVE BUILDING": com_water_coef["auto"],
    "JAIL SCHEDULE": com_water_coef["gen_inst"],
    "STORAGE GARAGE": com_water_coef["storage"],
    # "PAVING ASPHALT": com_water_coef["gen_com"],
    # "LEAN TO OR ATTACHED SHED": com_water_coef["other_com"],
    "POLICE STATION": com_water_coef["gen_inst"],
    "WAREHOUSE DISTRIBUT": com_water_coef["warehouse"],
    "MORTUARY": com_water_coef["gen_com"],
    # "METAL BUILDING": com_water_coef["other_com"],
    # "UNFINISHED CARPORT": com_water_coef["other_com"],
    "LUMBER": com_water_coef["manufacturing"],
    "MISCELLANEOUS BUILDN": com_water_coef["other_com"],
    "BATH HOUSE": com_water_coef["other_com"],
    "POST OFFICE": com_water_coef["gen_inst"],
    # "PAVING CONCRETE": com_water_coef["other_com"],
    "CLUB HOUSE": com_water_coef["gen_inst"],
    "COMMERCIAL BLDG (SV)": com_water_coef["other_com"],
    "NURSING HOME": com_water_coef["medical"],
    "OFFICE": com_water_coef["offices"],
    "SHOP": com_water_coef["gen_com"],
    "THEATRE": com_water_coef["gen_com"],
    "CONVENIENCE MARKET": com_water_coef["stores"],
}

caf_types = {
    "RESTAURANT LOUNGE": com_water_coef["restaurant"],
    "FAST FOOD RESTAURAN": com_water_coef["fast_food"],
}

gro_types = {
    "MARKET": com_water_coef["shopping"],
    "COMMUNITY SHOPPING ": com_water_coef["shopping"],
}

ind_types = {
    "LIGHT INDUSTRIAL": com_water_coef["manufacturing"],
    "MEDIUM INDUSTRIAL": com_water_coef["manufacturing"],
    # "PACK HOUSE": com_water_coef["other_ind"],
    "BOTTLING PLANT": com_water_coef["manufacturing"],
}

par_types = {
    "res": res_types,
    "mfh": mfh_types,
    "com": com_types,
    "caf": caf_types,
    "gro": gro_types,
    "ind": ind_types,
}
