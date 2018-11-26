# This function is for estimating financial benefits of installing PV

import pandas as pd
import requests
from bill_calculator import calc
import os

# When user selects the area the PV profile is being generated (assuming it is a dataframe like the sample load profile)
# Also the load profile of user is being generated (with or without user's inputs)
# And the tariff is selected by user as described in bill_calculator

# ---------- Preparing inputs for testing -----------
# Load profile (kWh)
LP = requests.get('https://energytariff.herokuapp.com/LoadProfiles/Avg')
LP = LP.json()
df = pd.DataFrame.from_dict(LP, orient='columns')
df['TS'] = pd.to_datetime(df['TS'], unit='ms')
df = df[['TS', 'Load']]
load_profile = df.copy()

# PV Profile (kWh)
cwd = os.getcwd()
xl = pd.ExcelFile(os.path.join(cwd,"PVProfile.xlsx"))
pv_profile = xl.parse('testdata')
pv_profile['TS'] = pd.to_datetime(pv_profile.TS)

# Tariff
Tariff_name = " AGL TOU Residential"
all_tariffs = requests.get('https://energytariff.herokuapp.com/Tariffs/AllTariffs')
all_tariffs = all_tariffs.json()
for i in range(len(all_tariffs)):
    if all_tariffs[i]['Name'] == Tariff_name:
        selected_tariff = all_tariffs[i]

# ---------- Function -----------
def saving_est(load_profile, pv_profile, selected_tariff):
    # first the bill should be calculated using the load profile
    old_bill = calc(load_profile, selected_tariff)
    net_load = load_profile.copy()
    net_load.Load = net_load.Load-pv_profile.PV
    new_bill = calc(net_load, selected_tariff)

    Results = {'Est_tot_PV_output': pv_profile.PV.sum(), 'Est_net_output': -1 * new_bill['Annual_kWh_Exp']}
    # there could be other parameters to report
    return Results