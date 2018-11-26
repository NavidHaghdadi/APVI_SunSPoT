# This function is for calculating the bill using the tariff and load profile
# load profile is half hourly usage data (kWh) for one year
# tariff data is an object (python dictionary) with input parameters and saved in the json file and can be retrieved
import pandas as pd
import requests
import numpy as np

# ---------- Preparing inputs for testing -----------------

# Tariffs can be downloaded from our API centre:
all_tariffs = requests.get('https://energytariff.herokuapp.com/Tariffs/AllTariffs')

# This is used to populate the dropdown list (use the "Name" key to show in the list). I will provide a list which maps
# the location (or postcode) to distributor so the list should be filtered to the user's distributor. Use the key
# 'Distributor' to filter. For example if user is in postcode 2020, the distributor will be Ausgrid and only tariffs
# with distributor = Ausgrid should be shown. At the moment all tariffs are in Ausgrid area but we will have more.
# User selects one of the tariffs.
# Icelab to generate a new json file which has only one tariff
# It can be done by the tariff code or the tariff name. For example lets assume user has selected tariff

Tariff_name = " AGL TOU Residential"
all_tariffs = requests.get('https://energytariff.herokuapp.com/Tariffs/AllTariffs')
all_tariffs = all_tariffs.json()
for i in range(len(all_tariffs)):
    if all_tariffs[i]['Name'] == Tariff_name:
        selected_tariff = all_tariffs[i]

# This 'selected tariff is the input for calculating the bill. User can also change the tariff parameters.
#  For example the daily parameter can be changed to a new number:
# print(selected_tariff['Parameters']['Daily']['Value'])
# selected_tariff['Parameters']['Daily']['Value'] = 0.95  # if user changes this value
# print(selected_tariff['Parameters']['Daily']['Value'])

# Another input is the load profile. This should be a one year half hourly load profile for a user.
# In the other function in load_profile_estimator.py, we will estimate the load profile of user based on the variety
# of inputs. Here for testing, we use the load profile from the below API

LP = requests.get('https://energytariff.herokuapp.com/LoadProfiles/Avg')
LP = LP.json()
df = pd.DataFrame.from_dict(LP, orient='columns')
df['TS'] = pd.to_datetime(df['TS'], unit='ms')
df = df[['TS', 'Load']]
main_load_profile = df.copy()


# ---------- Functions -----------------

def calc(main_load_profile, selected_tariff):

    main_tariff = selected_tariff.copy()

    if main_tariff['Type'] == 'Flat_rate':
        total_bill = fr_calc(main_load_profile, main_tariff)
    elif main_tariff['Type'] == 'TOU':
        total_bill = tou_calc(main_load_profile, main_tariff)
    else:
        total_bill = 'Error'

    return total_bill


def pre_processing_load(load_profile):

    # placeholder for a quick quality check function for load profile
    # make sure it is kwh
    # make sure it is one year
    # make sure it doesn't have missing value or changing the missing values to zero or to average
    # make sure the name is Load
    # time interval is half hour

    return load_profile


def fr_calc(load_profile, tariff):

    load_exp = load_profile['Load'].copy()
    load_exp[load_exp > 0] = 0
    load_imp = load_profile['Load'].copy()
    load_imp[load_imp < 0] = 0

    annual_kwh = load_imp.sum()
    annual_kwh_exp = load_exp.sum()

    num_of_days = len(load_profile["TS"].dt.normalize().unique())
    daily_charge = num_of_days*tariff['Parameters']['Daily']['Value']
    energy_charge = annual_kwh*tariff['Parameters']['Energy']['Value']*(1-tariff['Discount (%)']/100)
    fit_rebate = annual_kwh_exp*tariff['Parameters']['FiT']['Value']
    annual_bill = {'Annual_kWh': annual_kwh, 'Annual_kWh_Exp': annual_kwh_exp, 'Num_of_Days': num_of_days,
                   'Daily_Charge': daily_charge, 'Energy_Charge with discount': energy_charge, 'FiT_Rebate': fit_rebate,
                   'Total_Bill': energy_charge + daily_charge - fit_rebate}
    return annual_bill


def tou_calc(load_profile, tariff):
    load_exp = load_profile['Load'].copy()
    load_exp[load_exp > 0] = 0
    load_imp = load_profile['Load'].copy()
    load_imp[load_imp < 0] = 0
    load_profile['time_ind'] = 0

    annual_kwh = load_imp.sum()
    annual_kwh_exp = -1 * load_exp.sum()

    num_of_days = len(load_profile["TS"].dt.normalize().unique())
    daily_charge = num_of_days * tariff['Parameters']['Daily']['Value']
    tou_energy_charge = dict.fromkeys(tariff['Parameters']['Energy'])

    ti = 0
    all_tou_charge = 0
    for k, v in tariff['Parameters']['Energy'].items():
        this_part = tariff['Parameters']['Energy'][k].copy()
        ti += 1
        for k2, v2, in this_part['TimeIntervals'].items():
            start_hour = int(this_part['TimeIntervals'][k2][0][0:2])
            if start_hour == 24:
                start_hour = 0
            start_min = int(this_part['TimeIntervals'][k2][0][3:5])
            end_hour = int(this_part['TimeIntervals'][k2][1][0:2])
            if end_hour == 0:
                end_hour = 24
            end_min = int(this_part['TimeIntervals'][k2][1][3:5])
            if this_part['Weekday']:
                load_profile.time_ind = np.where(((load_profile['TS'].dt.weekday < 5) &
                                                  (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                  ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                   >= (60 * start_hour + start_min)) &
                                                  ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                   < (60 * end_hour + end_min))), ti, load_profile.time_ind)
            if this_part['Weekend']:
                load_profile.time_ind = np.where(((load_profile['TS'].dt.weekday >= 5) &
                                                  (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                  ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                  >= (60 * start_hour + start_min)) &
                                                  ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                  < (60 * end_hour + end_min))), ti, load_profile.time_ind)

        tou_energy_charge[k] = {'kWh': load_imp[load_profile.time_ind == ti].sum(),
                                'Charge': this_part['Value']*load_imp[load_profile.time_ind == ti].sum()}
        all_tou_charge = all_tou_charge + tou_energy_charge[k]['Charge']

    all_tou_charge = all_tou_charge*(1 - tariff['Discount (%)'] / 100)
    fit_rebate = annual_kwh_exp * tariff['Parameters']['FiT']['Value']
    annual_bill = {'Annual_kWh': annual_kwh, 'Annual_kWh_Exp': annual_kwh_exp, 'Num_of_Days': num_of_days,
                   'Daily_Charge': daily_charge, 'FiT_Rebate': fit_rebate, 'Energy_Charge': tou_energy_charge,
                   'Total_Energy_Charge with discount': all_tou_charge,
                   'Total_Bill': all_tou_charge + daily_charge - fit_rebate}
    return annual_bill


# ---------- Bill calculation -----------------
bill = calc(main_load_profile, selected_tariff)
print(bill)