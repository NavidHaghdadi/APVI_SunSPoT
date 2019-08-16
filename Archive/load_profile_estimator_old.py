# This function is for estimating the load profile based on the inputs provided by the user
# When user enters the information Icelab should generate a object with the below format and pass it to this function.
# the detail of the function is not yet finalised so the current file is only provided for input and output format.
#
import json
import os
import requests
import pandas as pd
import io

# ---------- Example of creating the user input file with format -----------

user_inputs = {'postcode': 2033, 'lat': -30, 'long': 150, 'load_profile_provided': 'yes', 'family_size': 2, 'poolpump': 'no',
               'controlled_load': 'no', 'AC_type': 'Split', 'dwell': 'SeparateHouse', 'smart_meter': 'yes', 'dryer_usage': 'medium',
                'HAS_GAS_HEATING': 'yes',  'HAS_GAS_HOT_WATER': 'yes', 'HAS_GAS_COOKING': 'yes','NUM_ROOMS_HEATED': 1, 'NUM_REFRIGERATORS': 1,
               'previous_usage': {'Bill 1': {'total': 'N/A', 'peak': 700, 'offpeak': 500, 'shoulder': 300,
                                  'start_date': '2018-02-02', 'end_date': '2018-05-01'},
                                  'Bill 2': {'total': 'N/A', 'peak': 300, 'offpeak': 200, 'shoulder': 100,
                                  'start_date': '2018-01-01', 'end_date': '2018-02-01'}}}


user_inputs_defaults = {'postcode': 2033, 'lat': -30, 'long': 150, 'load_profile_provided': 'yes', 'family_size': 2, 'poolpump': 'no',
               'controlled_load': 'no', 'AC_type': 'Split', 'dwell': 'SeparateHouse', 'smart_meter': 'yes', 'dryer_usage': 'medium',
                'HAS_GAS_HEATING': 'yes',  'HAS_GAS_HOT_WATER': 'yes', 'HAS_GAS_COOKING': 'yes','NUM_ROOMS_HEATED': 1, 'NUM_REFRIGERATORS': 1,
               'previous_usage':''}
# user_input_options:
# postcode: a valid postcode in Australia (perhaps match against a list..
# lat
# long
# load_profile_provided: yes / no
# family_size: 1, 2, 3, 4+
# poolpump: yes / no
# dryer_usage: high/medium/low/no
# controlled_load: yes / no
# AC_type: Split, Ducted, NoAirCon, OtherAirCon
# dwell: SeparateHouse, SemiDetached, Unit
# smart_meter: yes / no
# previous_usage : if clicked expand the first entry and if click + expand the next one and so on.

cwd = os.getcwd()
user_inputs_str = json.dumps(user_inputs)
with io.open(os.path.join(cwd, 'user_inputs_Example.json'), 'w', encoding='utf8') as outfile:
    outfile.write(str(user_inputs_str))

user_inputs_defaults_str = json.dumps(user_inputs_defaults)
with io.open(os.path.join(cwd, 'user_inputs_Default.json'), 'w', encoding='utf8') as outfile:
    outfile.write(str(user_inputs_defaults_str))

# ---------- Function -----------
def estimator(user_inouts):
    # As mentioned the function for estimating the load profile is not final yet. So here a mock load file is used. We
    # may want to use weather data to adjust the load profile and create a typical load profile.
    LP = requests.get('http://api.ceem.org.au/LoadProfiles/Avg')
    LP = LP.json()
    df = pd.DataFrame.from_dict(LP, orient='columns')
    df['TS'] = pd.to_datetime(df['TS'], unit='ms')
    df = df[['TS', 'Load']]
    est_load_profile = df.copy()

    return est_load_profile