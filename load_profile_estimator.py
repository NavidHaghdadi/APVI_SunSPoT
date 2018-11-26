# This function is for estimating the load profile based on the inputs provided by the user
# When user enters the information Icelab should generate a object with the below format and pass it to this function.
# the detail of the function is not yet finalised so the current file is only provided for input and output format.
#
import json
import io
import os
import requests
import pandas as pd
# Example of creating the user input file with format:
user_inputs = {'postcode': 2033, 'load_profile_provided': 'yes', 'family_size': '2', 'pool': 'yes',
               'controlled_load': 'yes', 'AC': 'yes', 'dwell': 'house', 'smart_meter': 'yes',
               'previous_usage': {'Bill 1': {'total': 'N/A', 'peak': '100', 'offpeak': '100', 'shoulder': '100',
                                  'start_date': '2018-02-02', 'end_date': '2018-05-01'},
                                  'Bill 2': {'total': 'N/A', 'peak': '120', 'offpeak': '110', 'shoulder': '50',
                                  'start_date': '2018-01-01', 'end_date': '2018-02-01'}}}

cwd = os.getcwd()
user_inputs_str = json.dumps(user_inputs)
# to_unicode = str
with io.open(os.path.join(cwd, 'user_inputs.json'), 'w', encoding='utf8') as outfile:
    outfile.write(str(user_inputs_str))


def estimator(user_inouts):
    # As mentioned the function for estimating the load profile is not final yet. So here a mock load file is used. We
    # may want to use weather data to adjust the load profile and create a typical load profile.
    LP = requests.get('https://energytariff.herokuapp.com/LoadProfiles/Avg')
    LP = LP.json()
    df = pd.DataFrame.from_dict(LP, orient='columns')
    df['TS'] = pd.to_datetime(df['TS'], unit='ms')
    df = df[['TS', 'Load']]
    est_load_profile = df.copy()

    return est_load_profile