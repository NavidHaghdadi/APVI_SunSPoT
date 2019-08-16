# This function is for estimating financial benefits of installing PV and battery
# VERY IMPORTANT: ALL KW AND KWH SHOULD BE CHECKED AGAIN
import numpy as np
import pandas as pd
import pickle
import requests
import os
import json
import warnings
import ast

warnings.filterwarnings("ignore")


# When user selects the area the PV profile is being generated
# (currently inputted as a csv file e.g. H_output.csv from Jessie)
# Also the load profile of user is being generated (with or without user's inputs)
# And the tariff is selected by user as described in bill_calculator
# a set of sample inputs for this function is provided in Testing_files.py
# This file contains four functions:
#  1- bill_calculator calculates the bill for any load profile and tariff
#  2- load_estimator estimates the load profile from demographic info and/or previous
#  usages and/or historical load profile
#  3- battery: estimates the net load for a load + PV + battery based on the tariff
#  - if the tariff is flat rate, it is maximising the self consumption. i.e. always storing the excess PV in battery
#  - if it is TOU, it is maximising the self consumption but also doesn't discharge the battery until peak time.
#  4- Main function to call these.

# -------------------- Bill Calculator function -----------


def bill_calculator(load_profile, tariff):
    # the input is load profile (kwh over half hourly) for one year and the tariff.
    # First input (column) of load_profile should be timestamp and second should be kwh
    # load_profile.columns={'TS','kWh'}

    def fr_calc(load_profile, tariff):
        load_exp = load_profile['kWh'].copy()
        load_exp[load_exp > 0] = 0
        load_imp = load_profile['kWh'].copy()
        load_imp[load_imp < 0] = 0
        annual_kwh = load_imp.sum()
        annual_kwh_exp = -1 * load_exp.sum()
        num_of_days = len(load_profile['TS'].dt.normalize().unique())
        daily_charge = num_of_days * tariff['Parameters']['Daily']['Value']
        energy_charge = annual_kwh * tariff['Parameters']['Energy']['Value'] * (1 - tariff['Discount (%)'] / 100)
        fit_rebate = annual_kwh_exp * tariff['Parameters']['FiT']['Value']
        annual_bill = {'Annual_kWh': annual_kwh, 'Annual_kWh_Exp': annual_kwh_exp, 'Num_of_Days': num_of_days,
                       'Daily_Charge': daily_charge, 'Energy_Charge with discount': energy_charge,
                       'FiT_Rebate': fit_rebate,
                       'Total_Bill': energy_charge + daily_charge - fit_rebate}
        return annual_bill

    def tou_calc(load_profile, tariff):
        load_exp = load_profile['kWh'].copy()
        load_exp[load_exp > 0] = 0
        load_imp = load_profile['kWh'].copy()
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
                    if start_hour <= end_hour:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday < 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile[
                                                             'TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) &
                                                          ((60 * load_profile['TS'].dt.hour + load_profile[
                                                              'TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                    else:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday < 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile[
                                                             'TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) |
                                                          ((60 * load_profile['TS'].dt.hour + load_profile[
                                                              'TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                if this_part['Weekend']:
                    if start_hour <= end_hour:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday >= 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile[
                                                             'TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) &
                                                          ((60 * load_profile['TS'].dt.hour + load_profile[
                                                              'TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                    else:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday >= 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile[
                                                             'TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) |
                                                          ((60 * load_profile['TS'].dt.hour + load_profile[
                                                              'TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
            tou_energy_charge[k] = {'kWh': load_imp[load_profile.time_ind == ti].sum(),
                                    'Charge': this_part['Value'] * load_imp[load_profile.time_ind == ti].sum()}
            all_tou_charge = all_tou_charge + tou_energy_charge[k]['Charge']

        all_tou_charge = all_tou_charge * (1 - tariff['Discount (%)'] / 100)
        fit_rebate = annual_kwh_exp * tariff['Parameters']['FiT']['Value']
        annual_bill = {'Annual_kWh': annual_kwh, 'Annual_kWh_Exp': annual_kwh_exp, 'Num_of_Days': num_of_days,
                       'Daily_Charge': daily_charge, 'FiT_Rebate': fit_rebate, 'Energy_Charge': tou_energy_charge,
                       'Total_Energy_Charge with discount': all_tou_charge,
                       'Total_Bill': all_tou_charge + daily_charge - fit_rebate}
        return annual_bill

    # Checking the type and run the appropriate function
    if tariff['Type'] == 'Flat_rate':
        total_bill = fr_calc(load_profile, tariff)
    elif tariff['Type'] == 'TOU':
        total_bill = tou_calc(load_profile, tariff)
    else:
        total_bill = 'Error'

    return total_bill


def load_estimator(user_inputs, distributor, user_input_load_profile):
    # This function is for estimating load profile from provided info. The inputs could be:
    #  - Historical load profile
    #  - Historical usage (user reads from bill)
    #  - Demographic information
    # It can be a combination of above as well.
    # The output is half hourly load profile for a typical year (adjusted for temperature)
    #
    demog_info = pd.DataFrame(0, index=[0], columns=['CONTROLLED_LOAD_CNT', 'DRYER_USAGE_CD', 'NUM_REFRIGERATORS',
                                                     'NUM_ROOMS_HEATED', 'HAS_GAS_HEATING', 'HAS_GAS_HOT_WATER',
                                                     'HAS_GAS_COOKING', 'HAS_POOLPUMP', 'NUM_OCCUPANTS', 'SemiDetached',
                                                     'SeparateHouse', 'Unit', 'Ducted', 'NoAirCon', 'OtherAirCon',
                                                     'SplitSystem'])

    demog_info['NUM_OCCUPANTS'] = user_inputs['family_size']

    if user_inputs['poolpump'] == 'yes':
        demog_info['HAS_POOLPUMP'] = 1

    if user_inputs['controlled_load'] == 'yes':
        demog_info['CONTROLLED_LOAD_CNT'] = 1

    if user_inputs['dryer_usage'] == 'high':
        demog_info['DRYER_USAGE_CD'] = 3
    elif user_inputs['dryer_usage'] == 'medium':
        demog_info['DRYER_USAGE_CD'] = 2
    elif user_inputs['dryer_usage'] == 'low':
        demog_info['DRYER_USAGE_CD'] = 1
    else:
        demog_info['DRYER_USAGE_CD'] = 0

    if user_inputs['AC_type'] == 'Split':
        demog_info['SplitSystem'] = 1
    elif user_inputs['AC_type'] == 'Ducted':
        demog_info['Ducted'] = 1
    elif user_inputs['AC_type'] == 'OtherAirCon':
        demog_info['OtherAirCon'] = 1
    else:
        demog_info['NoAirCon'] = 1

    if user_inputs['dwell'] == 'SemiDetached':
        demog_info['SemiDetached'] = 1
    elif user_inputs['dwell'] == 'Unit':
        demog_info['Unit'] = 1
    else:
        demog_info['SeparateHouse'] = 1

    if user_inputs['HAS_GAS_HEATING'] == 'yes':
        demog_info['HAS_GAS_HEATING'] = 1

    if user_inputs['HAS_GAS_HOT_WATER'] == 'yes':
        demog_info['HAS_GAS_HOT_WATER'] = 1

    if user_inputs['HAS_GAS_COOKING'] == 'yes':
        demog_info['HAS_GAS_COOKING'] = 1

    demog_info['NUM_ROOMS_HEATED'] = user_inputs['NUM_ROOMS_HEATED']

    demog_info['NUM_REFRIGERATORS'] = user_inputs['NUM_REFRIGERATORS']

    # Get the DegreeDays data for the particular location (It's currently in pickle file and it is 3mb.
    # We may want to move it to server and get only the data for desired location

    # Reading the whole RMY files and find the closest one to the user location
    weather_data_rmy_daily_all = pickle.load(open('AllWeather.pickle', 'rb'))
    list_of_ws = pd.read_csv('ermy_locations.csv')
    ind_ws = ((list_of_ws['Longitude'] - user_inputs['long']).abs() + (
            list_of_ws['Latitude'] - user_inputs['lat']).abs()).idxmin()
    weather_data_rmy_daily = weather_data_rmy_daily_all[
        weather_data_rmy_daily_all['loc'] == list_of_ws.loc[ind_ws, 'TLA'] + '_' + str(
            list_of_ws.loc[ind_ws, 'BOM_site'])].copy()

    # Estimate the temperature dependency
    # load the trained models for estimating coefficients
    cwd = os.getcwd()
    cdd_coef = pickle.load(open(os.path.join(cwd, 'Models\CDD_Coef.pickle'), 'rb'))
    hdd_coef = pickle.load(open(os.path.join(cwd, 'Models\HDD_Coef.pickle'), 'rb'))

    est_load_1 = pd.concat([pd.DataFrame({'Weekday': [1] * 24}), pd.concat([demog_info] * 24, ignore_index=True)],
                           axis=1)
    est_load_1['Hour'] = range(0, 24)
    est_load_2 = est_load_1.copy()
    est_load_2['Weekday'] = 0
    est_load_1 = pd.concat([est_load_1, est_load_2])

    est_load_1['cdd_coef'] = cdd_coef.predict(est_load_1.values)
    est_load_1['hdd_coef'] = hdd_coef.predict(est_load_1.iloc[:, 0:-1].values)
    est_load_1['cdd_coef'].clip(lower=0, inplace=True)
    est_load_1['hdd_coef'].clip(lower=0, inplace=True)

    # If load profile is provided: (date and location should be known)
    if user_inputs['load_profile_provided'] == 'yes':

        # Example: Load the example csv file and assume it is provided by user:
        # It should be kWh over half hourly periods. First column should be datetime and next one should be kWh

        sample_load = user_input_load_profile.copy()
        sample_load.rename(columns={sample_load.columns[0]: 'READING_DATETIME', sample_load.columns[1]: 'kWh'},
                           inplace=True)
        sample_load['READING_DATETIME'] = pd.to_datetime(sample_load['READING_DATETIME'], format='%d/%m/%Y %H:%M')
        sample_load['TS_N'] = sample_load['READING_DATETIME'].dt.normalize()

        # 1- Temperature data of the same period of time should be grabbed from NASA website
        start_date = sample_load['READING_DATETIME'].min().strftime('%Y%m%d')
        end_date = sample_load['READING_DATETIME'].max().strftime('%Y%m%d')

        # first check CEEM API centre
        url2 = 'https://energytariff.herokuapp.com/weather/{}/{}/{}/{}'.format(start_date, end_date,
                                                                    str(user_inputs['lat']), str(user_inputs['long']))
        temp_data2 = requests.get(url2)
        temp_data2 = temp_data2.json()
        if len(temp_data2) > 1:
            dh_temp_ws = pd.DataFrame.from_dict(ast.literal_eval(temp_data2))
        else:  # using NASA data
            url = 'https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?request=execute&identifier=' \
                  'SinglePoint&parameters=CDD18_3,HDD18_3&startDate={}&endDate={}&userCommunity=SSE&' \
                  'tempAverage=DAILY&outputList=JSON,ASCII&lat={}&lon={}&user=anonymous'.format(
                   start_date, end_date, str(user_inputs['lat']), str(user_inputs['long']))
            temp_data = requests.get(url)
            temp_data = temp_data.json()
            dh_temp_ws = pd.DataFrame.from_dict(temp_data['features'][0]['properties']['parameter'],
                                                orient='columns').reset_index()
            dh_temp_ws.rename(columns={'index': 'TS', 'CDD18_3': 'CDD', 'HDD18_3': 'HDD'}, inplace=True)

        # Adjusting the HDD and CDD
        dh_temp_ws['CDD'].where(dh_temp_ws['CDD'] > dh_temp_ws['HDD'], 0, inplace=True)
        dh_temp_ws['HDD'].where(dh_temp_ws['HDD'] > dh_temp_ws['CDD'], 0, inplace=True)

        dh_temp_ws['TS'] = pd.to_datetime(dh_temp_ws['TS'], format='%Y%m%d')
        # Adding HDH and CDH to load profile
        sample_load = sample_load.join(dh_temp_ws.set_index(['TS']), on=['TS_N'])
        sample_load['Hour'] = sample_load['READING_DATETIME'].dt.hour
        sample_load['Weekday'] = (sample_load['TS_N'].dt.dayofweek // 5 == 0).astype(int)

        sample_load = sample_load.join(est_load_1.set_index(['Hour', 'Weekday']), on=['Hour', 'Weekday'])
        sample_load['day'] = sample_load['TS_N'].dt.day
        sample_load['month'] = sample_load['TS_N'].dt.month

        # 3- Adjust the load profile to new temperature data (RMY)

        sample_load = sample_load.join(weather_data_rmy_daily.set_index(['day', 'month']), on=['day', 'month'],
                                       rsuffix='_New')
        sample_load['kWh_adj'] = sample_load['kWh'] + (sample_load['HDD_New'] - sample_load['HDD']) * sample_load[
            'hdd_coef'] + (sample_load['CDD_New'] - sample_load['CDD']) * sample_load['cdd_coef']

        sample_load['kWh_adj'] = sample_load['kWh_adj'].clip(lower=0)
        estimated_load = sample_load[['READING_DATETIME', 'kWh_adj']].rename(
            columns={'READING_DATETIME': 'TS', 'kWh_adj': 'kWh'})

        # Adding kW field
        time_res = round(
            ((estimated_load['TS'] - estimated_load['TS'].shift(periods=1)) / np.timedelta64(1, 'm')).median())

        if time_res == 30:
            estimated_load['kW'] = 2 * estimated_load['kWh']
        else:
            estimated_load['kW'] = estimated_load['kWh']
            estimated_load = estimated_load.set_index('TS')
            estimated_load = estimated_load.resample('30min').bfill().copy()
            estimated_load['kWh'] = estimated_load['kWh']/2  # as it is converted to half hourly
            estimated_load.reset_index(inplace=True)

    # Todo: check the energy etc to be similar

    else:

        # if the load profile is not imported. load the trained models for estimating underlying load
        cwd = os.getcwd()
        load_model = pickle.load(open(os.path.join(cwd, 'Models\TempCor.pickle'), 'rb'))

        # ['Hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring',
        #        'CONTROLLED_LOAD_CNT', 'DRYER_USAGE_CD', 'NUM_REFRIGERATORS',
        #        'NUM_ROOMS_HEATED', 'HAS_GAS_HEATING', 'HAS_GAS_HOT_WATER',
        #        'HAS_GAS_COOKING', 'HAS_POOLPUMP', 'NUM_OCCUPANTS', 'SemiDetached',
        #        'SeparateHouse', 'Unit', 'Ducted', 'NoAirCon', 'OtherAirCon',
        #        'SplitSystem']

        est_load_3 = pd.DataFrame(data={'Hour': range(0, 24)})
        est_load_3['Weekday'] = 1
        est_load_3['Summer'] = 1
        est_load_3['Winter'] = 0
        est_load_3['Fall'] = 0
        est_load_3['Spring'] = 0
        est_load_4 = pd.concat([demog_info] * 24).reset_index(drop=True)
        # pd.concat([est_load_4,est_load_3],axis=1)
        est_load_5 = pd.concat([est_load_3, est_load_4], axis=1)
        est_load_6 = est_load_5.copy()
        est_load_5['Weekday'] = 0
        est_load_5 = pd.concat([est_load_6, est_load_5], axis=0)

        est_load_6 = est_load_5.copy()

        est_load_6['Summer'] = 0
        est_load_6['Winter'] = 1
        est_load_5 = pd.concat([est_load_5, est_load_6], axis=0)

        est_load_6['Winter'] = 0
        est_load_6['Fall'] = 1
        est_load_5 = pd.concat([est_load_5, est_load_6], axis=0)

        est_load_6['Fall'] = 0
        est_load_6['Spring'] = 1
        est_load_5 = pd.concat([est_load_5, est_load_6], axis=0).reset_index(drop=True)

        est_load_5['kWh_pred'] = load_model.predict(est_load_5.values)

        est_load_5 = est_load_5.join(
            est_load_1[['Weekday', 'Hour', 'cdd_coef', 'hdd_coef']].set_index(['Hour', 'Weekday']),
            on=['Hour', 'Weekday'])

        full_load = pd.DataFrame(pd.date_range(start='1/1/1990T00:30', end='1/1/1991', freq='30min'), columns=['TS'])
        # full_load.drop(columns={'ghi','dni','dhi','WS','HDH','CDH'},inplace=True)
        # Note the load estimation training was based on half hourly kwh average.
        # So the load should be half hourly as well. Otherwise the value should be doubled.
        full_load['Summer'] = 1
        full_load['Winter'] = 1
        full_load['Fall'] = 1
        full_load['Spring'] = 1
        full_load['month'] = full_load['TS'].dt.month
        full_load['day'] = full_load['TS'].dt.day
        full_load['hour'] = full_load['TS'].dt.hour

        full_load['Summer'].where(full_load['month'].isin([12, 1, 2]), 0, inplace=True)
        full_load['Winter'].where(full_load['month'].isin([6, 7, 8]), 0, inplace=True)
        full_load['Fall'].where(full_load['month'].isin([3, 4, 5]), 0, inplace=True)
        full_load['Spring'].where(full_load['month'].isin([9, 10, 11]), 0, inplace=True)
        full_load['Weekday'] = (full_load['TS'].dt.dayofweek // 5 == 0).astype(int)

        full_load = full_load.join(weather_data_rmy_daily.iloc[:, 1:].set_index(['day', 'month']), on=['day', 'month'])

        full_load = full_load.join(est_load_5.set_index(['Hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring']),
                                   on=['hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring'])
        full_load['kWh_adj'] = full_load['kWh_pred'] + full_load['HDD'] * full_load['hdd_coef'] + full_load['CDD'] * \
                                full_load['cdd_coef']

        estimated_load = full_load[['TS', 'kWh_adj']].rename(columns={'kWh_adj': 'kWh'})

        # if Usage provided: all usage or peak/offpeak/shoulder
        if len(user_inputs['previous_usage']) > 0:

            if user_inputs['smart_meter'] == 'yes':
                with open('AllTOU.json') as f:
                    all_tou = json.load(f)
                for i in range(len(all_tou)):
                    if all_tou[i]['Distributor'] == distributor:
                        tou_times = all_tou[i]

                full_load['TOU'] = 'N/A'
                for k1, v1 in tou_times['TOU'].items():
                    for k, v in v1.items():
                        this_part = v.copy()
                        for k2, v2, in this_part['TimeIntervals'].items():
                            start_hour = int(v2[0][0:2])
                            if start_hour == 24:
                                start_hour = 0
                            start_min = int(v2[0][3:5])
                            end_hour = int(v2[1][0:2])
                            if end_hour == 0:
                                end_hour = 24
                            end_min = int(v2[1][3:5])
                            if this_part['Weekday']:
                                if start_hour <= end_hour:
                                    full_load.loc[(full_load['TS'].dt.weekday < 5) &
                                                  (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                                  (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    >= (60 * start_hour + start_min)) &
                                                   ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    < (60 * end_hour + end_min))), 'TOU'] = k1
                                else:
                                    full_load.loc[(full_load['TS'].dt.weekday < 5) &
                                                  (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                                  (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    >= (60 * start_hour + start_min)) |
                                                   ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    < (60 * end_hour + end_min))), 'TOU'] = k1
                            if this_part['Weekend']:
                                if start_hour <= end_hour:
                                    full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                                  (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                                  (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    >= (60 * start_hour + start_min)) &
                                                   ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    < (60 * end_hour + end_min))), 'TOU'] = k1
                                else:
                                    full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                                  (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                                  (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    >= (60 * start_hour + start_min)) |
                                                   ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                    < (60 * end_hour + end_min))), 'TOU'] = k1

            else:
                full_load['TOU'] = 'Total'

            kb = 1
            usage_info = pd.DataFrame()
            for v3 in user_inputs['previous_usage']:
                usage_info_1 = pd.DataFrame({'TS': pd.date_range(start=v3['start_date'], end=v3['end_date'])})
                usage_info_1['Peak'] = v3['peak']
                usage_info_1['OffPeak'] = v3['offpeak']
                usage_info_1['Shoulder'] = v3['shoulder']
                usage_info_1['Total'] = v3['total']
                usage_info_1['BillNo'] = kb
                kb += 1
                usage_info = pd.concat([usage_info, usage_info_1], axis=0)

            start_date = usage_info['TS'].min().strftime('%Y%m%d')
            end_date = usage_info['TS'].max().strftime('%Y%m%d')

            # first check CEEM API centre
            url2 = 'https://energytariff.herokuapp.com/weather/{}/{}/{}/{}'.format(start_date, end_date,
                                                                                   str(user_inputs['lat']),
                                                                                   str(user_inputs['long']))
            temp_data2 = requests.get(url2)
            temp_data2 = temp_data2.json()
            if len(temp_data2) > 1:
                dh_temp_ws = pd.DataFrame.from_dict(ast.literal_eval(temp_data2))
            else:  # using NASA data
                url = 'https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?request=execute&identifier=' \
                      'SinglePoint&parameters=CDD18_3,HDD18_3&startDate={}&endDate={}&userCommunity=SSE&' \
                      'tempAverage=DAILY&outputList=JSON,ASCII&lat={}&lon={}&user=anonymous'.format(
                       start_date, end_date, str(user_inputs['lat']), str(user_inputs['long']))
                temp_data = requests.get(url)
                temp_data = temp_data.json()
                dh_temp_ws = pd.DataFrame.from_dict(temp_data['features'][0]['properties']['parameter'],
                                                    orient='columns').reset_index()
                dh_temp_ws.rename(columns={'index': 'TS', 'CDD18_3': 'CDD', 'HDD18_3': 'HDD'}, inplace=True)

            # WeatherData_sel = WeatherData[WeatherData['TS'].dt.normalize().isin(usage_info['TS'])].copy()
            dh_temp_ws['CDD'].where(dh_temp_ws['CDD'] > dh_temp_ws['HDD'], 0, inplace=True)
            dh_temp_ws['HDD'].where(dh_temp_ws['HDD'] > dh_temp_ws['CDD'], 0, inplace=True)

            dh_temp_ws['TS'] = pd.to_datetime(dh_temp_ws['TS'], format='%Y%m%d')

            dh_temp_ws['month'] = dh_temp_ws['TS'].dt.month
            dh_temp_ws['day'] = dh_temp_ws['TS'].dt.day

            usage_info = usage_info.join(dh_temp_ws.set_index(['TS']), on=['TS'])

            if user_inputs['smart_meter'] == 'yes':
                usage_info_2 = pd.melt(usage_info, id_vars=['TS', 'BillNo', 'CDD', 'HDD', 'month', 'day'],
                                       value_vars=['Peak', 'OffPeak', 'Shoulder'], var_name='TOU',
                                       value_name='kWh_usage')
            else:
                usage_info_2 = pd.melt(usage_info, id_vars=['TS', 'BillNo', 'CDD', 'HDD', 'month', 'day'],
                                       value_vars=['Total'], var_name='TOU', value_name='kWh_usage')

            full_load_2 = full_load.join(usage_info_2.set_index(['month', 'day', 'TOU']), on=['month', 'day', 'TOU'],
                                         rsuffix='_new')

            full_load_2['kWh_adj_new'] = full_load_2['kWh_pred'] + full_load_2['HDD_new'] * full_load_2['hdd_coef'] + \
                                         full_load_2['CDD_new'] * full_load_2['cdd_coef']
            full_load_2['kWh_usage'] = full_load_2['kWh_usage'].astype(float)

            scaling = full_load_2[['BillNo', 'TOU', 'kWh_usage', 'kWh_adj_new']].groupby(['BillNo', 'TOU'],
                                                                                         as_index=False).agg(
                {'kWh_adj_new': 'sum', 'kWh_usage': 'mean'})
            scaling['Scale'] = scaling['kWh_usage'] / scaling['kWh_adj_new']
            scaling_2 = scaling.groupby(['TOU'], as_index=False).agg({'Scale': 'mean'})
            scaling_2['BillNo'] = 'nan'
            scaling_2['BillNo'] = scaling_2['BillNo'].astype(float)
            scaling_final = pd.concat([scaling_2, scaling[['TOU', 'BillNo', 'Scale']]], sort=False)
            full_load_2 = full_load_2.join(scaling_final.set_index(['TOU', 'BillNo']), on=['TOU', 'BillNo'])

            full_load_2['kWh_adj_final'] = full_load_2['kWh_adj'] * full_load_2['Scale']
            estimated_load = full_load_2[['TS', 'kWh_adj_final']].rename(
                columns={'kWh_adj_final': 'kWh'})

    load_profile = estimated_load.copy()

    load_profile['TS'] = load_profile['TS'].apply(lambda dt: dt.replace(year=1990))
    load_profile = load_profile.sort_values('TS')
    load_profile['kW'] = 2*load_profile['kWh']

    return load_profile


def battery(tariff, profiles_b, battery_kw, battery_kwh, distributor):
    # BattPow = 5  # 5KW
    # BattCap = 5  # 5kWh

    battery_eff = 0.85
    del_t = 2  # number of timestamps in each hour

    if tariff['Type'] == 'Flat_rate':

        profiles_fr = profiles_b.copy()
        profiles_fr['SOC'] = 0
        profiles_fr['ExcessPV'] = profiles_fr['PV'] - profiles_fr['Load']
        profiles_fr['ExcessLoad'] = profiles_fr['Load'] - profiles_fr['PV']
        profiles_fr['ExcessPV'].clip(lower=0, upper=battery_kw, inplace=True)
        profiles_fr['ExcessLoad'].clip(lower=0, upper=battery_kw, inplace=True)

        for i in range(1, len(profiles_fr)):
            profiles_fr.loc[i, 'SOC'] = max(0, min(battery_kwh, profiles_fr.loc[i - 1, 'SOC'] +
                                                   (battery_eff ** 0.5) * profiles_fr.loc[i, 'ExcessPV'] / del_t -
                                                   profiles_fr.loc[i, 'ExcessLoad'] / (
                                                           battery_eff ** 0.5) / del_t))
            profiles_fr.loc[i, 'ExcessCharge'] = profiles_fr.loc[i, 'SOC'] - profiles_fr.loc[i - 1, 'SOC']

        profiles_fr['ExcessCharge'] = profiles_fr['ExcessCharge'].apply(
            lambda x: x * (battery_eff ** 0.5) if x < 0 else x / (battery_eff ** 0.5))

        profiles_fr['NetLoad'] = profiles_fr['Load'] - profiles_fr['PV'] + profiles_fr['ExcessCharge'] * del_t
        profiles_b = profiles_fr.copy()

    elif tariff['Type'] == 'TOU':

        profiles_tou = profiles_b.copy()
        profiles_tou['SOC'] = 0
        profiles_tou['ExcessPV'] = profiles_tou['PV'] - profiles_tou['Load']
        profiles_tou['ExcessLoad'] = profiles_tou['Load'] - profiles_tou['PV']
        profiles_tou['ExcessPV'].clip(lower=0, upper=battery_kw, inplace=True)
        profiles_tou['ExcessLoad'].clip(lower=0, upper=battery_kw, inplace=True)

        full_load = profiles_tou.copy()

        with open('AllTOU.json') as f:
            all_tou = json.load(f)
        for i in range(len(all_tou)):
            if all_tou[i]['Distributor'] == distributor:
                tou_times = all_tou[i]

        full_load['TOU'] = 'N/A'
        for k1, v1 in tou_times['TOU'].items():
            for k, v in v1.items():
                this_part = v.copy()
                for k2, v2, in this_part['TimeIntervals'].items():
                    start_hour = int(v2[0][0:2])
                    if start_hour == 24:
                        start_hour = 0
                    start_min = int(v2[0][3:5])
                    end_hour = int(v2[1][0:2])
                    if end_hour == 0:
                        end_hour = 24
                    end_min = int(v2[1][3:5])
                    if this_part['Weekday']:
                        if start_hour <= end_hour:
                            full_load.loc[(full_load['TS'].dt.weekday < 5) &
                                          (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                          (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            >= (60 * start_hour + start_min)) &
                                           ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            < (60 * end_hour + end_min))), 'TOU'] = k1
                        else:
                            full_load.loc[(full_load['TS'].dt.weekday < 5) &
                                          (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                          (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            >= (60 * start_hour + start_min)) |
                                           ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            < (60 * end_hour + end_min))), 'TOU'] = k1
                    if this_part['Weekend']:
                        if start_hour <= end_hour:
                            full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                          (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                          (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            >= (60 * start_hour + start_min)) &
                                           ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            < (60 * end_hour + end_min))), 'TOU'] = k1
                        else:
                            full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                          (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                          (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            >= (60 * start_hour + start_min)) |
                                           ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                            < (60 * end_hour + end_min))), 'TOU'] = k1

        profiles_tou = full_load.copy()
        profiles_tou.loc[profiles_tou['TOU'] == 'OffPeak', 'ExcessLoad'] = 0
        profiles_tou.loc[profiles_tou['TOU'] == 'Shoulder', 'ExcessLoad'] = 0

        for i in range(1, len(profiles_tou)):
            profiles_tou.loc[i, 'SOC'] = max(0, min(battery_kwh, profiles_tou.loc[i - 1, 'SOC'] + (battery_eff ** 0.5) *
                                                    profiles_tou.loc[i, 'ExcessPV'] / del_t - profiles_tou.loc[
                                                        i, 'ExcessLoad'] / (battery_eff ** 0.5) / del_t))
            profiles_tou.loc[i, 'ExcessCharge'] = profiles_tou.loc[i, 'SOC'] - profiles_tou.loc[i - 1, 'SOC']

        profiles_tou['ExcessCharge'] = profiles_tou['ExcessCharge'].apply(
            lambda x: x * (battery_eff ** 0.5) if x < 0 else x / (battery_eff ** 0.5))

        profiles_tou['NetLoad'] = profiles_tou['Load'] - profiles_tou['PV'] + profiles_tou['ExcessCharge'] * del_t
        profiles_b = profiles_tou.copy()
    return profiles_b


# ---------- Main Function -----------


def saving_est(user_inputs, pv_profile, selected_tariff, pv_size_kw, battery_kw, battery_kwh, distributor,
               user_input_load_profile):
    # First the load profile is estimated using the user input data
    load_profile = load_estimator(user_inputs, distributor, user_input_load_profile)

    load_profile = load_profile.sort_values('TS')
    # and the bill should be calculated using the load profile
    old_bill = bill_calculator(load_profile, selected_tariff)

    # Now create a new load using the PV profile
    net_load = load_profile.copy()
    pv_profile['TS'] = pv_profile['TS'].apply(lambda dt: dt.replace(year=1990))
    pv_profile = pv_profile.sort_values('TS')
    pv_profile = pv_profile.set_index('TS')
    pv_profile = pv_profile.resample('30min').bfill().copy()
    pv_profile['PV'] = pv_profile['PV']/2 # converting kw to kwh
    pv_profile.reset_index(inplace=True)

    # Check and convert both load and PV profiles to half hourly
    net_load['kWh'] = net_load['kWh'] - pv_profile['PV']
    new_bill = bill_calculator(net_load, selected_tariff)
    pv_generation = pv_profile.PV.sum()
    total_saving_solar_only = old_bill['Total_Bill'] - new_bill['Total_Bill']
    saving_due_to_not_using_grid_solar_only = total_saving_solar_only - new_bill['FiT_Rebate']
    # Savings due to consuming PV energy instead of grid

    # Seasonal Pattern (1: Summer, 2: fall, 3: winter, 4: spring)
    load_seasonal_pattern = load_profile[['TS', 'kWh']].copy()
    load_seasonal_pattern['kWh'] = load_seasonal_pattern['kWh'] * 2  # convert to kW
    load_seasonal_pattern['hour'] = load_seasonal_pattern['TS'].dt.hour
    load_seasonal_pattern['season'] = (load_seasonal_pattern['TS'].dt.month % 12 + 3) // 3
    load_seasonal_pattern.drop(['TS'], inplace=True, axis=1)
    load_seasonal_pattern = load_seasonal_pattern.groupby(['season', 'hour'], as_index=False).mean()
    load_seasonal_pattern_json = load_seasonal_pattern.to_json(orient='values')

    pv_seasonal_pattern = pv_profile[['TS', 'PV']].copy()
    pv_seasonal_pattern['PV'] = pv_seasonal_pattern['PV'] * 2  # convert to kW
    pv_seasonal_pattern['hour'] = pv_seasonal_pattern['TS'].dt.hour
    pv_seasonal_pattern['season'] = (pv_seasonal_pattern['TS'].dt.month % 12 + 3) // 3
    pv_seasonal_pattern.drop(['TS'], inplace=True, axis=1)
    pv_seasonal_pattern = pv_seasonal_pattern.groupby(['season', 'hour'], as_index=False).mean()
    pv_seasonal_pattern_json = pv_seasonal_pattern.to_json(orient='values')

    # might remove LP from here and also the results
    load_profile_json = load_profile.to_json(orient='values')

    results = {'Annual_PV_Generation': pv_generation, 'Annual_PV_Generation_per_kW': pv_generation / pv_size_kw,
               'Est_Annual_PV_export_SolarOnly': new_bill['Annual_kWh_Exp'],
               'Est_Annual_PV_self_consumption_SolarOnly': pv_generation-new_bill['Annual_kWh_Exp'],
               'Saving_due_to_not_using_grid_SolarOnly': saving_due_to_not_using_grid_solar_only,
               'FiT_Payment_SolarOnly': new_bill['FiT_Rebate'],
               'Annual_Saving_SolarOnly': total_saving_solar_only,
               'Load_seasonal_pattern_kW': load_seasonal_pattern_json,
               'PV_seasonal_pattern_kW': pv_seasonal_pattern_json,
               "Old_Bill": old_bill['Total_Bill'], "New_Bill_SolarOnly": new_bill['Total_Bill'],
               'Load_Prof': load_profile_json}

    # LoadProf = pd.read_json(load_profile_json)
    # LoadProf[0]=pd.to_datetime(LoadProf[0],unit='ms')

    # Battery Analysis
    # We assume if the tariff is flat rate, the battery is maximising self consumption
    # If the tariff is TOU, the battery doesn't let you discharge at off-peak and shoulder times.

    if battery_kwh > 0:
        profiles_b = pv_profile.copy()
        profiles_b['PV'] = profiles_b['PV'] * 2  # convert to kW
        profiles_b['Load'] = load_profile['kWh'] * 2  # convert to kW
        battery_result = battery(selected_tariff, profiles_b, battery_kw, battery_kwh, distributor)
        net_load_batt = battery_result[['TS', 'NetLoad']].copy()
        net_load_batt.rename(columns={'NetLoad': 'kWh'}, inplace=True)
        net_load_batt['kWh'] = net_load_batt['kWh'] / 2  # convert to kWh

        pv_batt_seasonal_pattern = net_load_batt[['TS', 'kWh']].copy()
        pv_batt_seasonal_pattern['kWh'] = pv_batt_seasonal_pattern['kWh'] * 2  # convert to kW
        pv_batt_seasonal_pattern['hour'] = pv_batt_seasonal_pattern['TS'].dt.hour
        pv_batt_seasonal_pattern['season'] = (pv_batt_seasonal_pattern['TS'].dt.month % 12 + 3) // 3
        pv_batt_seasonal_pattern.drop(['TS'], inplace=True, axis=1)
        pv_batt_seasonal_pattern = pv_batt_seasonal_pattern.groupby(['season', 'hour'], as_index=False).mean()
        pv_batt_seasonal_pattern_json = pv_batt_seasonal_pattern.to_json(orient='values')

        new_bill_batt = bill_calculator(net_load_batt, selected_tariff)
        total_saving_sol_batt = old_bill['Total_Bill'] - new_bill_batt['Total_Bill']
        saving_due_to_not_using_grid_sol_batt = total_saving_sol_batt - new_bill_batt[
            'FiT_Rebate']  # Savings due to consuming PV energy instead of grid
        new_load_profile_json = battery_result.to_json(orient='values')
        results.update({'Est_Annual_PV_export_sol_batt': new_bill_batt['Annual_kWh_Exp'],
                        'Est_Annual_PV_self_consumption_sol_batt': pv_generation - new_bill_batt['Annual_kWh_Exp'],
                        'Saving_due_to_not_using_grid_sol_batt': saving_due_to_not_using_grid_sol_batt,
                        'FiT_Payment_sol_batt': new_bill_batt['FiT_Rebate'],
                        'New_Bill_sol_batt': new_bill_batt['Total_Bill'],
                        'pv_batt_seasonal_pattern_kW': pv_batt_seasonal_pattern_json,
                        'Annual_Saving_sol_batt': total_saving_sol_batt, 'NewProfile': new_load_profile_json})
    return results
