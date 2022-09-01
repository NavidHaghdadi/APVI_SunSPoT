import numpy as np
import pandas as pd
import pickle
import requests
import os
import json
import warnings
import ast

user_inputs = json.load(open('user_inputs_Default.json'))
user_inputs['lat']= -34.88747495312189
user_inputs['long']= 150.5993496746492


# load_estimator(user_inputs, distributor, user_input_load_profile, first_pass):
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

if user_inputs['dryer_usage'].lower() == 'high':
    demog_info['DRYER_USAGE_CD'] = 3
elif user_inputs['dryer_usage'].lower() == 'medium':
    demog_info['DRYER_USAGE_CD'] = 2
elif user_inputs['dryer_usage'].lower() == 'low':
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
#fixed this issue by removing the Nan rows
# list_of_ws2 = list_of_ws[list_of_ws['sam_created']==True].copy()
# list_of_ws2.to_csv(r'C:\Codes\apvi-sunspot-script-server\ermy_locations.csv',index=False)
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

    # the format should be NEM12 format. We assume if the input has more than 3 columns it is
    # provided in the NEM12 format and follows the standard format:
    # https://www.aemo.com.au/-/media/Files/Electricity/NEM/Retail_and_Metering/Metering-Procedures/2018/MDFF-Specification-NEM12--NEM13-v106.pdf

    if user_input_load_profile.shape[1] > 3:
        NEM12_1 = user_input_load_profile.copy()

        Chunks = np.where((NEM12_1[NEM12_1.columns[0]] == 200)|(NEM12_1[NEM12_1.columns[0]] == 900))[0]
        NEM12_2 = pd.DataFrame()
        for i in range(0, len(Chunks)-1):
            if NEM12_1.iloc[Chunks[i], 4].startswith('E'):
                this_part = NEM12_1.iloc[Chunks[i] + 1: Chunks[i + 1], :].copy()
                this_part = this_part[this_part[this_part.columns[0]] == 300].copy()
                this_part2 = this_part.iloc[:, 2:50]
                this_part2 = this_part2.astype(float)
                if (this_part2[this_part2 < 0.01].count().sum() / this_part2.count().sum()) < 0.3: # assume for controlled load more 30% of data points are zero
                    NEM12_2 = NEM12_1.iloc[Chunks[i] + 1: Chunks[i + 1], :].copy()
                    NEM12_2.reset_index(inplace=True, drop=True)

        NEM12_2 = NEM12_2[NEM12_2[NEM12_2.columns[0]] == 300].copy()
        NEM12_2[NEM12_2.columns[1]] = NEM12_2[NEM12_2.columns[1]].astype(int).astype(str)

        Nem12 = NEM12_2.iloc[:, 1:50].melt(id_vars=[1], var_name="HH", value_name="kWh") # it was 49.. need to check if Dan manually changed it
        Nem12['HH'] = Nem12['HH']-1
        Nem12['kWh'] = Nem12['kWh'].astype(float)

        Nem12['Datetime'] = pd.to_datetime(Nem12[1], format='%Y%m%d') + pd.to_timedelta(Nem12['HH'] * 30, unit='m')
        Nem12.sort_values('Datetime', inplace=True)
        # Nem12_ = Nem12.groupby(['Datetime','HH']).sum().reset_index()
        Nem12.reset_index(inplace=True, drop=True)
        sample_load = Nem12[['Datetime', 'kWh']].copy()
    else:
        sample_load = user_input_load_profile.copy()
        sample_load[sample_load.columns[0]] = pd.to_datetime(sample_load[sample_load.columns[0]], format='%d/%m/%Y %H:%M',errors='coerce')
        sample_load = sample_load.dropna()


    sample_load.rename(columns={sample_load.columns[0]: 'READING_DATETIME', sample_load.columns[1]: 'kWh'},
                       inplace=True)
    # sample_load['READING_DATETIME'] = pd.to_datetime(sample_load['READING_DATETIME'], format='%d/%m/%Y %H:%M')
    sample_load['TS_N'] = sample_load['READING_DATETIME'].dt.normalize()

    # 1- Temperature data of the same period of time should be grabbed from NASA website
    start_date = sample_load['READING_DATETIME'].min().strftime('%Y%m%d')
    end_date = sample_load['READING_DATETIME'].max().strftime('%Y%m%d')

    if first_pass == True:
        LocLat = -34
        LocLong = 151
    else:
        LocLat = user_inputs['lat']
        LocLong = user_inputs['long']
    # first check CEEM API centre
    url2 = 'https://energytariff.herokuapp.com/weather/{}/{}/{}/{}'.format(start_date, end_date,
                                                                str(LocLat), str(LocLong))
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
        if 'peak' in user_inputs['previous_usage'][0]:
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
                                                > (60 * start_hour + start_min)) &
                                               ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                <= (60 * end_hour + end_min))), 'TOU'] = k1
                            else:
                                full_load.loc[(full_load['TS'].dt.weekday < 5) &
                                              (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                              (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                > (60 * start_hour + start_min)) |
                                               ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                <= (60 * end_hour + end_min))), 'TOU'] = k1
                        if this_part['Weekend']:
                            if start_hour <= end_hour:
                                full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                              (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                              (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                > (60 * start_hour + start_min)) &
                                               ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                <= (60 * end_hour + end_min))), 'TOU'] = k1
                            else:
                                full_load.loc[(full_load['TS'].dt.weekday >= 5) &
                                              (full_load['TS'].dt.month.isin(this_part['Month'])) &
                                              (((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                > (60 * start_hour + start_min)) |
                                               ((60 * full_load['TS'].dt.hour + full_load['TS'].dt.minute)
                                                <= (60 * end_hour + end_min))), 'TOU'] = k1

        else:
            full_load['TOU'] = 'Total'

        kb = 1
        usage_info = pd.DataFrame()
        for v3 in user_inputs['previous_usage']:
            usage_info_1 = pd.DataFrame({'TS': pd.date_range(start=v3['start_date'], end=v3['end_date'])})

            if 'peak' in v3:
                usage_info_1['Peak'] = v3['peak']
            else:
                usage_info_1['Peak'] = 'N/A'

            if 'offpeak' in v3:
                usage_info_1['OffPeak'] = v3['offpeak']
            else:
                usage_info_1['OffPeak'] = 'N/A'

            if 'shoulder' in v3:
                usage_info_1['Shoulder'] = v3['shoulder']
            else:
                usage_info_1['Shoulder'] = 'N/A'

            if 'total' in v3:
                usage_info_1['Total'] = v3['total']
            else:
                usage_info_1['Total'] = 'N/A'

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

        if 'Peak' in usage_info.sum():
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
load_profile_1 = load_profile.copy()

load_profile['TS'] = load_profile['TS'].apply(lambda dt: dt.replace(year=1990))
load_profile = load_profile.sort_values('TS')
load_profile['kW'] = 2*load_profile['kWh']
