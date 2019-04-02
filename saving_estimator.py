# This function is for estimating financial benefits of installing PV

import numpy as np
import pandas as pd
import pickle
import requests
import os
import json

# When user selects the area the PV profile is being generated (assuming it is a dataframe like the sample load profile)
# Also the load profile of user is being generated (with or without user's inputs)
# And the tariff is selected by user as described in bill_calculator
# a set of sample inputs for this function is provided in Testing_files.py

# ---------- Bill Calculator function -----------


def bill_calculator(load_profile, tariff):
    # the input is load profile (half hourly) for one year and the tariff.
    # First input should be timestamp and second should be kwh
    # load_profile.columns={'TS','kWh'}

    def fr_calc(load_profile, tariff):
        load_exp = load_profile['kWh'].copy()
        load_exp[load_exp > 0] = 0
        load_imp = load_profile['kWh'].copy()
        load_imp[load_imp < 0] = 0

        annual_kwh = load_imp.sum()
        annual_kwh_exp = -1 * load_exp.sum()

        num_of_days = len(load_profile['TS'].dt.normalize().unique())
        daily_charge = num_of_days*tariff['Parameters']['Daily']['Value']
        energy_charge = annual_kwh*tariff['Parameters']['Energy']['Value']*(1-tariff['Discount (%)']/100)
        fit_rebate = annual_kwh_exp*tariff['Parameters']['FiT']['Value']
        annual_bill = {'Annual_kWh': annual_kwh, 'Annual_kWh_Exp': annual_kwh_exp, 'Num_of_Days': num_of_days,
                       'Daily_Charge': daily_charge, 'Energy_Charge with discount': energy_charge, 'FiT_Rebate': fit_rebate,
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
                                                         (((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) &
                                                          ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                    else:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday < 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) |
                                                          ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                if this_part['Weekend']:
                    if start_hour <= end_hour:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday >= 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) &
                                                          ((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           < (60 * end_hour + end_min))), ti, load_profile.time_ind)
                    else:
                        load_profile.time_ind = np.where((load_profile['TS'].dt.weekday >= 5) &
                                                         (load_profile['TS'].dt.month.isin(this_part['Month'])) &
                                                         (((60 * load_profile['TS'].dt.hour + load_profile['TS'].dt.minute)
                                                           >= (60 * start_hour + start_min)) |
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
    #  - Historical usage (read from bill)
    #  - Demographic information
    # It can be a combination of above as well.
    # The output is half hourly load profile for a typical year (adjusted for temperature)
    #
    DemogInfo = pd.DataFrame(0, index=[0], columns=['CONTROLLED_LOAD_CNT', 'DRYER_USAGE_CD', 'NUM_REFRIGERATORS',
                                                    'NUM_ROOMS_HEATED', 'HAS_GAS_HEATING', 'HAS_GAS_HOT_WATER',
                                                    'HAS_GAS_COOKING', 'HAS_POOLPUMP', 'NUM_OCCUPANTS', 'SemiDetached',
                                                    'SeparateHouse', 'Unit', 'Ducted', 'NoAirCon', 'OtherAirCon',
                                                    'SplitSystem'])

    DemogInfo['NUM_OCCUPANTS'] = user_inputs['family_size']

    if user_inputs['poolpump'] == 'yes':
        DemogInfo['HAS_POOLPUMP'] = 1

    if user_inputs['controlled_load'] == 'yes':
        DemogInfo['CONTROLLED_LOAD_CNT'] = 1

    if user_inputs['dryer_usage'] == 'high':
        DemogInfo['DRYER_USAGE_CD'] = 3
    elif user_inputs['dryer_usage'] == 'medium':
        DemogInfo['DRYER_USAGE_CD'] = 2
    elif user_inputs['dryer_usage'] == 'low':
        DemogInfo['DRYER_USAGE_CD'] = 1
    else:
        DemogInfo['DRYER_USAGE_CD'] = 0

    if user_inputs['AC_type'] == 'Split':
        DemogInfo['SplitSystem'] = 1
    elif user_inputs['AC_type'] == 'Ducted':
        DemogInfo['Ducted'] = 1
    elif user_inputs['AC_type'] == 'OtherAirCon':
        DemogInfo['OtherAirCon'] = 1
    else:
        DemogInfo['NoAirCon'] = 1

    if user_inputs['dwell'] == 'SemiDetached':
        DemogInfo['SemiDetached'] = 1
    elif user_inputs['dwell'] == 'Unit':
        DemogInfo['Unit'] = 1
    else:
        DemogInfo['SeparateHouse'] = 1

    if user_inputs['HAS_GAS_HEATING'] == 'yes':
        DemogInfo['HAS_GAS_HEATING'] = 1

    if user_inputs['HAS_GAS_HOT_WATER'] == 'yes':
        DemogInfo['HAS_GAS_HOT_WATER'] = 1

    if user_inputs['HAS_GAS_COOKING'] == 'yes':
        DemogInfo['HAS_GAS_COOKING'] = 1

    DemogInfo['NUM_ROOMS_HEATED'] = user_inputs['NUM_ROOMS_HEATED']

    DemogInfo['NUM_REFRIGERATORS'] = user_inputs['NUM_REFRIGERATORS']

    # Get the DegreeDays data for the particular location (It's currently in pickle file and it is 3mb.
    # We may want to move it to server and get only the data for desired location

    # Reading the whole RMY files and find the closest one to the user location
    WeatherData_RMY_daily_all = pickle.load(open('AllWeather.pickle', 'rb'))
    ListofWS = pd.read_csv('ermy_locations.csv')
    indWS = ((ListofWS['Longitude'] - user_inputs['long']).abs() + (
                ListofWS['Latitude'] - user_inputs['lat']).abs()).idxmin()
    WeatherData_RMY_daily = WeatherData_RMY_daily_all[
        WeatherData_RMY_daily_all['loc'] == ListofWS.loc[indWS, 'TLA'] + '_' + str(
            ListofWS.loc[indWS, 'BOM_site'])].copy()

    # Estimate the temperature dependency
    # load the trained models for estimating coefficients
    cwd = os.getcwd()
    CDD_Coef = pickle.load(open(os.path.join(cwd, 'Models\CDD_Coef.pickle'), 'rb'))
    HDD_Coef = pickle.load(open(os.path.join(cwd, 'Models\HDD_Coef.pickle'), 'rb'))

    NewDF = pd.concat([pd.DataFrame({'Weekday': [1] * 24}), pd.concat([DemogInfo] * 24, ignore_index=True)], axis=1)
    NewDF['Hour'] = range(0, 24)
    NewDF2 = NewDF.copy()
    NewDF2['Weekday'] = 0
    NewDF = pd.concat([NewDF, NewDF2])

    NewDF['CDD_coef'] = CDD_Coef.predict(NewDF.values)
    NewDF['HDD_coef'] = HDD_Coef.predict(NewDF.iloc[:, 0:-1].values)
    NewDF['CDD_coef'].clip(lower=0, inplace=True)
    NewDF['HDD_coef'].clip(lower=0, inplace=True)

    # If load profile is provided: (date and location should be known)
    if user_inputs['load_profile_provided'] == 'yes':

        # Example: Load the example csv file and assume it is provided by user:
        # It should be kWh over half hourly periods. First column should be datetime and next one should be kWh

        SampleLoad = user_input_load_profile.copy()
        # todo: quality control of the load data
        SampleLoad.rename(columns={SampleLoad.columns[0]:'READING_DATETIME',SampleLoad.columns[1]:'kWh'},inplace=True)
        SampleLoad['READING_DATETIME'] = pd.to_datetime(SampleLoad['READING_DATETIME'])
        SampleLoad['TS_N'] = SampleLoad['READING_DATETIME'].dt.normalize()
        # 1- Temperature data of the same period of time should be grabbed

        StartDate = SampleLoad['READING_DATETIME'].min().strftime('%Y%m%d')
        EndDate = SampleLoad['READING_DATETIME'].max().strftime('%Y%m%d')

        URL = 'https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?request=execute&identifier=SinglePoint&parameters=CDD18_3,HDD18_3&startDate={}&endDate={}&userCommunity=SSE&tempAverage=DAILY&outputList=JSON,ASCII&lat={}&lon={}&user=anonymous'.format(
            StartDate, EndDate, str(user_inputs['lat']), str(user_inputs['long']))

        LP = requests.get(URL)
        LP = LP.json()
        DH_Temp_WS = pd.DataFrame.from_dict(LP['features'][0]['properties']['parameter'],
                                            orient='columns').reset_index()
        DH_Temp_WS.rename(columns={'index': 'TS', 'CDD18_3': 'CDD', 'HDD18_3': 'HDD'}, inplace=True)

        # Adjusting the HDD and CDD
        DH_Temp_WS['CDD'].where(DH_Temp_WS['CDD'] > DH_Temp_WS['HDD'], 0, inplace=True)
        DH_Temp_WS['HDD'].where(DH_Temp_WS['HDD'] > DH_Temp_WS['CDD'], 0, inplace=True)

        DH_Temp_WS['TS'] = pd.to_datetime(DH_Temp_WS['TS'], format='%Y%m%d')
        # Adding HDH and CDH to load profile
        SampleLoad = SampleLoad.join(DH_Temp_WS.set_index(['TS']), on=['TS_N'])
        SampleLoad['Hour'] = SampleLoad['READING_DATETIME'].dt.hour
        SampleLoad['Weekday'] = ((SampleLoad['TS_N'].dt.dayofweek) // 5 == 0).astype(int)

        SampleLoad = SampleLoad.join(NewDF.set_index(['Hour', 'Weekday']), on=['Hour', 'Weekday'])
        SampleLoad['day'] = SampleLoad['TS_N'].dt.day
        SampleLoad['month'] = SampleLoad['TS_N'].dt.month

        # 3- Adjust the load profile to new temperature data (RMY)

        SampleLoad = SampleLoad.join(WeatherData_RMY_daily.set_index(['day', 'month']), on=['day', 'month'],
                                     rsuffix='_New')
        SampleLoad['kWh_adj'] = SampleLoad['kWh'] + (SampleLoad['HDD_New'] - SampleLoad['HDD']) * SampleLoad[
            'HDD_coef'] + (SampleLoad['CDD_New'] - SampleLoad['CDD']) * SampleLoad['CDD_coef']

        SampleLoad['kWh_adj'] = SampleLoad['kWh_adj'].clip(lower=0)
        EstimatedLoad = SampleLoad[['READING_DATETIME', 'kWh_adj']].rename(
            columns={'READING_DATETIME': 'TS', 'kWh_adj': 'kWh'})

    # Todo: check the energy etc to be similar

    else:

        # load the trained models for estimating underlying load
        cwd = os.getcwd()
        LoadModel = pickle.load(open(os.path.join(cwd, 'Models\TempCor.pickle'), 'rb'))

        # ['Hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring',
        #        'CONTROLLED_LOAD_CNT', 'DRYER_USAGE_CD', 'NUM_REFRIGERATORS',
        #        'NUM_ROOMS_HEATED', 'HAS_GAS_HEATING', 'HAS_GAS_HOT_WATER',
        #        'HAS_GAS_COOKING', 'HAS_POOLPUMP', 'NUM_OCCUPANTS', 'SemiDetached',
        #        'SeparateHouse', 'Unit', 'Ducted', 'NoAirCon', 'OtherAirCon',
        #        'SplitSystem']

        NewDF_L = pd.DataFrame(data={'Hour': range(0, 24)})
        NewDF_L['Weekday'] = 1
        NewDF_L['Summer'] = 1
        NewDF_L['Winter'] = 0
        NewDF_L['Fall'] = 0
        NewDF_L['Spring'] = 0
        NewDF_L2 = pd.concat([DemogInfo] * 24).reset_index(drop=True)
        # pd.concat([NewDF_L2,NewDF_L],axis=1)
        NewDF_L3 = pd.concat([NewDF_L, NewDF_L2], axis=1)
        NewDF_L4 = NewDF_L3.copy()
        NewDF_L3['Weekday'] = 0
        NewDF_L3 = pd.concat([NewDF_L4, NewDF_L3], axis=0)

        NewDF_L4 = NewDF_L3.copy()

        NewDF_L4['Summer'] = 0
        NewDF_L4['Winter'] = 1
        NewDF_L3 = pd.concat([NewDF_L3, NewDF_L4], axis=0)

        NewDF_L4['Winter'] = 0
        NewDF_L4['Fall'] = 1
        NewDF_L3 = pd.concat([NewDF_L3, NewDF_L4], axis=0)

        NewDF_L4['Fall'] = 0
        NewDF_L4['Spring'] = 1
        NewDF_L3 = pd.concat([NewDF_L3, NewDF_L4], axis=0).reset_index(drop=True)


        NewDF_L3['kWh_pred'] = LoadModel.predict(NewDF_L3.values)

        NewDF_L3 = NewDF_L3.join(NewDF[['Weekday', 'Hour', 'CDD_coef', 'HDD_coef']].set_index(['Hour', 'Weekday']),
                                 on=['Hour', 'Weekday'])
        # Todo: important : check the kwh over 30 min or 1 hour.. in estimation it should be kW not kWh..
        FullLoad = pd.DataFrame(pd.date_range(start='1/1/1990T01:00', end='1/1/1991', freq='H'), columns=['TS'])
        # FullLoad.drop(columns={'ghi','dni','dhi','WS','HDH','CDH'},inplace=True)

        FullLoad['Summer'] = 1
        FullLoad['Winter'] = 1
        FullLoad['Fall'] = 1
        FullLoad['Spring'] = 1
        FullLoad['month'] = FullLoad['TS'].dt.month
        FullLoad['day'] = FullLoad['TS'].dt.day
        FullLoad['hour'] = FullLoad['TS'].dt.hour

        FullLoad['Summer'].where(FullLoad['month'].isin([12, 1, 2]), 0, inplace=True)
        FullLoad['Winter'].where(FullLoad['month'].isin([6, 7, 8]), 0, inplace=True)
        FullLoad['Fall'].where(FullLoad['month'].isin([3, 4, 5]), 0, inplace=True)
        FullLoad['Spring'].where(FullLoad['month'].isin([9, 10, 11]), 0, inplace=True)
        FullLoad['Weekday'] = ((FullLoad['TS'].dt.dayofweek) // 5 == 0).astype(int)

        FullLoad = FullLoad.join(WeatherData_RMY_daily.iloc[:, 1:].set_index(['day', 'month']), on=['day', 'month'])

        FullLoad = FullLoad.join(NewDF_L3.set_index(['Hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring']),
                                 on=['hour', 'Weekday', 'Summer', 'Winter', 'Fall', 'Spring'])
        FullLoad['kWh_adj'] = FullLoad['kWh_pred'] + FullLoad['HDD'] * FullLoad['HDD_coef'] + FullLoad['CDD'] * \
                              FullLoad['CDD_coef']

        EstimatedLoad = FullLoad[['TS', 'kWh_adj']].rename(columns={'kWh_adj': 'kWh'})

        # # if Usage provided: all usage or peak/offpeak/shoulder
        if len(user_inputs['previous_usage']) > 0:

            # todo: create a list of DNSPs TOU times and use here

            if user_inputs['smart_meter'] == 'yes':

                Distributor = distributor
                with open('AllTOU.json') as f:
                    AllTOU = json.load(f)
                for i in range(len(AllTOU)):
                    if AllTOU[i]['Distributor'] == Distributor:
                        TOUTimes = AllTOU[i]

                FullLoad['TOU'] = 'N/A'
                for k1, v1 in TOUTimes['TOU'].items():
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
                                    FullLoad.loc[(FullLoad['TS'].dt.weekday < 5) &
                                                 (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                                 (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   >= (60 * start_hour + start_min)) &
                                                  ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   < (60 * end_hour + end_min))), 'TOU'] = k1
                                else:
                                    FullLoad.loc[(FullLoad['TS'].dt.weekday < 5) &
                                                 (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                                 (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   >= (60 * start_hour + start_min)) |
                                                  ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   < (60 * end_hour + end_min))), 'TOU'] = k1
                            if this_part['Weekend']:
                                if start_hour <= end_hour:
                                    FullLoad.loc[(FullLoad['TS'].dt.weekday >= 5) &
                                                 (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                                 (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   >= (60 * start_hour + start_min)) &
                                                  ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   < (60 * end_hour + end_min))), 'TOU'] = k1
                                else:
                                    FullLoad.loc[(FullLoad['TS'].dt.weekday >= 5) &
                                                 (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                                 (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   >= (60 * start_hour + start_min)) |
                                                  ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                                   < (60 * end_hour + end_min))), 'TOU'] = k1

            else:
                FullLoad['TOU'] = 'Total'

            kb = 1
            UsageInfo = pd.DataFrame()
            for k3, v3 in user_inputs['previous_usage'].items():
                # start_day=min(pd.to_datetime(v3['start_date']),start_day)
                # end_day = max(pd.to_datetime(v3['end_date']), end_day)

                UsageInfo1 = pd.DataFrame({'TS': pd.date_range(start=v3['start_date'], end=v3['end_date'])})
                UsageInfo1['Peak'] = v3['peak']
                UsageInfo1['OffPeak'] = v3['offpeak']
                UsageInfo1['Shoulder'] = v3['shoulder']
                UsageInfo1['Total'] = v3['total']
                UsageInfo1['BillNo'] = kb
                kb += 1
                UsageInfo = pd.concat([UsageInfo, UsageInfo1], axis=0)

            StartDate = UsageInfo['TS'].min().strftime('%Y%m%d')
            EndDate = UsageInfo['TS'].max().strftime('%Y%m%d')

            URL = 'https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?request=execute&identifier=SinglePoint&parameters=CDD18_3,HDD18_3&startDate={}&endDate={}&userCommunity=SSE&tempAverage=DAILY&outputList=JSON,ASCII&lat={}&lon={}&user=anonymous'.format(
                StartDate, EndDate, str(user_inputs['lat']), str(user_inputs['long']))

            LP = requests.get(URL)
            LP = LP.json()
            DH_Temp_WS = pd.DataFrame.from_dict(LP['features'][0]['properties']['parameter'],
                                                orient='columns').reset_index()
            DH_Temp_WS.rename(columns={'index': 'TS', 'CDD18_3': 'CDD', 'HDD18_3': 'HDD'}, inplace=True)

            # WeatherData_sel = WeatherData[WeatherData['TS'].dt.normalize().isin(UsageInfo['TS'])].copy()
            DH_Temp_WS['CDD'].where(DH_Temp_WS['CDD'] > DH_Temp_WS['HDD'], 0, inplace=True)
            DH_Temp_WS['HDD'].where(DH_Temp_WS['HDD'] > DH_Temp_WS['CDD'], 0, inplace=True)

            DH_Temp_WS['TS'] = pd.to_datetime(DH_Temp_WS['TS'], format='%Y%m%d')

            # # todo: check if the 30min or hourly temp for calculating CDH and HDH in training and estimation..
            # #also load half and hourly ..

            DH_Temp_WS['month'] = DH_Temp_WS['TS'].dt.month
            DH_Temp_WS['day'] = DH_Temp_WS['TS'].dt.day

            UsageInfo = UsageInfo.join(DH_Temp_WS.set_index(['TS']), on=['TS'])

            if user_inputs['smart_meter'] == 'yes':
                UsageInfo2 = pd.melt(UsageInfo, id_vars=['TS', 'BillNo', 'CDD', 'HDD', 'month', 'day'],
                                     value_vars=['Peak', 'OffPeak', 'Shoulder'], var_name='TOU', value_name='kWh_usage')
            else:
                UsageInfo2 = pd.melt(UsageInfo, id_vars=['TS', 'BillNo', 'CDD', 'HDD', 'month', 'day'],
                                     value_vars=['Total'], var_name='TOU', value_name='kWh_usage')

            FullLoad2 = FullLoad.join(UsageInfo2.set_index(['month', 'day', 'TOU']), on=['month', 'day', 'TOU'],
                                      rsuffix='_new')
            #
            FullLoad2['kWh_adj_new'] = FullLoad2['kWh_pred'] + FullLoad2['HDD_new'] * FullLoad2['HDD_coef'] + FullLoad2[
                'CDD_new'] * FullLoad2['CDD_coef']
            FullLoad2['kWh_usage'] = FullLoad2['kWh_usage'].astype(float)

            Scaling = FullLoad2[['BillNo', 'TOU', 'kWh_usage', 'kWh_adj_new']].groupby(['BillNo', 'TOU'],
                                                                                       as_index=False).agg(
                {'kWh_adj_new': 'sum', 'kWh_usage': 'mean'})
            Scaling['Scale'] = Scaling['kWh_usage'] / Scaling['kWh_adj_new']
            Scaling2 = Scaling.groupby(['TOU'], as_index=False).agg({'Scale': 'mean'})
            Scaling2['BillNo'] = 'nan'
            Scaling2['BillNo'] = Scaling2['BillNo'].astype(float)
            Scaling_final = pd.concat([Scaling2, Scaling[['TOU', 'BillNo', 'Scale']]], sort=False)
            FullLoad2 = FullLoad2.join(Scaling_final.set_index(['TOU', 'BillNo']), on=['TOU', 'BillNo'])

            FullLoad2['kWh_adj_final'] = FullLoad2['kWh_adj'] * FullLoad2['Scale']
            EstimatedLoad = FullLoad2[['TS', 'kWh_adj_final']].rename(
                columns={'kWh_adj_final': 'kWh'})

    load_profile = EstimatedLoad.copy()

    load_profile['TS'] = load_profile['TS'].apply(lambda dt: dt.replace(year=1990))
    load_profile=load_profile.sort_values('TS')
    return load_profile


def battery(tariff, profiles_B, battery_kW, battery_kWh, distributor):

     # BattPow = 5  # 5KW
     # BattCap = 5  # 5kWh

    BattEff = 0.85
    delT = 2  # number of timestamps in each hour

    if tariff['Type'] == 'Flat_rate':

        Profiles_FR = profiles_B.copy()
        Profiles_FR['SOC'] = 0
        Profiles_FR['ExcessPV'] = Profiles_FR['PV'] - Profiles_FR['Load']
        Profiles_FR['ExcessLoad'] = Profiles_FR['Load'] - Profiles_FR['PV']
        Profiles_FR['ExcessPV'].clip(lower=0, upper=battery_kW, inplace=True)
        Profiles_FR['ExcessLoad'].clip(lower=0, upper=battery_kW, inplace=True)

        for i in range(1, len(Profiles_FR)):
            Profiles_FR.loc[i, 'SOC'] = max(0, min(battery_kWh,
                                                   Profiles_FR.loc[i - 1, 'SOC'] + (BattEff ** 0.5) * Profiles_FR.loc[
                                                       i, 'ExcessPV'] / delT - Profiles_FR.loc[i, 'ExcessLoad'] / (
                                                               BattEff ** 0.5) / delT))
            Profiles_FR.loc[i, 'ExcessCharge'] = Profiles_FR.loc[i, 'SOC'] - Profiles_FR.loc[i - 1, 'SOC']

        Profiles_FR['ExcessCharge'] = Profiles_FR['ExcessCharge'].apply(
            lambda x: x * (BattEff ** 0.5) if x < 0 else x / (BattEff ** 0.5))

        Profiles_FR['NetLoad'] = Profiles_FR['Load'] - Profiles_FR['PV'] + Profiles_FR['ExcessCharge'] * delT
        Profiles_B=Profiles_FR.copy()

    elif tariff['Type'] == 'TOU':

        Profiles_TOU = profiles_B.copy()
        Profiles_TOU['SOC'] = 0
        Profiles_TOU['ExcessPV'] = Profiles_TOU['PV'] - Profiles_TOU['Load']
        Profiles_TOU['ExcessLoad'] = Profiles_TOU['Load'] - Profiles_TOU['PV']
        Profiles_TOU['ExcessPV'].clip(lower=0, upper=battery_kW, inplace=True)
        Profiles_TOU['ExcessLoad'].clip(lower=0, upper=battery_kW, inplace=True)

        FullLoad = Profiles_TOU.copy()

        Distributor = distributor
        with open('AllTOU.json') as f:
            AllTOU = json.load(f)
        for i in range(len(AllTOU)):
            if AllTOU[i]['Distributor'] == Distributor:
                TOUTimes = AllTOU[i]

        FullLoad['TOU'] = 'N/A'
        for k1, v1 in TOUTimes['TOU'].items():
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
                            FullLoad.loc[(FullLoad['TS'].dt.weekday < 5) &
                                         (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                         (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           >= (60 * start_hour + start_min)) &
                                          ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           < (60 * end_hour + end_min))), 'TOU'] = k1
                        else:
                            FullLoad.loc[(FullLoad['TS'].dt.weekday < 5) &
                                         (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                         (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           >= (60 * start_hour + start_min)) |
                                          ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           < (60 * end_hour + end_min))), 'TOU'] = k1
                    if this_part['Weekend']:
                        if start_hour <= end_hour:
                            FullLoad.loc[(FullLoad['TS'].dt.weekday >= 5) &
                                         (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                         (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           >= (60 * start_hour + start_min)) &
                                          ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           < (60 * end_hour + end_min))), 'TOU'] = k1
                        else:
                            FullLoad.loc[(FullLoad['TS'].dt.weekday >= 5) &
                                         (FullLoad['TS'].dt.month.isin(this_part['Month'])) &
                                         (((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           >= (60 * start_hour + start_min)) |
                                          ((60 * FullLoad['TS'].dt.hour + FullLoad['TS'].dt.minute)
                                           < (60 * end_hour + end_min))), 'TOU'] = k1

        Profiles_TOU = FullLoad.copy()
        Profiles_TOU.loc[Profiles_TOU['TOU'] == 'OffPeak', 'ExcessLoad'] = 0
        Profiles_TOU.loc[Profiles_TOU['TOU'] == 'Shoulder', 'ExcessLoad'] = 0

        for i in range(1, len(Profiles_TOU)):
            Profiles_TOU.loc[i, 'SOC'] = max(0, min(battery_kWh, Profiles_TOU.loc[i - 1, 'SOC'] + (BattEff ** 0.5) *
                                                    Profiles_TOU.loc[i, 'ExcessPV'] / delT - Profiles_TOU.loc[
                                                        i, 'ExcessLoad'] / (BattEff ** 0.5) / delT))
            Profiles_TOU.loc[i, 'ExcessCharge'] = Profiles_TOU.loc[i, 'SOC'] - Profiles_TOU.loc[i - 1, 'SOC']

        Profiles_TOU['ExcessCharge'] = Profiles_TOU['ExcessCharge'].apply(
            lambda x: x * (BattEff ** 0.5) if x < 0 else x / (BattEff ** 0.5))

        Profiles_TOU['NetLoad'] = Profiles_TOU['Load'] - Profiles_TOU['PV'] + Profiles_TOU['ExcessCharge'] * delT
        Profiles_B=Profiles_TOU.copy()
    return Profiles_B

# ---------- Main Function -----------


def saving_est(user_inputs, pv_profile, selected_tariff, pv_size_kw, battery_kW, battery_kWh, distributor,
               user_input_load_profile=None):

    # First the load profile is estimated using the user input data
    load_profile = load_estimator(user_inputs, distributor, user_input_load_profile)
    load_profile=load_profile.set_index('TS')
    load_profile = load_profile.resample('30min').bfill().copy()
    load_profile.reset_index(inplace=True)
    # and the bill should be calculated using the load profile
    old_bill = bill_calculator(load_profile, selected_tariff)
    net_load = load_profile.copy()
    pv_profile['TS'] = pv_profile['TS'].apply(lambda dt: dt.replace(year=1990))
    pv_profile = pv_profile.sort_values('TS')
    pv_profile = pv_profile.set_index('TS')
    pv_profile = pv_profile.resample('30min').bfill().copy()
    pv_profile.reset_index(inplace=True)
    # Check and convert both load and PV profiles to half hourly
    net_load['kWh'] = net_load['kWh']- pv_profile['PV']
    new_bill = bill_calculator(net_load, selected_tariff)
    PVGen=pv_profile.PV.sum()
    Total_saving_SolarOnly = old_bill['Total_Bill'] - new_bill['Total_Bill']
    Saving_DueToNotUsingGrid_SolarOnly = Total_saving_SolarOnly - new_bill['FiT_Rebate'] # Savings due to consuming PV energy instead of grid

    # Seasonal Pattern (1: Summer, 2: fall, 3: winter, 4: spring)
    LSP = load_profile[['TS', 'kWh']].copy()
    LSP['kWh'] = LSP['kWh'] * 2  # convert to kW
    LSP['hour'] = LSP['TS'].dt.hour
    LSP['season'] = (LSP['TS'].dt.month % 12 + 3) // 3
    LSP.drop(['TS'], inplace=True, axis=1)
    LSP = LSP.groupby(['season', 'hour'], as_index=False).mean()
    LSP_json = LSP.to_json(orient='values')
    PVSP = pv_profile[['TS', 'PV']].copy()
    PVSP['PV']=PVSP['PV']*2   # convert to kW
    PVSP['hour'] = PVSP['TS'].dt.hour
    PVSP['season'] = (PVSP['TS'].dt.month % 12 + 3) // 3
    PVSP.drop(['TS'], inplace=True, axis=1)
    PVSP =PVSP.groupby(['season', 'hour'], as_index=False).mean()
    PVSP_json = PVSP.to_json(orient='values')

    # might remove LP from here and also the results
    LP_json=load_profile.to_json(orient='values')
    Results = {'Annual_PV_Generation': PVGen, 'Annual_PV_Generation_per_kW': PVGen / pv_size_kw,
               'Est_Annual_PV_export_SolarOnly': new_bill['Annual_kWh_Exp'],
               'Saving_due_to_not_using_grid_SolarOnly':
                   Saving_DueToNotUsingGrid_SolarOnly, 'FiT_Payment_SolarOnly': new_bill['FiT_Rebate'],
               'Annual_Saving_SolarOnly': Total_saving_SolarOnly,
               'Load_seasonal_pattern_kW': LSP_json, 'PV_seasonal_pattern_kW': PVSP_json,
               "Old_Bill": old_bill['Total_Bill'], "New_Bill_SolarOnly": new_bill['Total_Bill'],
               'Load_Prof': LP_json}

    # LoadProf = pd.read_json(LP_json)
    # LoadProf[0]=pd.to_datetime(LoadProf[0],unit='ms')

    # Battery Analysis
    # We assume if the tariff is flat rate, the battery is maximising self consumption
    # If the tariff is TOU, the battery doesn't let you discharge at off-peak and shoulder times.

    if battery_kWh>0:
        profiles_B = pv_profile.copy()
        profiles_B['PV'] = profiles_B['PV'] * 2  # convert to kW
        profiles_B['Load'] = load_profile['kWh'] * 2  # convert to kW
        BattRes = battery(selected_tariff, profiles_B, battery_kW, battery_kWh,distributor)
        net_load_batt = BattRes[['TS','NetLoad']].copy()
        net_load_batt.rename(columns={'NetLoad':'kWh'},inplace=True)
        net_load_batt['kWh']=net_load_batt['kWh']/2 # convert to kWh

        PVBattSP = net_load_batt[['TS', 'kWh']].copy()
        PVBattSP['kWh'] = PVBattSP['kWh'] * 2  # convert to kW
        PVBattSP['hour'] = PVBattSP['TS'].dt.hour
        PVBattSP['season'] = (PVBattSP['TS'].dt.month % 12 + 3) // 3
        PVBattSP.drop(['TS'], inplace=True, axis=1)
        PVBattSP = PVBattSP.groupby(['season', 'hour'], as_index=False).mean()
        PVBattSP_json = PVBattSP.to_json(orient='values')

        new_bill_batt = bill_calculator(net_load_batt, selected_tariff)
        Total_saving_SolBatt = old_bill['Total_Bill'] - new_bill_batt['Total_Bill']
        Saving_DueToNotUsingGrid_SolBatt = Total_saving_SolBatt - new_bill_batt[
            'FiT_Rebate']  # Savings due to consuming PV energy instead of grid
        NewLP_json = BattRes.to_json(orient='values')
        Results.update({'Est_Annual_PV_export_SolBatt': new_bill_batt['Annual_kWh_Exp'],
               'Saving_due_to_not_using_grid_SolBatt': Saving_DueToNotUsingGrid_SolBatt,
                        'FiT_Payment_SolBatt': new_bill_batt['FiT_Rebate'],
                        'New_Bill_SolBatt': new_bill_batt['Total_Bill'],'PVBatt_seasonal_pattern_kW': PVBattSP_json,
                        'Annual_Saving_SolBatt': Total_saving_SolBatt,'NewProfile':NewLP_json})

    return Results