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
#  - if the tariff is flat rate or block rate, it is maximising the self consumption. i.e. always storing the excess PV in battery
#  - if it is TOU, it is maximising the self consumption but also doesn't discharge the battery until peak time.
#  4- Main function to call these.

# -------------------- Bill Calculator function -----------
def bill_calculator(load_profile, tariff, network_load=None, fit=True):
    # the input is load profile (kwh over half hourly) for one year and the tariff.
    # First input (column) of load_profile should be timestamp and second should be kwh
    # load_profile.columns={'TS','kWh'}
    # This function is originally created for CEEM's tariff tool and some of the functions are not used in SunSPot
    # CEEM Bill_calculator github: https://github.com/UNSW-CEEM/Bill_Calculator
    # CEEM tariff tool github: https://github.com/UNSW-CEEM/TDA_Python
    #
    load_profile = load_profile[['TS', 'kWh']].copy()
    load_profile.set_index('TS', inplace=True)
    load_profile = load_profile.fillna(0)

    def time_select(load_profile_s, par):
        load_profile_s_t_a = pd.DataFrame()
        for k2_1, v2_1, in par['TimeIntervals'].items():
            if v2_1[0][0:2] == '24':
                v2_1[0] = v2_1[1].replace("24", "00")
            if v2_1[1][0:2] == '24':
                v2_1[1] = v2_1[1].replace("24", "00")
            if v2_1[0] != v2_1[1]:
                load_profile_s_t = load_profile_s.between_time(start_time=v2_1[0], end_time=v2_1[1], include_start=False,
                                                           include_end=True)
            else:
                load_profile_s_t = load_profile_s.copy()

            if not par['Weekday']:
                load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.weekday >= 5].copy()

            if not par['Weekend']:
                load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.weekday < 5].copy()

            load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.month.isin(par['Month']), :].copy()

            load_profile_s_t_a = pd.concat([load_profile_s_t_a, load_profile_s_t])
        return load_profile_s_t_a

    # Calculate imports and exports
    results = {}

    Temp_imp = load_profile.values
    Temp_exp = Temp_imp.copy()
    Temp_imp[Temp_imp < 0] = 0
    Temp_exp[Temp_exp > 0] = 0
    load_profile_import = pd.DataFrame(Temp_imp, columns=load_profile.columns, index=load_profile.index)
    load_profile_export = pd.DataFrame(Temp_exp, columns=load_profile.columns, index=load_profile.index)

    results['LoadInfo'] = pd.DataFrame(index=[col for col in load_profile.columns],
                                       data=np.sum(load_profile_import.values, axis=0), columns=['Annual_kWh'])
    if fit:
        results['LoadInfo']['Annual_kWh_exp'] = -1 * np.sum(load_profile_export.values, axis=0)
    # If it is retailer put retailer as a component to make it similar to network tariffs
    if tariff['ProviderType'] == 'Retailer':
        tariff_temp = tariff.copy()
        del tariff_temp['Parameters']
        tariff_temp['Parameters'] = {'Retailer': tariff['Parameters']}
        tariff = tariff_temp.copy()

    for TarComp, TarCompVal in tariff['Parameters'].items():
        results[TarComp] = pd.DataFrame(index=results['LoadInfo'].index)

    # Calculate the FiT
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'FiT' in TarCompVal.keys():
            results[TarComp]['Charge_FiT_Rebate'] = -1 * results['LoadInfo']['Annual_kWh_exp'] * TarCompVal['FiT']['Value']
        elif 'FiT_TOU' in TarCompVal.keys():
            load_profile_ti_exp = pd.DataFrame()
            load_profile_ti_exp_charge = pd.DataFrame()
            for k, v in TarCompVal['FiT_TOU'].items():
                this_part = v.copy()
                if 'Weekday' not in this_part:
                    this_part['Weekday'] = True
                    this_part['Weekend'] = True
                if 'TimeIntervals' not in this_part:
                    this_part['TimeIntervals'] = {'T1': ['00:00', '00:00']}
                if 'Month' not in this_part:
                    this_part['Month'] = list(range(1, 13))
                load_profile_t_a = time_select(load_profile_export, this_part)
                load_profile_ti_exp[k] = load_profile_t_a.sum()
                results[TarComp]['kWh_Exp' + k] = load_profile_ti_exp[k].copy()
                load_profile_ti_exp_charge[k] = this_part['Value'] * load_profile_ti_exp[k]
                results[TarComp]['FiT_C_TOU' + k] = load_profile_ti_exp_charge[k].copy()
            results[TarComp]['Charge_FiT_Rebate'] = load_profile_ti_exp_charge.sum(axis=1)

    # Check if daily exists and calculate the charge
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Daily' in TarCompVal.keys():
            num_days = (len(load_profile.index.normalize().unique()) - 1)
            break
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Daily' in TarCompVal.keys():
            results[TarComp]['Charge_Daily'] = num_days * TarCompVal['Daily']['Value']

    # Energy
    # Flat Rate:
    # Check if flat rate charge exists and calculate the charge
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'FlatRate' in TarCompVal.keys():
            results[TarComp]['Charge_FlatRate'] = results['LoadInfo']['Annual_kWh'] * TarCompVal['FlatRate']['Value']


    # Block Annual:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockAnnual' in TarCompVal.keys():
            block_use = results['LoadInfo'][['Annual_kWh']].copy()
            block_use_charge = block_use.copy()
            # separating the blocks of usage
            lim = 0
            for k, v in TarCompVal['BlockAnnual'].items():
                block_use[k] = block_use['Annual_kWh']
                block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                block_use[k] = block_use[k] - lim
                block_use[k][block_use[k] < 0] = 0
                lim = float(v['HighBound'])
                block_use_charge[k] = block_use[k] * v['Value']
            del block_use['Annual_kWh']
            del block_use_charge['Annual_kWh']
            results[TarComp]['Charge_BlockAnnual'] = block_use_charge.sum(axis=1)

    # Block Quarterly:
    # check if it has quarterly and if yes calculate the quarterly energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockQuarterly' in TarCompVal.keys():
            for Q in range(1, 5):
                load_profile_q = load_profile_import.loc[
                                 load_profile_import.index.month.isin(list(range((Q - 1) * 3 + 1, Q * 3 + 1))), :]
                results['LoadInfo']['kWh_Q' + str(Q)] = [
                    np.nansum(load_profile_q[col].values[load_profile_q[col].values > 0])
                    for col in load_profile_q.columns]
            break

    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockQuarterly' in TarCompVal.keys():
            for Q in range(1, 5):
                block_use = results['LoadInfo'][['kWh_Q' + str(Q)]].copy()
                block_use_charge = block_use.copy()
                lim = 0
                for k, v in TarCompVal['BlockQuarterly'].items():
                    block_use[k] = block_use['kWh_Q' + str(Q)]
                    block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                    block_use[k] = block_use[k] - lim
                    block_use[k][block_use[k] < 0] = 0
                    lim = float(v['HighBound'])
                    block_use_charge[k] = block_use[k] * v['Value']
                del block_use['kWh_Q' + str(Q)]
                del block_use_charge['kWh_Q' + str(Q)]
                results[TarComp]['C_Q' + str(Q)] = block_use_charge.sum(axis=1)
            results[TarComp]['Charge_BlockQuarterly'] = results[TarComp][
                ['C_Q' + str(Q) for Q in range(1, 5)]].sum(axis=1)

    # Block Monthly:
    # check if it has Monthly and if yes calculate the Monthly energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockMonthly' in TarCompVal.keys():
            for m in range(1, 13):
                load_profile_m = load_profile_import.loc[load_profile_import.index.month == m, :]
                results['LoadInfo']['kWh_m' + str(m)] = [
                    np.nansum(load_profile_m[col].values[load_profile_m[col].values > 0])
                    for col in load_profile_m.columns]
            break

    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockMonthly' in TarCompVal.keys():
            for Q in range(1, 13):
                block_use = results['LoadInfo'][['kWh_m' + str(Q)]].copy()
                block_use_charge = block_use.copy()
                lim = 0
                for k, v in TarCompVal['BlockMonthly'].items():
                    block_use[k] = block_use['kWh_m' + str(Q)]
                    block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                    block_use[k] = block_use[k] - lim
                    block_use[k][block_use[k] < 0] = 0
                    lim = float(v['HighBound'])
                    block_use_charge[k] = block_use[k] * v['Value']
                del block_use['kWh_m' + str(Q)]
                del block_use_charge['kWh_m' + str(Q)]
                results[TarComp]['C_m' + str(Q)] = block_use_charge.sum(axis=1)
            results[TarComp]['Charge_BlockMonthly'] = results[TarComp][['C_m' + str(Q) for Q in range(1, 13)]].sum(
                axis=1)

    # Block Daily:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockDaily' in TarCompVal.keys():
            DailykWh = load_profile_import.resample('D').sum()
            block_use_temp_charge = DailykWh.copy()
            block_use_temp_charge.iloc[:, :] = 0
            lim = 0
            for k, v in TarCompVal['BlockDaily'].items():
                block_use_temp = DailykWh.copy()
                block_use_temp[block_use_temp > float(v['HighBound'])] = float(v['HighBound'])
                block_use_temp = block_use_temp - lim
                block_use_temp[block_use_temp < 0] = 0
                lim = float(v['HighBound'])
                block_use_temp_charge = block_use_temp_charge + block_use_temp * v['Value']
            results[TarComp]['Charge_BlockDaily'] = block_use_temp_charge.sum(axis=0)


    # TOU energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'TOU' in TarCompVal.keys():
            load_profile_ti = pd.DataFrame()
            load_profile_ti_charge = pd.DataFrame()
            for k, v in TarCompVal['TOU'].items():
                this_part = v.copy()
                if 'Weekday' not in this_part:
                    this_part['Weekday'] = True
                    this_part['Weekend'] = True
                if 'TimeIntervals' not in this_part:
                    this_part['TimeIntervals'] = {'T1': ['00:00', '00:00']}
                if 'Month' not in this_part:
                    this_part['Month'] = list(range(1, 13))
                load_profile_t_a = time_select(load_profile_import, this_part)
                load_profile_ti[k] = load_profile_t_a.sum()
                results[TarComp]['kWh_' + k] = load_profile_ti[k].copy()
                load_profile_ti_charge[k] = this_part['Value'] * load_profile_ti[k]
                results[TarComp]['C_' + k] = load_profile_ti_charge[k].copy()
            results[TarComp]['Charge_TOU'] = load_profile_ti_charge.sum(axis=1)

    # Demand charge:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Demand' in TarCompVal.keys():
            for DemCharComp, DemCharCompVal in TarCompVal['Demand'].items():
                ts_num = DemCharCompVal['Demand Window Length']  # number of timestamp
                num_of_peaks = DemCharCompVal['Number of Peaks']
                if ts_num > 1:
                    load_profile_r = load_profile_import.rolling(ts_num, min_periods=1).mean()
                else:
                    load_profile_r = load_profile_import.copy()
                load_profile_f = time_select(load_profile_r, DemCharCompVal)

                # if capacity charge is applied meaning the charge only applies when you exceed the capacity for
                #  a certain number of times
                if 'Capacity' in DemCharCompVal:
                    # please note the capacity charge only works with user's demand peak (not coincident peak)
                    # Customers can exceed their capacity level on x separate days per month during each interval
                    # (day or night). If they exceed more than x times, they will be charged for the highest
                    # exceedance of their capacity the capacity charge (if they don't exceed) is already included
                    # in the fixed charge so they only pay for the difference
                    capacity = DemCharCompVal['Capacity']['Value']
                    if 'Capacity Exceeded No' in DemCharCompVal:
                        cap_exc_no = DemCharCompVal['Capacity Exceeded No']
                    else:
                        cap_exc_no = 0
                    load_profile_f = load_profile_f - (capacity / 2)
                    load_profile_f = load_profile_f.clip(lower=0)
                    load_profile_f_g = load_profile_f.groupby(load_profile_f.index.normalize()).max()
                    for m in range(1, 13):
                        arr = load_profile_f_g.loc[load_profile_f_g.index.month == m, :].copy().values
                        cap_exc_no_val = np.sum(arr > 0, axis=0)
                        load_profile_f.loc[load_profile_f.index.month == m, cap_exc_no_val <= cap_exc_no] = 0
                    load_profile_f2 = load_profile_f.copy()
                else:
                    load_profile_f2 = load_profile_f.copy()
                based_on_network_peak = False
                if 'Based on Network Peak' in DemCharCompVal:
                    if DemCharCompVal['Based on Network Peak']:
                        based_on_network_peak = True
                # minimum demand or demand charge
                min_dem1 = 0
                min_dem2 = 0
                if 'Min Demand (kW)' in DemCharCompVal:
                    min_dem1 = DemCharCompVal['Min Demand (kW)']
                if 'Min Demand Charge ($)' in DemCharCompVal:
                    if DemCharCompVal['Value'] > 0:
                        min_dem2 = DemCharCompVal['Min Demand Charge ($)'] / DemCharCompVal['Value']
                min_dem = min(min_dem1, min_dem2)
                if based_on_network_peak:
                    new_load = pd.merge(load_profile_f2, network_load, left_index=True, right_index=True)
                    average_peaks_all = np.empty((0, new_load.shape[1] - 1), dtype=float)
                    for m in DemCharCompVal['Month']:
                        new_load2 = new_load.loc[new_load.index.month == m, :].copy()
                        new_load2.sort_values(by='NetworkLoad', inplace=True, ascending=False)
                        average_peaks_all = np.append(average_peaks_all,
                                                      [2 * new_load2.iloc[:num_of_peaks, :-1].values.mean(axis=0)],
                                                      axis=0)
                    average_peaks_all = np.clip(average_peaks_all, a_min=min_dem, a_max=None)
                    average_peaks_all_sum = average_peaks_all.sum(axis=0)
                else:
                    average_peaks_all = np.empty((0, load_profile_f.shape[1]), dtype=float)
                    for m in DemCharCompVal['Month']:
                        arr = load_profile_f.loc[load_profile_f.index.month == m, :].copy().values
                        arr.sort(axis=0)
                        arr = arr[::-1]
                        average_peaks_all = np.append(average_peaks_all, [2 * arr[:num_of_peaks, :].mean(axis=0)],
                                                      axis=0)
                    average_peaks_all = np.clip(average_peaks_all, a_min=min_dem, a_max=None)
                    average_peaks_all_sum = average_peaks_all.sum(axis=0)
                results[TarComp]['Avg_kW_' + DemCharComp] = average_peaks_all_sum / len(DemCharCompVal['Month'])
                results[TarComp]['C_' + DemCharComp] = average_peaks_all_sum * DemCharCompVal['Value']
                results[TarComp]['Charge_Demand'] = results[TarComp][
                    [col for col in results[TarComp] if col.startswith('C_')]].sum(axis=1)

    for k, v in results.items():
        if k != 'LoadInfo':
            results[k]['Bill'] = results[k][[col for col in results[k].columns if col.startswith('Charge')]].sum(axis=1)
    energy_comp_list = ['BlockAnnual', 'BlockQuarterly', 'BlockMonthly', 'BlockDaily', 'FlatRate', 'TOU']
    tariff_comp_list = []
    for TarComp, TarCompVal in tariff['Parameters'].items():
        for TarComp2, TarCompVal2 in tariff['Parameters'][TarComp].items():
            tariff_comp_list.append(TarComp2)
    tariff_comp_list = list(set(tariff_comp_list))
    energy_lst = [value for value in tariff_comp_list if value in energy_comp_list]

    if len(energy_lst) < 1:
        raise ValueError("There is no energy charge component. Please fix the tariff and try again!")
    elif len(energy_lst) > 1:
        raise ValueError("There are more than one energy charge component. Please fix the tariff and try again!")
    else:
        return results

# -------------------- Load estimation function -----------
def load_estimator(user_inputs, distributor, user_input_load_profile, first_pass):
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

        # the format should be NEM12 format. We assume if the input has more than 3 columns it is
        # provided in the NEM12 format and follows the standard format:
        # https://www.aemo.com.au/-/media/Files/Electricity/NEM/Retail_and_Metering/Metering-Procedures/2018/MDFF-Specification-NEM12--NEM13-v106.pdf

        if user_input_load_profile.shape[1] > 3:
            NEM12_1 = user_input_load_profile.copy()

            Chunks = np.where(NEM12_1[0] == 200)[0]
            for i in range(0, len(Chunks)):
                if NEM12_1.iloc[Chunks[i], 3] == 'B1':
                    NEM12_2 = NEM12_1.iloc[Chunks[i] + 1: Chunks[i + 1] - 1, :].copy()

            NEM12_2 = NEM12_2[NEM12_2[0] == 300].copy()
            NEM12_2[1] = NEM12_2[1].astype(int).astype(str)

            Nem12 = NEM12_2.iloc[:, 1:49].melt(id_vars=[1], var_name="HH", value_name="kWh")
            Nem12['kWh'] = Nem12['kWh'].astype(float)
            Nem12['Datetime'] = pd.to_datetime(Nem12[1], format='%Y%m%d') + pd.to_timedelta(Nem12['HH'] * 30, unit='m')
            Nem12.sort_values('Datetime', inplace=True)
            Nem12.reset_index(inplace=True, drop=True)
            sample_load = Nem12[['Datetime', 'kWh']].copy()
        else:
            sample_load = user_input_load_profile.copy()
            sample_load[0] = pd.to_datetime(sample_load[0], format='%d/%m/%Y %H:%M')


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

    load_profile['TS'] = load_profile['TS'].apply(lambda dt: dt.replace(year=1990))
    load_profile = load_profile.sort_values('TS')
    load_profile['kW'] = 2*load_profile['kWh']

    return load_profile

# -------------------- Battery Calculator function -----------
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
               user_input_load_profile, first_pass=False):
    # First the load profile is estimated using the user input data
    load_profile = load_estimator(user_inputs, distributor, user_input_load_profile, first_pass)

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
    total_saving_solar_only = old_bill['Retailer']['Bill'].values[0] - new_bill['Retailer']['Bill'].values[0]
    saving_due_to_not_using_grid_solar_only = total_saving_solar_only + new_bill['Retailer']['Charge_FiT_Rebate'].values[0]
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
    load_profile_kWh = load_profile.drop(columns=['kW'])
    load_profile_json = load_profile_kWh.to_json(orient='values')

    results = {'Annual_PV_Generation': pv_generation, 'Annual_PV_Generation_per_kW': pv_generation / pv_size_kw,
               'Est_Annual_PV_export_SolarOnly': new_bill['LoadInfo']['Annual_kWh_exp'].values[0],
               'Est_Annual_PV_self_consumption_SolarOnly': pv_generation-new_bill['LoadInfo']['Annual_kWh_exp'].values[0],
               'Saving_due_to_not_using_grid_SolarOnly': saving_due_to_not_using_grid_solar_only,
               'FiT_Payment_SolarOnly': -1 * new_bill['Retailer']['Charge_FiT_Rebate'].values[0],
               'Annual_Saving_SolarOnly': total_saving_solar_only,
               'Load_seasonal_pattern_kW': load_seasonal_pattern_json,
               'PV_seasonal_pattern_kW': pv_seasonal_pattern_json,
               "Old_Bill": old_bill['Retailer']['Bill'].values[0], "New_Bill_SolarOnly": new_bill['Retailer']['Bill'].values[0],
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
        total_saving_sol_batt = old_bill['Retailer']['Bill'].values[0] - new_bill_batt['Retailer']['Bill'].values[0]
        saving_due_to_not_using_grid_sol_batt = total_saving_sol_batt + new_bill_batt['Retailer']['Charge_FiT_Rebate'].values[0]
        # Savings due to consuming PV energy instead of grid
        new_load_profile_json = battery_result.to_json(orient='values')
        results.update({'Est_Annual_PV_export_sol_batt': new_bill_batt['LoadInfo']['Annual_kWh_exp'].values[0],
                        'Est_Annual_PV_self_consumption_sol_batt': pv_generation - new_bill_batt['LoadInfo']['Annual_kWh_exp'].values[0],
                        'Saving_due_to_not_using_grid_sol_batt': saving_due_to_not_using_grid_sol_batt,
                        'FiT_Payment_sol_batt': -1 * new_bill_batt['Retailer']['Charge_FiT_Rebate'].values[0],
                        'New_Bill_sol_batt': new_bill_batt['Retailer']['Bill'].values[0],
                        'pv_batt_seasonal_pattern_kW': pv_batt_seasonal_pattern_json,
                        'Annual_Saving_sol_batt': total_saving_sol_batt, 'NewProfile': new_load_profile_json})
    return results
