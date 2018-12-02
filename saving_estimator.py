# This function is for estimating financial benefits of installing PV

import numpy as np
# When user selects the area the PV profile is being generated (assuming it is a dataframe like the sample load profile)
# Also the load profile of user is being generated (with or without user's inputs)
# And the tariff is selected by user as described in bill_calculator
# a set of sample inputs for this function is provided in Testing_files.py


# ---------- Bill Calculator function -----------


def bill_calculator(load_profile, tariff):

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
        annual_kwh_exp = -1 * load_exp.sum()

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


# ---------- Function -----------
def saving_est(load_profile, pv_profile, selected_tariff):
    # first the bill should be calculated using the load profile
    old_bill = bill_calculator(load_profile, selected_tariff)
    net_load = load_profile.copy()
    net_load.Load = net_load.Load - pv_profile.PV
    new_bill = bill_calculator(net_load, selected_tariff)

    Results = {'Est_tot_PV_output': pv_profile.PV.sum(), 'Est_PV_export': new_bill['Annual_kWh_Exp'],
               'Est_Annual_Saving': old_bill['Total_Bill'] - new_bill['Total_Bill']}
    # there could be other parameters to report
    # print(Results)
    return Results