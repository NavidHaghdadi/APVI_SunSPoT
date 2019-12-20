
import pandas as pd

# Testing output results
import ast

# reading load file
load_text_file = open("LPoutputSample.txt", "r")
x = load_text_file.readlines()
b = ast.literal_eval(x[0])
x2 = pd.DataFrame(b)
x2.columns = ['TS_Epoch', 'kWh', 'kW', 'TOU']
x2['TS'] = pd.to_datetime(x2['TS_Epoch'], unit='ms')

LoadOutput = x2.copy()

# reading PV file
text_file = open("PVOutputSample2.txt", "r")
x = text_file.readlines()
x[0] = x[0].replace('â†µ', ' ')

span = 2
x2 = x[0].split(" ")
x3 = [" ".join(x2[i:i+span]) for i in range(0, len(x2), span)]
x4 = pd.DataFrame([sub.split(",") for sub in x3])

PVOutput = x4.copy()

NEM12_1 = pd.read_csv('SampleNEM12.csv', skiprows=1, header=None)

import numpy as np

Chunks = np.where(NEM12_1[0] == 200)[0]
for i in range(0, len(Chunks)):
    if NEM12_1.iloc[Chunks[i], 3] == 'B1':
       NEM12_2 = NEM12_1.iloc[Chunks[i] + 1: Chunks[i+1] - 1, :].copy()

NEM12_2 = NEM12_2[NEM12_2[0] == 300].copy()
NEM12_2[1] = NEM12_2[1].astype(int).astype(str)

Nem12 = NEM12_2.iloc[:, 1:49].melt(id_vars=[1],var_name="HH", value_name="kWh")

Nem12['Datetime'] = pd.to_datetime(Nem12[1], format='%Y%m%d') + pd.to_timedelta(Nem12['HH']*30, unit='m')
Nem12.sort_values('Datetime', inplace=True)
Nem12.reset_index(inplace=True, drop=True)
LoadProfile = Nem12[['Datetime', 'kWh']].copy()


#
SGSCData = pd.read_csv(r'D:\Dropbox\UNSW\Database\SGSC\Home level\General_filtered.csv')
SGSCData['READING_DATETIME'] = pd.to_datetime(SGSCData['READING_DATETIME'])
SGSCData.set_index('READING_DATETIME', inplace=True)
SGSCData_mean = SGSCData.mean(axis=1)

SGSCData_mean_2013 = SGSCData_mean[SGSCData_mean.index.year>=2013].copy()
SGSCData_mean_2013 = SGSCData_mean_2013.iloc[1:].copy()



SGSCDemog = pd.read_csv(r'D:\Dropbox\UNSW\Database\SGSC\Home level\DemogInfo_filtered.csv')

DemogDist=dict()
for c in SGSCDemog.columns:
    DemogDist[c]=SGSCDemog[c].value_counts()