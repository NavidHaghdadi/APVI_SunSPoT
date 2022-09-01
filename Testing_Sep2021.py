import pandas as pd
import numpy as np
import ast
# Run the SunSPoT and download the load profile to a txt file to see the results


my_file = open("Load profile results/LP1.txt", "r")
content = my_file.read()

LP = pd.DataFrame.from_records(ast.literal_eval(content))
LP.to_csv('Load profile results/LP1_refromat.csv',index=False)
LP.to_csv('Load profile results/LP1_refromat2.csv',index=False)


my_file = open("Load profile results/LPJuly2022.txt", "r")
content = my_file.read()

LP = pd.DataFrame.from_records(ast.literal_eval(content))
LP.to_csv('Load profile results/LP1_refromat_jul2022.csv',index=False)
