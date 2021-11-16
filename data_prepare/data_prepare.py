# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:58:37 2017

@author: zhanghe
"""

import numpy as np
import pandas as pd
import re
from itertools import compress

# # Choose the linac here.
linac = 'NL'  ## 'NL' or 'SL'

cavity = []
pattern = re.compile(r'\dL\d{2}-\d')
filename = linac+'parameters'  
with open(filename, "r") as ins:
    for line in ins:
        match = pattern.match(line)
        if match:
            cavity.append(match.group())
#            print match.group()
#    print cavity
            
columns = ['MaxGSet', 'ModAnode','OpsGSetMax', 'PhaseRMS', 'Q0', 'QExternal', 'TripOffset',
           'TripSlope']
df_ = pd.DataFrame(index=cavity, columns=columns)

with open(filename, "r") as ins:
    global column_index
    for line in ins:
        match = pattern.match(line)
        if match:
            row_index = match.group()
            column_index = 0
        else:
            df_.loc[row_index][column_index] = float(line)
            column_index += 1
#        print row_index
#        print column_index
            
for i in range(df_.index.size):
    if np.isnan(df_.iloc[i][4]):
        df_.iloc[i][2:] = df_.iloc[i][0:6]
        df_.iloc[i][1] = np.nan
    elif np.isnan(df_.iloc[i][5]):
        if df_.iloc[i][1]>3: # ModeAnode is missing
            df_.iloc[i][2:] = df_.iloc[i][1:7]
            df_.iloc[i][1] = np.nan
        elif df_.iloc[i][1]<4: #OpsGSetMax is missing
            df_.iloc[i][3:] = df_.iloc[i][2:7]
            df_.iloc[i][2] = df_.iloc[i][0]
        # df_.iloc[i][2:] = df_.iloc[i][1:7]
        # df_.iloc[i][1] = np.nan
    elif np.isnan(df_.iloc[i][6]):
        if df_.iloc[i][5]<100:   # MaxG, ModeAnode are missing, not TripOffset, TripSlope        
            df_.iloc[i][2:] = df_.iloc[i][0:6]
            df_.iloc[i][1] = np.nan
    elif np.isnan(df_.iloc[i][7]):
        if df_.iloc[i][1]>3: # ModeAnode is missing
            df_.iloc[i][2:] = df_.iloc[i][1:7]
            df_.iloc[i][1] = np.nan
        elif df_.iloc[i][1]<4: #OpsGSetMax is missing
            df_.iloc[i][3:] = df_.iloc[i][2:7]
            df_.iloc[i][2] = df_.iloc[i][0]
        
            
#index_nan = np.isnan(np.array(df_.iloc[:,7], dtype='float64'))
#index = list(compress(range(df_.index.size), index_nan))
#for i in index:
#    if not np.isnan(df_.iloc[i][6]) or (np.isnan(df_.iloc[i][6]) and
#        np.isnan(df_.iloc[i][5])):
#        df_.iloc[i][2:] = df_.iloc[i][1:7]
#        df_.iloc[i][1] = np.nan
#
#index_gradient = np.array(df_.iloc[:,2]>df_.iloc[:,0], dtype='float64')
#index = list(compress(range(df_.index.size), index_gradient))
#for i in index:
#    df_.iloc[i][2:] = df_.iloc[i][0:6]
#    df_.iloc[i][1] = np.nan 
            
            
cavity_length = np.arange(200).astype('float64')
cavity_length[cavity_length<160] = 0.5
cavity_length[cavity_length>0.5] = 0.7
df_['Length'] = pd.Series(cavity_length, index=df_.index)  
df_ = df_.fillna(0)  
df_.to_csv('lem_'+linac.lower()+'.csv', sep=',', columns = ('MaxGSet', 'OpsGSetMax', 'PhaseRMS', 'TripOffset', 'TripSlope', 'Q0', 'Length'))
