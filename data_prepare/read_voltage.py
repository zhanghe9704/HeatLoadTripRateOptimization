# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:06:06 2017

@author: zhanghe
"""

import numpy as np
import data_prepare

data_prepare

filename = 'SLgradients.sdds'
lines = np.loadtxt(filename, dtype='str', skiprows=7)  # load data as string, first row is the cavity name
data = lines[:, 2].astype('float64')
voltage = data[0:200]


length = data_prepare.cavity_length
gradient = voltage/length
trip_slope = data_prepare.df_['TripSlope'].values
trip_offset = data_prepare.df_['TripOffset'].values
Q = data_prepare.df_['Q0'].values

cnst = np.empty(200)
cnst.fill(968.0)
cnst[length == 0.5] = 960.0

gradient = gradient*1e-6
fault = trip_slope * (gradient - trip_offset)
fault_rate = np.sum(np.exp(-10.26813067 + fault[trip_slope > 0]))
number_trips = 3600.0 * fault_rate
heat_load = np.sum(1e12 * (gradient * gradient) * length / (Q * cnst))

print 'total energy: ', voltage.sum()
print 'number of trips in an hour: ', number_trips
print 'heat_load: ', heat_load