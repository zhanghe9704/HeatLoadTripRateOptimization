from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import random
import time
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle

from PyGMO import problem, population

import optimize.nsga_II as algo
# import optimize.nspso as algo
# import optimize.sms_emoa as algo
# import optimize.spea2 as algo
import algorithm_race.nsga_II as nsga_race
import algorithm_race.spea2 as spea2_race
import algorithm_race.sms_emoa as sms_emoa_race
import algorithm_race.nspso as nspso_race
import analysis.compare as cp
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem


# create a folder under the current work directory to save data
path = savedata.folder.create('plots')

# Remove the constraints using death penalty
prob = lem.sl()
print('orignal problem:')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print('death penalty removed:')
print(prob_dth)
#
# Create original population
pop_size = 128
pop = population(prob_dth)

sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
plt.figure()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label='SL NSGA_II 30,000')
plt.ylim(ymin=0)
plt.xlabel('Heat load [W]')
plt.ylabel('Trip rate [per hour]')
plt.legend()
plt.grid()
plt.savefig(path+'/SL_nsga_II_gen_30k.eps', format="eps")
plt.show()