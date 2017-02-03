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

import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem

random.seed()
# plt.interactive(False)

# create a folder under the current work directory to save data
path = savedata.folder.create('SL_opt')

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
dim = lem.dim
x = np.empty(dim)
for _ in range(pop_size):
   prob.create_pop(x)
   pop.push_back(x)
print ("Initial pop generated!")

# Plot the initial population.
plt.figure()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label='orginal')
plt.show()

# Start optimization
n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
pop = algo.opt(pop, n_gen, path)

