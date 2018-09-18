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



random.seed()
# create a folder under the current work directory to save data
path = savedata.folder.create('NL_recon_opt_trip_limit')

# Remove the constraints using death penalty
prob = lem.nl()
print('orignal problem:')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print('death penalty removed:')
print(prob_dth)
#
# Create original population
pop_size = 128
pop = population(prob_dth)

plt.figure()
sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

lem.c_dim = 1
lem.c_ineq_dim = 1
prob = lem.nl()
print(prob)

n_off = 10
(prob, id_down_2) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down.txt', id_down_2, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)
# ngen = [100, 200, 500, 1000, 3000, 5000]
ngen = [100, 200]

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

pop = algo.opt(pop_new, ngen, path)

lem.c_dim = 2
lem.c_ineq_dim = lem.c_dim
lem.trip_max = 10
prob = lem.lem_upgrade()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_new = population(prob_dth)
for ind in pop:
    pop_new.push_back(ind.best_x)

ngen = np.array([500, 1000, 2000, 3000, 5000]) - 200

print (ngen)
pop_new = algo.opt(pop_new, ngen, path)

print (len(pop))
print ('Finished!')