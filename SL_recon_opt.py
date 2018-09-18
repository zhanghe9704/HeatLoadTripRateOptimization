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
path = savedata.folder.create('SL_recon_opt')

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

plt.figure()
sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

lem.c_dim = 1
lem.c_ineq_dim = 1
prob = lem.sl()
print(prob)

n_off = 15
(prob, id_down_2) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down.txt', id_down_2, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)
ngen = [200, 500, 1000, 3000]

#Scaling
pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/scale_'+str(n_off)+'.sl', pop_new)

plt.xlim(xmin=2500)
plt.ylim(ymin=0)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_scaling_'+str(n_off)+'.eps', format="eps")
plt.show()


pop_new = algo.opt(pop_new, ngen, path)

#Dev
pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev_'+str(n_off)+'.sl', pop_new)

plt.xlim(xmin=2500)
plt.ylim(ymin=0)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_dev_'+str(n_off)+'.eps', format="eps")
plt.show()


pop_new = algo.opt(pop_new, ngen, path)

#Dev2
pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop4(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev2_'+str(n_off)+'.sl', pop_new)

plt.xlim(xmin=2500)
plt.ylim(ymin=0)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_dev2_'+str(n_off)+'.eps', format="eps")
plt.show()

pop_new = algo.opt(pop_new, ngen, path)

