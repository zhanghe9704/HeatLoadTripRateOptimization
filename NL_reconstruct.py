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
path = savedata.folder.create('NL_reconstruct')

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
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

lem.c_dim = 1
lem.c_ineq_dim = 1
prob = lem.nl()
print(prob)

n_off = 2
(prob, id_down_2) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down_'+str(n_off)+'.txt', id_down_2, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/scale_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 5
(prob, id_down_5) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down_'+str(n_off)+'.txt', id_down_5, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_5)
    prob.recreate_pop(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='g', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/scale_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 10
(prob, id_down_10) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down_'+str(n_off)+'.txt', id_down_10, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_10)
    prob.recreate_pop(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='m', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/scale_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 15
(prob, id_down_15) = lem.revise_problem(n_off)
np.savetxt(path+'/id_down_'+str(n_off)+'.txt', id_down_15, fmt='%i', delimiter=' ')
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print(prob_dth)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_15)
    prob.recreate_pop(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='k', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/scale_'+str(n_off)+'.nl', pop_new)

plt.ylim(ymin=0, ymax=4000)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_scaling.eps', format="eps")
plt.show()


plt.figure()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

lem.dim = 200
lem.c_dim = 1
lem.c_ineq_dim = 1
prob = lem.nl()
print(prob)

n_off = 2
lem.dim = lem.dim - len(id_down_2)
lem.data = np.delete(lem.data, id_down_2, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 5
lem.dim = lem.dim - len(id_down_5)
lem.data = np.delete(lem.data, id_down_5, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_5)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='g', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 10
lem.dim = lem.dim - len(id_down_10)
lem.data = np.delete(lem.data, id_down_10, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_10)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='m', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 15
lem.dim = lem.dim - len(id_down_15)
lem.data = np.delete(lem.data, id_down_15, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_15)
    prob.recreate_pop5(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='k', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev_'+str(n_off)+'.nl', pop_new)

plt.ylim(ymin=0)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_dev.eps', format="eps")
plt.show()



plt.figure()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

lem.dim = 200
lem.c_dim = 1
lem.c_ineq_dim = 1
prob = lem.nl()
print(prob)

n_off = 2
lem.dim = lem.dim - len(id_down_2)
lem.data = np.delete(lem.data, id_down_2, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_2)
    prob.recreate_pop4(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='r', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev2_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 5
lem.dim = lem.dim - len(id_down_5)
lem.data = np.delete(lem.data, id_down_5, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_5)
    prob.recreate_pop4(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='g', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev2_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 10
lem.dim = lem.dim - len(id_down_10)
lem.data = np.delete(lem.data, id_down_10, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_10)
    prob.recreate_pop4(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='m', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev2_'+str(n_off)+'.nl', pop_new)

lem.dim = 200
prob = lem.nl()
n_off = 15
lem.dim = lem.dim - len(id_down_15)
lem.data = np.delete(lem.data, id_down_15, 0)
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, id_down_15)
    prob.recreate_pop4(x)
    pop_new.push_back(x)

cur_f = np.array([ind.cur_f for ind in pop_new]).T
plt.scatter(cur_f[0], cur_f[1], c='k', label = str(n_off)+' Cav. off')
sav.save_pop(path+'/dev2_'+str(n_off)+'.nl', pop_new)

plt.ylim(ymin=0)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend()
plt.savefig(path+'/'+'PF_reconstruct_dev2.eps', format="eps")
plt.show()