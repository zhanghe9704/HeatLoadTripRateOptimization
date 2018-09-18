from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os
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
# import algorithm_race.nsga_II as nsga_race
# import algorithm_race.spea2 as spea2_race
# import algorithm_race.sms_emoa as sms_emoa_race
# import algorithm_race.nspso as nspso_race
import analysis.compare as cp
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem


random.seed()
# plt.interactive(False)

# create a folder under the current work directory to save data
path = savedata.folder.create('compare')

# arr_color = np.array(['r', 'g', 'm', 'k'])
rec_type = 'dev2'

prob_type = 'sl'

if (prob_type=='sl'):
    # path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
    prob_fun = lem.sl
    file_gradient = 'data_prepare/SLgradients.sdds'
elif (prob_type=='nl'):
    # path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
    prob_fun = lem.nl
    file_gradient = 'data_prepare/NLgradients.sdds'

prob = prob_fun()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_size = 128
pop = population(prob_dth)
if (prob_type=='sl'):
    sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
elif (prob_type=='nl'):
    sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)

#Compare the optimazed gradients with the operating gradients. some cavities are off

gradient = cp.gradient(file_gradient)
trip = prob.calc_number_trips(gradient)
heat = prob.calc_heat_load(gradient)
energy = prob.calc_energy(gradient)
print(energy)
print(trip)
print(heat)
print(" ")

idx_off = list(compress(range(gradient.size), gradient == 0))
print('The following cavities are off: ', idx_off)
print(lem.lines[idx_off, 0])

lem.dim = lem.dim - len(idx_off)
lem.data = np.delete(lem.data, idx_off, 0)
lem.c_dim = 2
lem.c_ineq_dim = 2
lem.trip_max = 10
lem.energy_tol = 0.5
if (prob_type=='sl'):
    lem.energy_tol = 4.5
prob = lem.lem_upgrade()
print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print('death penalty removed:')
print(prob_dth)


pop_new = population(prob_dth)
for ind in pop:
    x = np.delete(ind.best_x, idx_off)
    if (rec_type is 'dev2'):
        prob.recreate_pop4(x)
    elif (rec_type is 'scale'):
        prob.recreate_pop(x)
    pop_new.push_back(x)

rd = np.random.rand(4)
for r in rd:
    x = prob.trip_rate_opt(r*lem.energy_tol)
    x = prob.adjust_gradient(x)
    pop_new.push_back(x)

#
print('Start optimization!')
n_gen = [500, 1000, 2000, 3000]
pop_new = algo.opt(pop_new, n_gen, path)
print('End optimization!')

pop = pop_new
# pop = population(prob_dth)
# sav.load_pop('pop_nsga_II_100k_195.nl', pop)
check_trips = np.empty(len(pop))
trip_number = prob.calc_number_trips(gradient[gradient > 0])
for idx, ind in enumerate(pop):
    check_trips[idx] = ind.cur_f[1] - trip_number

check_trips = np.fabs(check_trips)
idx = np.argmin(check_trips)
print(idx)
print(pop[idx].cur_f[1])
print(pop[idx].cur_f[0])
gradient2 = np.asarray(pop[idx].cur_x, dtype='float64')
print(prob.calc_energy(gradient2))
# print(prob.calc_number_trips(gradient2))
# print(prob.calc_heat_load(gradient2))

with open(path+"/compare.txt", "a") as myfile:
    myfile.write("Operating setting:\n")
    myfile.write("Energy: %.2f\n"%energy)
    myfile.write("Heat load: %.2f\n"%heat)
    myfile.write("Trip rate: %.2f\n"%trip)
    myfile.write("Selected optimized setting:\n")
    myfile.write("Energy: %.2f\n"%prob.calc_energy(gradient2))
    myfile.write("Heat load: %.2f\n"%pop[idx].cur_f[0])
    myfile.write("Trip rate: %.2f\n"%pop[idx].cur_f[1])
    myfile.write("The cavities are down: \n")
    for item in lem.lines[idx_off, 0]:
        myfile.write(item+'\n')


plt.figure()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label=prob_type+' NSGA_II')
plt.scatter(heat, trip, c='r', label=prob_type+' current setting')
plt.scatter(pop[idx].cur_f[0],pop[idx].cur_f[1], c='m', label=prob_type+' selected')
plt.legend()
plt.grid()
plt.savefig(path+'/'+prob_type+'_nsga_II_gen_'+str(n_gen[-1])+'_compare_195.eps', format="eps")
plt.show()


for idx, e in enumerate(idx_off):
    gradient2 = np.insert(gradient2, e, 0)

lem.dim += len(idx_off)
prob = prob_fun()
cp.compare(gradient/gradient.max(),gradient2/gradient2.max(), path)
cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient2, path)
cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient, path)

