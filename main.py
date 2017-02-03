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
# plt.interactive(False)

# create a folder under the current work directory to save data
path = savedata.folder.create('nsga_II_reconstruction')

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
dim = lem.dim
x = np.empty(dim)
for _ in range(pop_size):
   prob.create_pop(x)
   pop.push_back(x)
print ("Initial pop generated!")


# # Plot the initial population.
# plt.figure()
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label='orginal')
# plt.show()

x = prob.trip_rate_opt()
print prob.calc_energy(x)
print prob.calc_heat_load(x)
print prob.calc_number_trips(x)

x = prob.adjust_gradient(x)
print prob.calc_energy(x)
print prob.calc_heat_load(x)
print prob.calc_number_trips(x)


pop.push_back(x)

#
# x = prob.heat_load_opt()
# print prob.calc_energy(x)
# print prob.calc_heat_load(x)
# print prob.calc_number_trips(x)

# # n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
# n_gen = [3185]
# pop = algo.opt(pop, n_gen, path)
# sav.save_pop('pop_nsga_II_100k.sl', pop)
# pop2 = population(prob_dth)
# sav.load_pop('pop_nsga_II_100k.sl', pop2)
# plt.figure()
# cur_f = np.array([ind.cur_f for ind in pop2]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label='NL NSGA_II 100000')
# plt.scatter(2992, 1.59, c='r', label='NL current setting')
# plt.legend()
# plt.grid()
# plt.savefig('NL_nsga_II_gen_100k.eps', format="eps")
# plt.show()


# # Compare the algorithms for efficiency
# pop1 = population(prob_dth)
# sav.load_pop(path+'/pop_nsga_II_2000', pop1)
# cur_f1 = np.array([ind.cur_f for ind in pop1]).T
#
# pop2 = population(prob_dth)
# sav.load_pop(path+'/pop_spea2_2820', pop2)
# cur_f2 = np.array([ind.cur_f for ind in pop2]).T
#
# pop3 = population(prob_dth)
# sav.load_pop(path+'/pop_sms_emoa_400000', pop3)
# cur_f3 = np.array([ind.cur_f for ind in pop3]).T
#
# pop4 = population(prob_dth)
# sav.load_pop(path+'/pop_nspso_3185', pop4)
# cur_f4 = np.array([ind.cur_f for ind in pop4]).T
#
# plt.figure()
# plt.scatter(cur_f1[0], cur_f1[1], c='r', label='NSGA II, gen=2000, t=86 s')
# plt.scatter(cur_f2[0], cur_f2[1], c='b', label='SPEA2, gen=2820, t=86 s')
# plt.scatter(cur_f3[0], cur_f3[1], c='g', label='SMS_EMOA, gen=400k, t=84 s')
# plt.scatter(cur_f4[0], cur_f4[1], c='k', label='NSPSO, gen=3185, t=86 s')
#
# plt.legend()
# plt.grid()
# plt.savefig(path+'/efficiency.eps', format="eps")
# plt.show()


# # Search for the best parameters
# ngen = 2000
# # m = [0.005, 0.01, 0.02, 0.04, 0.06]
# # cr = [0.95]
# # m = [0.06]
# # cr = [0.999, 0.95, 0.9, 0.8, 0.6]
# # nsga_race.race(pop, ngen, m, cr, path)
# # spea2_race.race(pop, ngen, m, cr, path)
# # sms_emoa_race.race(pop, ngen, m, cr, path)
#
# # c = [2.0, 2.4, 2.8, 4.0]
# # v = [0.01]
# # chi = [2.0]
# c = [2.4]
# v = [0.001, 0.005, 0.01, 0.03, 0.05]
# chi = [2.0]
# # c = [2.4]
# # v = [0.01]
# # chi = [0.6, 1.0, 1.4, 2.0, 2.4]
# nspso_race.race(pop, ngen, c, v, chi, path)

# #
# n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
# # n_gen = [3185]
# pop = algo.opt(pop, n_gen, path)
#
# lem.c_dim = 2
# # lem.trip_max = 30
# prob = lem.sl()
# print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
# # print('death penalty removed and contraint on trip rate added:')
# print(prob_dth)
#
# pop_new = population(prob_dth)
# dim = lem.dim
# for ind in pop:
#    pop_new.push_back(ind.best_x)
# #
# n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
# pop_new = algo.opt(pop_new, n_gen, path)
#
# sav.save_pop('pop_nsga_II_100k.sl', pop_new)
# pop2 = population(prob_dth)
# sav.load_pop('pop_nsga_II_100k', pop2)
# plt.figure()
# cur_f = np.array([ind.cur_f for ind in pop2]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label='NL NSGA_II 100000')
# plt.scatter(2992, 1.59, c='r', label='NL current setting')
# plt.legend()
# plt.grid()
# plt.savefig('NL_nsga_II_gen_100k.eps', format="eps")
# plt.show()



# #Compare the optimazed gradients with the operating gradients. some cavities are off
# file_gradient = 'data_prepare/NLgradients.sdds'
# gradient = cp.gradient(file_gradient)
# print(prob.calc_energy(gradient))
# print(prob.calc_number_trips(gradient))
# print(prob.calc_heat_load(gradient))
# print(" ")
#
# idx_off = list(compress(range(gradient.size), gradient == 0))
# print('The following cavities are off: ', idx_off)
#
# lem.dim = lem.dim - len(idx_off)
# lem.data = np.delete(lem.data, idx_off, 0)
# prob = lem.lem_upgrade()
# print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
# print('death penalty removed:')
# print(prob_dth)
#
# # Create original population
# pop_size = 128
# pop = population(prob_dth)
# dim = lem.dim
# x = np.empty(dim)
# for _ in range(pop_size):
#    prob.create_pop(x)
#    pop.push_back(x)
# print ("Initial pop generated!")
#
# # Plot the initial population.
# plt.figure()
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label='orginal')
# plt.show()
#
# # n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
# n_gen = [200]
# pop = algo.opt(pop, n_gen, path)
#
# lem.c_dim = 2
# lem.trip_max = 30
# prob = lem.lem_upgrade()
# print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
# # print('death penalty removed and contraint on trip rate added:')
# print(prob_dth)
#
# pop_new = population(prob_dth)
# dim = lem.dim
# for ind in pop:
#    pop_new.push_back(ind.best_x)
#
# n_gen = [500, 1000, 2000, 5000, 10000, 30000, 100000]
# pop_new = algo.opt(pop_new, n_gen, path)
# sav.save_pop('pop_nsga_II_100k_195.nl', pop_new)
#
# pop = pop_new
# # pop = population(prob_dth)
# # sav.load_pop('pop_nsga_II_100k_195.nl', pop)
# check_trips = np.empty(len(pop))
# trip_number = prob.calc_number_trips(gradient[gradient > 0])
# for idx, ind in enumerate(pop):
#     check_trips[idx] = ind.cur_f[1] - trip_number
#
# check_trips = np.fabs(check_trips)
# idx = np.argmin(check_trips)
# print(idx)
# print(pop[idx].cur_f[1])
# print(pop[idx].cur_f[0])
# gradient2 = np.asarray(pop[idx].cur_x, dtype='float64')
# print(prob.calc_energy(gradient2))
# print(prob.calc_number_trips(gradient2))
# print(prob.calc_heat_load(gradient2))
#
# plt.figure()
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label='NL NSGA_II 100k')
# plt.scatter(2992, 1.59, c='r', label='NL current setting')
# plt.scatter(pop[idx].cur_f[0],pop[idx].cur_f[1], c='m', label='NL selected')
# plt.legend()
# plt.grid()
# plt.savefig(path+'/NL_nsga_II_gen_100k_compare_195.eps', format="eps")
# plt.show()
#
#
# for idx, e in enumerate(idx_off):
#     gradient2 = np.insert(gradient2, e, 0)
#
# lem.dim += len(idx_off)
# prob = lem.nl()
# cp.compare(gradient,gradient2, path)
# cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient2, path)
# cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient, path)
#


# PF reconstruct
# plt.figure(2)
# sav.load_pop('pop_nsga_II_100k', pop)
# cur_f = np.array([ind.cur_f for ind in pop]).T
# plt.scatter(cur_f[0], cur_f[1], c='b', label = 'PF_all')

# lem.c_dim = 1
# lem.c_ineq_dim = 1
# prob = lem.nl()
# print(prob)
#
# n_off = 15
# (prob, id_down) = lem.revise_problem(n_off)
# print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
# print(prob_dth)
#
# pop_new = population(prob_dth)
# for ind in pop:
#     x = np.delete(ind.best_x, id_down)
#     prob.recreate_pop(x)
#     pop_new.push_back(x)
#
# cur_f = np.array([ind.cur_f for ind in pop_new]).T
# plt.scatter(cur_f[0], cur_f[1], c='k', label = 'New individuals')
#
# # pop_new2 = population(prob_dth)
# # for ind in pop:
# #     x = np.delete(ind.best_x, id_down)
# #     prob.recreate_pop2(x)
# #     pop_new2.push_back(x)
# #
# # cur_f = np.array([ind.cur_f for ind in pop_new2]).T
# # plt.scatter(cur_f[0], cur_f[1], c='m', label = 'cavities_off 2')
# #
# # pop_new3 = population(prob_dth)
# # for ind in pop:
# #     x = np.delete(ind.best_x, id_down)
# #     prob.recreate_pop3(x)
# #     pop_new3.push_back(x)
# #
# # cur_f = np.array([ind.cur_f for ind in pop_new3]).T
# # plt.scatter(cur_f[0], cur_f[1], c='r', label = 'cavities_off 3')
# #
# # pop_new4 = population(prob_dth)
# # for ind in pop:
# #     x = np.delete(ind.best_x, id_down)
# #     prob.recreate_pop4(x)
# #     pop_new4.push_back(x)
# #
# # cur_f = np.array([ind.cur_f for ind in pop_new4]).T
# # plt.scatter(cur_f[0], cur_f[1], c='c', label = 'cavities_off 4')
# #
# # pop_new5 = population(prob_dth)
# # for ind in pop:
# #     x = np.delete(ind.best_x, id_down)
# #     prob.recreate_pop5(x)
# #     pop_new5.push_back(x)
# #
# # cur_f = np.array([ind.cur_f for ind in pop_new5]).T
# # plt.scatter(cur_f[0], cur_f[1], c='g', label = 'cavities_off 5')
#
# plt.ylim(ymin=0, ymax=4000)
# plt.grid()
# plt.xlabel('Heat Load [W]')
# plt.ylabel('Trip Rate [per hour]')
# plt.legend()
# plt.savefig(path+'/'+'PF_reconstruct.eps', format="eps")
# plt.show()
#
# n_gen = [200, 400, 600, 1000, 2000]
# pop_new = algo.opt(pop_new, n_gen, path)
#
# pop_new4 = algo.opt(pop_new4, n_gen, path)
#
# pop_new5 = algo.opt(pop_new5, n_gen, path)

#
# pop.plot_pareto_fronts()