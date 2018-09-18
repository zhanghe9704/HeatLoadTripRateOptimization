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

arr_off = np.array([15])
# arr_off = np.array([5, 10, 15])
arr_color = np.array(['r', 'g', 'm', 'k'])
rec_type = 'scale'

prob_type = 'sl'

if (prob_type=='sl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
    prob_fun = lem.sl
elif (prob_type=='nl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
    prob_fun = lem.nl


for n_off in arr_off:
    id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
    print id_down

    lem.dim = 200
    prob = prob_fun()
    # prob = lem.sl()
    lem.dim = lem.dim - len(id_down)
    lem.data = np.delete(lem.data, id_down, 0)
    prob = lem.lem_upgrade()
    prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

    pop  = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'.'+prob_type, pop)

    rd = np.random.rand(4)
    for r in rd:
        x = prob.trip_rate_opt(r*lem.energy_tol)
        x = prob.adjust_gradient(x)
        pop.push_back(x)

    # cur_f = np.array([ind.cur_f for ind in pop]).T
    # idx = np.argmin(cur_f[1])
    # print(idx)
    # print(pop[idx].cur_f[1])
    # print(pop[idx].cur_f[0])
    # gradient2 = np.asarray(pop[idx].cur_x, dtype='float64')
    # cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient2, path)


    path_type = path+'/'+rec_type+'_'+str(n_off)
    if not (os.path.exists(path_type)):
        os.mkdir(path_type)
    ngen = np.array([200, 500, 1000, 2000, 3000])  #for nsga
    # ngen = [200, 500, 1000, 2000, 3000];
    # ngen = np.array([200, 500, 1000, 2000, 3000, 4000]) #for spea2
    pop = algo.opt(pop, ngen, path_type)

    # cur_f = np.array([ind.cur_f for ind in pop]).T
    # idx = np.argmin(cur_f[1])
    # print(idx)
    # print(pop[idx].cur_f[1])
    # print(pop[idx].cur_f[0])
    # gradient2 = np.asarray(pop[idx].cur_x, dtype='float64')
    # cp.compare_max(np.asarray(prob.range_up, dtype='float64'), gradient2, path)