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

prob_type = 'sl'

if (prob_type=='sl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
    prob_fun = lem.sl
elif (prob_type=='nl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
    prob_fun = lem.nl

# Remove the constraints using death penalty
prob = prob_fun()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_size = 128
pop = population(prob_dth)

n_off = 15
id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
    # print id_down

lem.dim = 200
prob = prob_fun()

lem.dim = lem.dim - len(id_down)
lem.data = np.delete(lem.data, id_down, 0)
prob = lem.lem_upgrade()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop  = population(prob_dth)
x = np.empty(lem.dim)
for _ in range(pop_size):
   prob.create_pop(x)
   pop.push_back(x)

n_gen = [30000]
start = time.time()
pop = algo.opt(pop, n_gen, path)
end = time.time()
time = end - start
print(time)
sav.save_pop(path+'/30k_'+str(n_off)+'_'+str(time)+'.sl', pop)

print('Finished!')