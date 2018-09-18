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
import analysis.plot_setting

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

arr_off = np.array([2, 5, 10, 15])
arr_color = np.array(['r', 'g', 'm', 'k'])
rec_type = 'scale'

prob_type = 'sl'

if (prob_type=='sl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
    prob_fun = lem.sl
elif (prob_type=='nl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
    prob_fun = lem.nl

# path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
# path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'

# Remove the constraints using death penalty
prob = prob_fun()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_size = 128
pop = population(prob_dth)

# plt.figure()
analysis.plot_setting.init_plotting();

n_total = 200
if (prob_type=='sl'):
    sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
elif (prob_type=='nl'):
    sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label = str(n_total)+' cavities')


# n_off = 2

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
    cur_f = np.array([ind.cur_f for ind in pop]).T
    plt.scatter(cur_f[0][cur_f[1]<1000], cur_f[1][cur_f[1]<1000], c=arr_color[arr_off==n_off], label = str(n_total-n_off)+' cavities')

plt.ylim(ymin=0, ymax = 55)
plt.xlim(xmin=2500, xmax=3300)
plt.grid()
plt.xlabel('Heat Load [W]')
plt.ylabel('Trip Rate [per hour]')
plt.legend(loc='upper right')
# plt.legend()
plt.savefig(path+'/'+prob_type+'_rec_'+rec_type+'.eps', format="eps")
plt.show()
