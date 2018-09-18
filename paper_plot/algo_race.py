from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from PyGMO import problem, population

import optimize.nsga_II as algo
import analysis.compare as cp
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem
import analysis.plot_setting as set_plot

set_plot.init_plotting()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

prob_fun = lem.nl
file_gradient = 'data_prepare/NLgradients.sdds'

prob = prob_fun()
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
# pop_size = 128
pop_nsga = population(prob)
pop_spea2 = population(prob)
pop_sms_emoa = population(prob)
pop_nspso = population(prob)

path = 'Plot/Plots for paper/algorithm_race'
sav.load_pop(path+'/pop_nsga_II_2000', pop_nsga)
sav.load_pop(path+'/pop_sms_emoa_400000', pop_sms_emoa)
sav.load_pop(path+'/pop_spea2_2820', pop_spea2)
sav.load_pop(path+'/pop_nspso_3185',pop_nspso)



ax1.set_xlim(xmin=2580, xmax=2700)
ax2.set_xlim(xmin=2580, xmax=2850)
# ax1.set_xticks(np.arange(2650, 3050+1, 100))
ax1.set_ylim(ymin=0, ymax=10)
ax2.set_ylim(ymin=0, ymax=40)
cur_f = np.array([ind.cur_f for ind in pop_spea2]).T
ax1.scatter(cur_f[0], cur_f[1], c='b', s=8*2, edgecolor='b', linewidth=0, label='SPEA2')
ax2.scatter(cur_f[0], cur_f[1], c='b', s=8*2, edgecolor='b', linewidth=0, label='SPEA2')
cur_f = np.array([ind.cur_f for ind in pop_sms_emoa]).T
ax1.scatter(cur_f[0], cur_f[1], c='g', s=8*2, edgecolor='g', linewidth=0, label='SMS\_EMOA')
ax2.scatter(cur_f[0], cur_f[1], c='g', s=8*2, edgecolor='g', linewidth=0, label='SMS\_EMOA')
cur_f = np.array([ind.cur_f for ind in pop_nsga]).T
ax1.scatter(cur_f[0], cur_f[1], c='k', s=8*2, edgecolor='k', linewidth=0, label='NSGAII')
ax2.scatter(cur_f[0], cur_f[1], c='k', s=8*2, edgecolor='k', linewidth=0, label='NSGAII')
cur_f = np.array([ind.cur_f for ind in pop_nspso]).T
ax2.scatter(cur_f[0], cur_f[1], c='m', s=8*2, edgecolor='m', linewidth=0, label='NSPSO')
# ax1.scatter(heat, trip, c='r', label='LEM')
# ax1.scatter(pop[idx].cur_f[0],pop[idx].cur_f[1], c='m', label='Selected')
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
ax1.legend(loc='upper right', scatterpoints=1)
ax1.grid()
ax2.set_xlabel('Heat load [W]')
ax2.set_ylabel('Trips [per hour]')
ax2.legend(loc='upper right', scatterpoints=1)
ax2.grid()



plt.tight_layout()
plt.savefig(path+'/algo_race.eps', format="eps")
plt.savefig(path+'/algo_race.pdf', format="pdf")
plt.savefig(path+'/algo_race.png', format="png")

plt.show()