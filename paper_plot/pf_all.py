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
# create a folder under the current work directory to save data
path = savedata.folder.create('plots')


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

sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)

cur_f = np.array([ind.cur_f for ind in pop]).T
ax1.scatter(cur_f[0], cur_f[1], c='b', edgecolor='b', s = 8*2, label='North Linac')
ax1.set_ylim(ymin=0, ymax=28)
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
ax1.legend(loc="upper right")
ax1.set_xticks(np.arange(2500, 2900+1, 100))
ax1.grid()




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

sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)

cur_f = np.array([ind.cur_f for ind in pop]).T
ax2.scatter(cur_f[0], cur_f[1], c='b', edgecolor='b', s = 8*2, label='South Linac')
ax2.set_ylim(ymin=0, ymax=18)
ax2.set_xlabel('Heat load [W]')
ax2.set_ylabel('Trips [per hour]')
ax2.set_xticks(np.arange(2500, 2900+1, 100))
plt.legend(loc="upper right")
plt.tight_layout()
ax2.grid()

plt.savefig(path+'/PF_all_gen_30k.eps', format="eps")
plt.savefig(path+'/PF_all_gen_30k.pdf', format="pdf")
plt.savefig(path+'/PF_all_gen_30k.png', format="png")
plt.show()

