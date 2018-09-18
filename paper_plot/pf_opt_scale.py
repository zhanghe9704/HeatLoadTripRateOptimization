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

ts = [8, 20, 41, 82, 124]
tn = [8, 21, 43, 87, 130]
n_gen = [200, 500, 1000, 2000, 3000]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

set_plot.init_plotting()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
# create a folder under the current work directory to save data

rec_type = 'scale'
n_off = 15
time_tag = '2017_04_07__14_37_29'
path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
prob_fun = lem.nl
id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
lem.dim = 200
prob = prob_fun()
lem.dim = lem.dim - len(id_down)
lem.data = np.delete(lem.data, id_down, 0)
prob = lem.lem_upgrade()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

pop_3k = population(prob_dth)
sav.load_pop(path+'/'+'30k_151330.57044387.nl', pop_3k)
cur_f = np.array([ind.cur_f for ind in pop_3k]).T
ax1.scatter(cur_f[0], cur_f[1], c = 'k', s = 8, edgecolor = 'k', linewidth = 0,
                label='30,000 gen., 1330 s ')



for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen)+'_'+time_tag, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    my_color = color[idx%len(color)]
    ax1.scatter(cur_f[0], cur_f[1], c = my_color, s = 8, edgecolor = my_color, linewidth = 0,
                label=str(gen) + ' gen., ' + str(tn[idx]) + ' s')


ax1.set_ylim(ymin=0, ymax = 50)
# ax1.set_xlim(xmin=2815, xmax = 2885)
ax1.set_xlim(xmin=2815, xmax = 3050)
# ax1.set_xticks(np.arange(2820,2880+1,20))
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
ax1.set_title('(a) North Linac')
# ax1.annotate('(a) North Linac', xytext=(2817,2), xy=(2820,2))
ax1.legend(loc="upper right", handletextpad=0.2, labelspacing=-0.0)
ax1.grid()

time_tag = '2017_04_07__14_42_18'
path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
prob_fun = lem.sl
id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
lem.dim = 200
prob = prob_fun()
lem.dim = lem.dim - len(id_down)
lem.data = np.delete(lem.data, id_down, 0)
prob = lem.lem_upgrade()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)


pop_3k = population(prob_dth)
sav.load_pop(path+'/'+'30k_15_1317.36492395.sl', pop_3k)
cur_f = np.array([ind.cur_f for ind in pop_3k]).T
ax2.scatter(cur_f[0], cur_f[1], c = 'k', s = 8, edgecolor = 'k', linewidth = 0,
                label='30,000 gen., 1317 s ')

for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen)+'_'+time_tag, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    ax2.scatter(cur_f[0], cur_f[1], c=color[idx%len(color)], edgecolor = color[idx%len(color)], s=8, linewidth = 0,
                label=str(gen) + ' gen., ' + str(ts[idx]) + ' s')



ax2.set_ylim(ymin=0, ymax=50)
ax2.set_xlim(xmin=2800, xmax=4200)
# ax2.set_xlim(xmin=2850, xmax=3050)
# ax2.set_xticks(np.arange(2850,3050+1,50))
ax2.set_xlabel('Heat load [W]')
ax2.set_ylabel('Trips [per hour]')
ax2.set_title('(b) South Linac')
# ax2.annotate('(b) South Linac', xytext=(2860,1.3), xy=(2860,1.3))
ax2.legend(loc="upper right", handletextpad=0.2, labelspacing=-0.0)
ax2.grid()
plt.tight_layout()

path = savedata.folder.create('plots')
plt.savefig(path+'/reopt_scale.eps', format="eps")
plt.savefig(path+'/reopt_scale.pdf', format="pdf")
plt.savefig(path+'/reopt_scale.png', format="png")
plt.show()