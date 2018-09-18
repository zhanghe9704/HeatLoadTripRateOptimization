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
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
# create a folder under the current work directory to save data


ts = [8, 22, 45, 89, 134]
tn = [8, 21, 42, 86, 130]
n_gen = [200, 500, 1000, 2000, 3000]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# set_plot.init_plotting()
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
# create a folder under the current work directory to save data

rec_type = 'dev2'
n_off = 15
time_tag = '2017_04_07__16_53_56'
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
ax3.scatter(cur_f[0], cur_f[1], c = 'k', s = 8, edgecolor = 'k', linewidth = 0,
                label='30,000 gen., 1330 s ')
for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen)+'_'+time_tag, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    my_color = color[idx%len(color)]
    ax3.scatter(cur_f[0], cur_f[1], c = my_color, s = 8, edgecolor = my_color, linewidth = 0,
                label=str(gen) + ' gen., ' + str(ts[idx]) + ' s')

ax3.set_title('(c) North linac, n = 2')
ax3.set_ylim(ymin=0, ymax = 50)
# ax3.set_xticks(np.arange(2820, 3000+1,40))
ax3.set_xlabel('Heat load [W]')
ax3.set_ylabel('Trips [per hour]')
# ax3.annotate('(a) North Linac', xytext=(2910,27), xy=(2910,27))
ax3.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax3.grid()

tn = [8, 20, 41, 83, 125]
rec_type = 'dev'
for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen), pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    my_color = color[idx%len(color)]
    ax1.scatter(cur_f[0], cur_f[1], c = my_color, s = 8, edgecolor = my_color, linewidth = 0,
                label=str(gen) + ' gen., ' + str(ts[idx]) + ' s')

ax1.set_title('(a) North linac, n = 1')
ax1.set_ylim(ymin=0, ymax = 50)
# ax1.set_xticks(np.arange(2820, 3000+1,40))
# ax1.set_xlim(xmax=3000,xmin=2820)
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
# ax3.annotate('(a) North Linac', xytext=(2910,27), xy=(2910,27))
ax1.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax1.grid()


rec_type = 'dev2'
time_tag = '2017_04_07__16_41_36'
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
ax4.scatter(cur_f[0], cur_f[1], c = 'k', s = 8, edgecolor = 'k', linewidth = 0,
                label='30,000 gen., 1317 s ')

for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen)+'_'+time_tag, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    ax4.scatter(cur_f[0], cur_f[1], c=color[idx%len(color)], edgecolor = color[idx%len(color)], s=8, linewidth = 0,
                label=str(gen) + ' gen., ' + str(ts[idx]) + ' s')

ax4.set_title('(d) South linac, n = 2')
ax4.set_ylim(ymin=0, ymax=50)
ax4.set_xlim(xmin=2800, xmax=4200)
# ax4.set_xticks(np.arange(3000, 4400+1, 250))
ax4.set_xlabel('Heat load [W]')
ax4.set_ylabel('Trips [per hour]')
# ax4.annotate('(b) South Linac', xytext=(3700,3.2), xy=(3370,3.2))
ax4.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax4.grid()


ts = [8, 21, 42, 87, 131]
rec_type = 'dev'

for idx, gen in enumerate(n_gen):
    pop = population(prob_dth)
    sav.load_pop(path+'/'+rec_type+'_'+str(n_off)+'/pop_nsga_II_'+str(gen), pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    ax2.scatter(cur_f[0], cur_f[1], c=color[idx%len(color)], edgecolor = color[idx%len(color)], s=8, linewidth = 0,
                label=str(gen) + ' gen., ' + str(ts[idx]) + ' s')

ax2.set_title('(b) South linac, n = 1')
ax2.set_ylim(ymin=0, ymax=50)
ax2.set_xlim(xmin=2800, xmax=4200)
# ax2.set_xlim(xmin=3000)
# ax2.set_xticks(np.arange(3000, 4400+1, 250))
ax2.set_xlabel('Heat load [W]')
ax2.set_ylabel('Trips [per hour]')
# ax4.annotate('(b) South Linac', xytext=(3700,3.2), xy=(3370,3.2))
ax2.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax2.grid()

plt.tight_layout()
path = savedata.folder.create('plots')
plt.savefig(path+'/recon_dev_all.eps', format="eps")
plt.savefig(path+'/recon_dev_all.pdf', format="pdf")
plt.savefig(path+'/recon_dev_all.png', format="png")
plt.show()

