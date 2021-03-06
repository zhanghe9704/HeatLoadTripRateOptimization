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


arr_off = np.array([2, 5, 10, 15])
arr_color = np.array(['r', 'g', 'm', 'k'])
rec_type = 'dev'

prob_type = 'nl'

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


n_total = 200
if (prob_type=='sl'):
    sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
elif (prob_type=='nl'):
    sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)
cur_f = np.array([ind.cur_f for ind in pop]).T
ax1.scatter(cur_f[0], cur_f[1], c='b',
            s=8, edgecolor='b', linewidth=0,
            label = str(n_total)+' cavities')


ax3.scatter(cur_f[0], cur_f[1], c='b',
            s=8, edgecolor='b', linewidth=0,
            label = str(n_total)+' cavities')
# n_off = 2

for n_off in arr_off:
    id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
    # print id_down

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
    ax1.scatter(cur_f[0][cur_f[1]<1000], cur_f[1][cur_f[1]<1000], c=arr_color[arr_off==n_off],
                s=8, edgecolor=arr_color[arr_off == n_off], linewidth=0,
                label = str(n_total-n_off)+' cavities')
    print 'nl n=1', n_off, cur_f[1].min(), cur_f[1].max()


ax1.set_ylim(ymin=0, ymax=100)
ax1.set_title('(a) North linac, n = 1')
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
ax1.set_xticks(np.arange(2500,3300+1,200))
# ax1.annotate('(a) North Linac', xytext=(2900,27), xy=(3030,27))
ax1.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax1.grid()
plt.tight_layout()

rec_type = 'dev2'

# n_off = 2

for n_off in arr_off:
    id_down = np.loadtxt(path+'/id_down_'+str(n_off)+'.txt')
    # print id_down

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
    ax3.scatter(cur_f[0][cur_f[1]<1000], cur_f[1][cur_f[1]<1000], c=arr_color[arr_off==n_off],
                s=8, edgecolor=arr_color[arr_off == n_off], linewidth=0,
                label = str(n_total-n_off)+' cavities')
    print 'nl n=2', n_off, cur_f[1].min(), cur_f[1].max()


ax3.set_ylim(ymin=0, ymax=100)
ax3.set_title('(c) North linac, n = 2')
ax3.set_xlabel('Heat load [W]')
ax3.set_ylabel('Trips [per hour]')
ax3.set_xticks(np.arange(2500,3300+1,200))
# ax3.annotate('(a) North Linac', xytext=(2900,27), xy=(3030,27))
ax3.legend(loc="upper right", handletextpad=0.2, labelspacing=-0)
ax3.grid()
plt.tight_layout()

rec_type = 'dev'
prob_type = 'sl'

if (prob_type == 'sl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/SL/PF_reconstruct/2017_01_25__14_25_12__SL_reconstruct'
    prob_fun = lem.sl
elif (prob_type == 'nl'):
    path = '/home/zhanghe/PycharmProjects/LEMUpgrade_12GeV/Plot/NL/PF_reconstruct/2017_01_25__15_37_56__NL_reconstruct'
    prob_fun = lem.nl

# Remove the constraints using death penalty
lem.dim = 200
prob = prob_fun()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_size = 128
pop = population(prob_dth)

n_total = 200
if (prob_type == 'sl'):
    sav.load_pop('Plot/SL/2017_01_18__20_44_14__SL_opt/pop_nsga_II_30000', pop)
elif (prob_type == 'nl'):
    sav.load_pop('Plot/NL/NL_pop_nsga_II_30k.nl', pop)
cur_f = np.array([ind.cur_f for ind in pop]).T
ax2.scatter(cur_f[0], cur_f[1], c='b',
            s=8, edgecolor='b', linewidth=0,
            label=str(n_total)+' cavities')

ax4.scatter(cur_f[0], cur_f[1], c='b',
            s=8, edgecolor='b', linewidth=0,
            label=str(n_total)+' cavities')
# n_off = 2

for n_off in arr_off:
    id_down = np.loadtxt(path + '/id_down_' + str(n_off) + '.txt')
    # print id_down

    lem.dim = 200
    prob = prob_fun()
    # prob = lem.sl()
    lem.dim = lem.dim - len(id_down)
    lem.data = np.delete(lem.data, id_down, 0)
    prob = lem.lem_upgrade()
    prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

    pop = population(prob_dth)
    sav.load_pop(path + '/' + rec_type + '_' + str(n_off) + '.' + prob_type, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    ax2.scatter(cur_f[0][cur_f[1] < 1000], cur_f[1][cur_f[1] < 1000], c=arr_color[arr_off == n_off],
                s=8, edgecolor=arr_color[arr_off == n_off], linewidth=0,
                label=str(n_total - n_off)+' cavities')
    print 'sl n=1', n_off, cur_f[1].min(), cur_f[1].max()


ax2.set_ylim(ymin=0, ymax=45)
ax2.set_xlim(xmin=2450)
ax2.set_title('(b) South linac, n = 1')
ax2.set_xlabel('Heat load [W]')
ax2.set_ylabel('Trips [per hour]')
ax2.legend(loc="upper left", handletextpad=0.2, labelspacing=-0)
plt.tight_layout()
ax2.grid()
# ax2.annotate('(b) South Linac', xytext=(2500,18.8), xy=(2500,18.8))



rec_type = 'dev2'

# n_off = 2

for n_off in arr_off:
    id_down = np.loadtxt(path + '/id_down_' + str(n_off) + '.txt')
    # print id_down

    lem.dim = 200
    prob = prob_fun()
    # prob = lem.sl()
    lem.dim = lem.dim - len(id_down)
    lem.data = np.delete(lem.data, id_down, 0)
    prob = lem.lem_upgrade()
    prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)

    pop = population(prob_dth)
    sav.load_pop(path + '/' + rec_type + '_' + str(n_off) + '.' + prob_type, pop)
    cur_f = np.array([ind.cur_f for ind in pop]).T
    ax4.scatter(cur_f[0][cur_f[1] < 1000], cur_f[1][cur_f[1] < 1000], c=arr_color[arr_off == n_off],
                s=8, edgecolor=arr_color[arr_off == n_off], linewidth=0,
                label=str(n_total - n_off)+' cavities')
    print 'sl n=2', n_off, cur_f[1].min(), cur_f[1].max()

ax4.set_ylim(ymin=0, ymax=45)
ax4.set_xlim(xmin=2450)
ax4.set_title('(d) South linac, n = 2')
ax4.set_xlabel('Heat load [W]')
ax4.set_ylabel('Trips [per hour]')
ax4.legend(loc="upper left", handletextpad=0.2, labelspacing=-0)
plt.tight_layout()
ax4.grid()

# ax4.annotate('(b) South Linac', xytext=(2500,18.8), xy=(2500,18.8))


plt.tight_layout()
path = savedata.folder.create('plots')
plt.savefig(path+'/recon_dev2.eps', format="eps")
plt.savefig(path+'/recon_dev2.pdf', format="pdf")
plt.savefig(path+'/recon_dev2.png', format="png")
plt.show()

