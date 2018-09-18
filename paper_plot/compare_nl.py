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

path = 'Plot/NL/2017_01_28__00_49_09__compare'

prob_fun = lem.nl
file_gradient = 'data_prepare/NLgradients.sdds'

prob = prob_fun()
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
pop_size = 128
pop = population(prob_dth)

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
prob = lem.lem_upgrade()
# print(prob)
prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
print('death penalty removed:')
# print(prob_dth)

pop_new = population(prob_dth)
sav.load_pop(path+'/pop_nsga_II_3000', pop_new)

pop = pop_new
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

path = savedata.folder.create('compare')

cur_f = np.array([ind.cur_f for ind in pop]).T
ax1.set_xlim(xmin=2650, xmax=3050)
ax1.set_xticks(np.arange(2650, 3050+1, 100))
ax1.scatter(cur_f[0], cur_f[1], c='g', edgecolor = 'g', linewidth = 0, label='PF')
ax1.scatter(heat, trip, c='r',marker='^',  s = 40,  label='LEM')
ax1.scatter(pop[idx].cur_f[0],pop[idx].cur_f[1], c='k', marker='v', s = 40, label='Selected')
ax1.set_xlabel('Heat load [W]')
ax1.set_ylabel('Trips [per hour]')
ax1.legend(loc='upper right', scatterpoints=1,handletextpad=0.2, labelspacing=-0.06)
ax1.grid()

for idx, e in enumerate(idx_off):
    gradient2 = np.insert(gradient2, e, 0)

grad_current = gradient/gradient.max()
grad_opt = gradient2/gradient2.max()

# grad_current = gradient
# grad_opt = gradient2

ax2.bar(np.arange(grad_current.size)[grad_current>0], ((grad_opt-grad_current)/grad_current*100)[grad_current>0],
        edgecolor = 'b', label='Gradient change')
# ax2.legend(loc='upper right')
ax2.grid()
# ax2.xlim([-1, 201])
ax2.set_xlabel('Index of cavity')
ax2.set_ylabel('Change of gradient (\%)')

plt.tight_layout()
plt.savefig(path+'/compare_nl.eps', format="eps")
plt.savefig(path+'/compare_nl.pdf', format="pdf")
plt.savefig(path+'/compare_nl.png', format="png")
plt.show()