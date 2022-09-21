from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import pandas as pd
import copy

from pygmo import problem, population, unconstrain, bfe, member_bfe

import optimize.nsga_II as algo
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem
import user_problem.cebaf_dt_v1 as cav

# # Choose the linac here
linac = 'North' ## 'South' or 'North'
# random.seed(1)
random.seed()
# plt.interactive(False)

# create a folder under the current work directory to save data
path = savedata.folder.create('nsga_II_reconstruction')

# cavity table file
file = 'user_problem\\cavity_table.pkl'
# q curve file
# file_q = 'user_problem\\q_curves_'+linac.lower()+'.pkl'
file_q = ''

# cavities = pd.read_pickle(file)
# # Remove the constraints using death penalty
# if linac.upper() == 'SOUTH':
#     # lem_prob = lem.sl()
#     sl = cavities[cavities['cavity_id'].str.contains('2L')]
#     lem_prob = lem.prbl(sl)
# elif linac.upper() == 'NORTH':
#     # lem_prob = lem.nl()
#     nl = cavities[cavities['cavity_id'].str.contains('1L')]
#     lem_prob = lem.prbl(nl)


## Define the digitial twin    
cavities = cav.digitalTwin(file, file_q, linac)
# cavities = cav.digitalTwin(file, linac=linac)

# # Define the cryomodule
cryomodule = '1L06'
energy_constraint = 31.8
energy_margin = 0.2
cavities = cav.cryoModule(file, file_q, cryomodule, energy_constraint, energy_margin)

lem_prob = lem.prbl(cavities)
    
prob = problem(lem_prob)
print('orignal problem:')
print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
prob_dth = problem(unconstrain(prob, method='kuri')) #'death penalty','kuri', 'weighted', 'ignore_c', 'ignore_o'
print('death penalty removed:')
print(prob_dth)
#

# b = bfe(lem_prob.batch_fitness_gpu)
b = bfe(lem_prob.batch_fitness_cpu)
# b = bfe(member_bfe())  #member bfe not implemented for unconstrained problem

# Create initial population
pop_size = 128
pop = population(prob_dth)
dim = prob.get_nx()
x = np.empty(dim)
for _ in range(pop_size):
    lem_prob.create_pop_w_constr(x)
    pop.push_back(x)
print ("Initial pop generated!")

f = pop.get_f()
f0 = []
f1 = []
for e in f:
    f0.append(e[0])
    f1.append(e[1])
plt.figure()
plt.scatter(f0, f1, c='b', label=cavities.getName().upper()+' Initial Pop')
plt.ylabel("Trip Rate [per hour]")
plt.xlabel("Heat Load [W]")
plt.legend()
plt.grid()
plt.savefig(cavities.getName().upper()+'_nsga_II_gen_ip.eps', format="eps")
plt.show()
print("Initial pop plotted!")

# sys.exit()

# # # Find the gradients that only minimize the trip rate.
# x = lem_prob.trip_rate_opt()
# print(lem_prob.calc_energy(x))
# print(lem_prob.calc_heat_load(x))
# print(lem_prob.calc_number_trips(x))

# # # Adjust the gradients according to the giving limits. 
# x = lem_prob.adjust_gradient(x)
# print(lem_prob.calc_energy(x))
# print(lem_prob.calc_heat_load(x))
# print(lem_prob.calc_number_trips(x))


# # # Find the gradients that only minimize the heat load.
# x = lem_prob.heat_load_opt()
# print(lem_prob.calc_energy(x))
# print(lem_prob.calc_heat_load(x))
# print(lem_prob.calc_number_trips(x))

# # Run the optimizer for 30k generations and save the result
n_gen = [30000]
# # pop = algo.opt(pop, n_gen, path)
pop = algo.opt(pop, n_gen, path, b)
sav.save_pop('pop_nsga_II_'+str(n_gen[-1])+'_'+cavities.getName().lower(), pop)
print('pop_nsga_II_'+str(n_gen[-1])+'_'+cavities.getName().lower())

# # Load the saved result and plot
pop2 = population(prob_dth)
sav.load_pop('pop_nsga_II_'+str(n_gen[-1])+'_'+cavities.getName().lower(), pop2)
plt.figure()

f = pop2.get_f()
f0 = []
f1 = []
for e in f:
    f0.append(e[0])
    f1.append(e[1])

# print('nsga2: ', f0,f1)
plt.scatter(f0, f1, c='b', label=cavities.getName().upper()+' NSGA_II '+str(n_gen[-1]))
plt.ylabel("Trip Rate [per hour]")
plt.xlabel("Heat Load [W]")
plt.legend()
plt.grid()
# plt.ylim(ymin=0)
# plt.ylim(ymax=50)
# plt.xlim(xmin=0)
# plt.xlim(xmax=50)
plt.savefig(cavities.getName().upper()+'_nsga_II_gen_'+str(n_gen[-1])+'.eps', format="eps")
plt.show()


sys.exit()

# # Turn off a few cavities and reconstruct the pareto_front
n_off = 5

lem_prob_new = copy.deepcopy(lem_prob)

idx_off = lem.revise_problem(lem_prob_new, n_off)  # New problem with less cavities
# idx_off = lem.revise_problem(lem_prob_new,  [2, 10, 11, 33, 40, 60, 124, 186, 191, 199]) 
print('The following ', n_off, ' cavities are off: ', idx_off)

prob_new = problem(lem_prob_new)
print('problem with ', n_off, ' cavities off:')
print(prob_new)
prob_dth_new = problem(unconstrain(prob_new, method='death penalty'))  #'death penalty','kuri', 'weighted', 'ignore_c', 'ignore_o'
print('death penalty removed:')
print(prob_dth_new)

# # Load previous result for the original problem
pop_org = population(prob_dth)
sav.load_pop('pop_nsga_II_'+str(n_gen[-1])+'_'+cavities.getName().lower(), pop_org)
# # Delete the off cavities and use as initial population for the new problem
pop_new = population(prob_dth_new)
for ind in pop_org.get_x():
    x = np.delete(ind, idx_off)
    lem_prob_new.recreate_pop_dpdg_sqr(x)
    pop_new.push_back(x)
# # # We could also add a few individuals with low trip rate to lead the optimization to the low trip rate region
for i in range(4):
    x = lem_prob_new.trip_rate_opt(i*0.5)
    x = lem_prob_new.adjust_gradient(x)
    pop_new.push_back(x)
    
# # # We could also add a few individuals with low heat load to lead the optimization to the low heat load region   
# x = lem_prob_new.heat_load_opt()
# x = lem_prob_new.adjust_gradient(x)
# pop_new.push_back(x)

b = bfe(lem_prob_new.batch_fitness_cpu)
n_gen = [200, 500, 1000, 2000, 3000]
pop_new = algo.opt(pop_new, n_gen, path, b)

# # Add the off cavities back with zero gradients
pop_recon = population(prob_dth)
for ind in pop_new.get_x():
    for idx in idx_off:
        ind = np.insert(ind, idx, 0)
    pop_recon.push_back(ind)

cav_off = []
for i in idx_off:
    cav_off.append(lem_prob.cavity_id[i])
print(idx_off)
print(cav_off)