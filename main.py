from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import pandas as pd

from pygmo import problem, population, unconstrain

import optimize.nsga_II as algo
# import optimize.nspso as algo
# import optimize.sms_emoa as algo
# import optimize.spea2 as algo
# import algorithm_race.nsga_II as nsga_race
# import algorithm_race.spea2 as spea2_race
# import algorithm_race.sms_emoa as sms_emoa_race
# import algorithm_race.nspso as nspso_race
# import analysis.compare as cp
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem

# # Choose the linac here
linac = 'SL' ## 'NL' or 'SL'
random.seed()
# plt.interactive(False)

# create a folder under the current work directory to save data
path = savedata.folder.create('nsga_II_reconstruction')

# cavity table file
file = 'data_prepare\\cebaf_cavity_table\\cavity_table.pkl'
cavities = pd.read_pickle(file)

# Remove the constraints using death penalty
if linac.upper() == 'SL':
    # lem_prob = lem.sl()
    sl = cavities[cavities['cavity_id'].str.contains('2L')]
    lem_prob = lem.prbl(sl)
elif linac.upper() == 'NL':
    # lem_prob = lem.nl()
    nl = cavities[cavities['cavity_id'].str.contains('1L')]
    lem_prob = lem.prbl(nl)
    
prob = problem(lem_prob)
print('orignal problem:')
print(prob)
# prob_dth = problem.death_penalty(prob, problem.death_penalty.method.KURI)
prob_dth = unconstrain(prob, method='kuri')  #'death penalty','kuri', 'weighted', 'ignore_c', 'ignore_o'
print('death penalty removed:')
print(prob_dth)
#
# Create original population
pop_size = 128
pop = population(prob_dth)
dim = lem.dim
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
plt.scatter(f0, f1, c='b', label=linac.upper()+' Initial Pop')
plt.legend()
plt.grid()
plt.savefig(linac.upper()+'_nsga_II_gen_ip.eps', format="eps")
plt.show()
print("Initial pop plotted!")

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
pop = algo.opt(pop, n_gen, path)
sav.save_pop('pop_nsga_II_30k.'+linac.lower(), pop)

# # Load the saved result and plot
pop2 = population(prob_dth)
sav.load_pop('pop_nsga_II_30k.'+linac.lower(), pop2)
plt.figure()

f = pop2.get_f()
f0 = []
f1 = []
for e in f:
    f0.append(e[0])
    f1.append(e[1])
plt.scatter(f0, f1, c='b', label=linac.upper()+' NSGA_II 30000')
plt.legend()
plt.grid()
plt.savefig(linac.upper()+'_nsga_II_gen_30k.eps', format="eps")
plt.show()


# sys.exit()

# # Turn off a few cavities and reconstruct the pareto_front
n_off = 5
lem_prob, idx_off = lem.revise_problem(n_off)  # New problem with less cavities
print('The following ', n_off, ' cavities are off: ', idx_off)
prob_new = problem(lem_prob)
print('problem with ', n_off, ' cavities off:')
print(prob_new)
prob_dth_new = unconstrain(prob_new, method='kuri')  #'death penalty','kuri', 'weighted', 'ignore_c', 'ignore_o'
print('death penalty removed:')
print(prob_dth_new)

# # Load previous result for the original problem
pop_org = population(prob_dth)
sav.load_pop('pop_nsga_II_30k.'+linac.lower(), pop_org)
# # Delete the off cavities and use as initial population for the new problem
pop_new = population(prob_dth_new)
for ind in pop_org.get_x():
    x = np.delete(ind, idx_off)
    lem_prob.recreate_pop_dpdg_sqr(x)
    pop_new.push_back(x)
# # # We could also add a few individuals with low trip rate to lead the optimization to the low trip rate region
# for i in range(4):
#     x = lem_prob.trip_rate_opt(i*0.5)
#     x = lem_prob.adjust_gradient(x)
#     pop_new.push_back(x)
    
# # # We could also add a few individuals with low heat load to lead the optimization to the low heat load region   
# x = lem_prob.heat_load_opt()
# x = lem_prob.adjust_gradient(x)
# pop_new.push_back(x)

n_gen = [200, 500, 1000, 2000, 3000]
pop_new = algo.opt(pop_new, n_gen, path)

# # Add the off cavities back with zero gradients
pop_recon = population(prob_dth)
for ind in pop_new.get_x():
    for idx in idx_off:
        ind = np.insert(ind, idx, 0)
    pop_recon.push_back(ind)
