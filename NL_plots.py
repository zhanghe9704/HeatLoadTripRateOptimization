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


# set global settings
def init_plotting():
    plt.rcParams['figure.figsize'] = (4, 4)  # 1/4 of line width
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'CM Sans Serif'
    #    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    #    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['axes.linewidth'] = 1
    #    plt.gca().spines['right'].set_color('none')
    #    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

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
# plt.figure()
init_plotting()
cur_f = np.array([ind.cur_f for ind in pop]).T
plt.scatter(cur_f[0], cur_f[1], c='b', label='North Linac')
plt.ylim(ymin=0, ymax=28)
plt.xlabel('Heat load [W]')
plt.ylabel('Trip rate [per hour]')
plt.legend(loc="upper right")
plt.grid()
plt.savefig(path+'/NL_nsga_II_gen_30k.eps', format="eps")
plt.show()