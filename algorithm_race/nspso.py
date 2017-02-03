from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time

from PyGMO import algorithm

from savedata.record_pop import save_pop


def race(pop, ngen, c, v, chi, path):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure()

    cnt = 0
    for _, ci in enumerate(c):
        for _, vi in enumerate(v):
            for _, chii in enumerate(chi):
                print(ci, vi, chii)
                pop_race = pop
                algo = algorithm.nspso(gen=ngen, C1=ci, C2=ci, CHI=chii, v_coeff=vi)
                pop_race = algo.evolve(pop_race)
                save_pop(path + '/pop_nspso_' + str(ngen)+'_c_'+'{:.3f}'.format(ci)+'_v_'+'{:.3f}'.format(vi)+'_chi_'+
                         '{:.3f}'.format(chii), pop_race)
                cur_f = np.array([ind.cur_f for ind in pop_race]).T
                plt.scatter(cur_f[0], cur_f[1], c=color[cnt % 6],
                            label='c = ' + '{:.1f}'.format(ci) + ', v = ' + '{:.3f}'.format(vi) +
                                  ', chi = ' + '{:.1f}'.format(chii))
                plt.legend()
                cnt += 1

    plt.ylim(0,40)
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.title('nspso, '+str(ngen)+' generations')
    plt.ylabel("Trip Rate [per hour]")
    plt.xlabel("Heat Load [W]")
    plt.savefig(path+'/'+'nspso_race_ngen_'+str(ngen)+'_'+date_str+'.eps', format="eps")
    plt.show()
    return pop