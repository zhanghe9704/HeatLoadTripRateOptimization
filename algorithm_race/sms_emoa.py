from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time

from PyGMO import algorithm

from savedata.record_pop import save_pop


def race(pop, ngen, m, cr, path):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure()

    cnt = 0
    for _, mi in enumerate(m):
        for _, cri in enumerate(cr):
            print(mi, cri)
            pop_race = pop
            algo = algorithm.sms_emoa(gen=ngen, m=mi, cr=cri)
            pop_race = algo.evolve(pop_race)
            save_pop(path + '/pop_sms_emoa_' + str(ngen)+'_m_'+'{:.3f}'.format(mi)+'_cr_'+'{:.3f}'.format(cri), pop_race)
            cur_f = np.array([ind.cur_f for ind in pop_race]).T
            plt.scatter(cur_f[0], cur_f[1], c=color[cnt % 6],
                        label='m = ' + '{:.3f}'.format(mi) + ', cr = ' + '{:.2f}'.format(cri))
            plt.legend()
            cnt += 1

    # plt.ylim(0,15)
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.title('sms_emoa, '+str(ngen)+' generations')
    plt.ylabel("Trip Rate [per hour]")
    plt.xlabel("Heat Load [W]")
    plt.savefig(path+'/'+'sms_emoa_race_ngen_'+str(ngen)+'_'+date_str+'.eps', format="eps")
    plt.show()
    return pop