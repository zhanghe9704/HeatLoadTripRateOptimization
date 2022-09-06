from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time

from pygmo import algorithm, spea2

from savedata.record_pop import save_pop

def opt(pop, ngen, path, *arg):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    n_gen = [0] + ngen
    t = np.zeros(len(n_gen))
    # filename = "pop"+str(n_gen[0])

    plt.figure()
    for i in range(len(n_gen)-1):
        algo = algorithm(spea2(n_gen[i + 1] - n_gen[i], m=0.06, cr=0.6))
        if len(arg)>0:
            uda = algo.extract(spea2)
            uda.set_bfe(arg[0])
        start = time.time()
        pop = algo.evolve(pop)
        end = time.time()
        save_pop(path + '/pop_spea2_' + str(n_gen[i + 1]), pop)
        t[i+1] = end - start + t[i]
        # cur_f = np.array([ind.cur_f for ind in pop]).T
        # np.save('cur_f_nsga_II_'+str(n_gen[i+1]), cur_f)
        f = pop.get_f()
        f0 = []
        f1 = []
        for e in f:
            f0.append(e[0])
            f1.append(e[1])
        plt.scatter(f[0], f[1], c=color[i%6],
                    label = 'n_gen = '+str(n_gen[i+1])+', t='+str(int(t[i+1]))+'s')
        plt.legend()
    # np.save('t_nsga_II', t)
    #plt.ylim(0,20)
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.title('spea2')
    plt.ylim(ymin=0)
    if (f[1].max() > 50):
        plt.ylim(ymax=50)
    plt.grid()
    plt.ylabel("Trip Rate [per hour]")
    plt.xlabel("Heat Load [W]")
    plt.savefig(path+'/'+'spea2_'+date_str+'.eps', format="eps")
    # plt.show()
    return pop