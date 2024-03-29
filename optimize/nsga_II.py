from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time

from pygmo import algorithm, nsga2

from savedata.record_pop import save_pop


def opt(pop, ngen, path, *arg):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if (isinstance(ngen, np.ndarray)):
        ngen = ngen.tolist()
    n_gen = [0] + ngen
    # print n_gen
    t = np.zeros(len(n_gen))
    # filename = "pop"+str(n_gen[0])
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.figure()
    for i in range(len(n_gen)-1):
        # algo = algorithm.nsga_II(n_gen[i+1]-n_gen[i], m=0.01, cr = 0.95)
        algo = algorithm(nsga2(n_gen[i+1]-n_gen[i], m=0.01, cr = 0.95))
        if len(arg)>0:
            uda = algo.extract(nsga2)
            uda.set_bfe(arg[0])
        start = time.time()
        pop = algo.evolve(pop)
        end = time.time()
        save_pop(path+'/pop_nsga_II_'+str(n_gen[i+1]), pop)
        save_pop(path + '/pop_nsga_II_' + str(n_gen[i + 1]) +'_' + date_str, pop)
        t[i+1] = end - start + t[i]
        # cur_f = np.array([ind.cur_f for ind in pop]).T

        # # np.save('cur_f_nsga_II_'+str(n_gen[i+1]), cur_f)
        # plt.scatter(cur_f[0], cur_f[1], c=color[i%6],
        #             label = 'n_gen = '+str(n_gen[i+1])+', t='+str(int(t[i+1]))+'s')
        
        f = pop.get_f()
        f0 = []
        f1 = []
        for e in f:
            f0.append(e[0])
            f1.append(e[1])
        plt.scatter(f0, f1, c=color[i%6],
                    label = 'n_gen = '+str(n_gen[i+1])+', t='+str(int(t[i+1]))+'s')
        plt.legend()
    # np.save('t_nsga_II', t)
    # plt.ylim(ymin=0)
    # if (cur_f[1].max()>50):
    #     plt.ylim(ymax=50)
    if (max(f1)>50):
        plt.ylim(ymax=50)
    plt.grid()

    # plt.title('nsga_II')
    plt.ylabel("Trip Rate [per hour]")
    plt.xlabel("Heat Load [W]")
    plt.savefig(path+'/'+'nsga_II_'+date_str+'.eps', format="eps")
    plt.show()
    return pop