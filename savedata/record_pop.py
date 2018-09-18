try:
    import cPickle as pickle
except ImportError:
    import pickle

from PyGMO import population


def save_pop(file_name, pop):
    pop_list = []
    for ind in pop:
        pop_list = pop_list + [ind]
    with open(file_name, 'wb') as output:
        pickle.dump(pop_list, output)


def load_pop(file_name, pop):
    with open(file_name, 'rb') as input:
        pop_list = pickle.load(input)
    for ind in pop_list:
        pop.push_back(ind.best_x)