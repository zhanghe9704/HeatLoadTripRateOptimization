import numpy as np
import random

def remove(dim, data, n_down):
    id_down = np.array([random.randint(0, dim - 1) for _ in range(n_down)])
    while (len(id_down) != len(np.unique(id_down))):
        id_down = np.array([random.randint(0, dim - 1) for _ in range(n_down)])

        #    print id_down
    dim -= len(id_down)
    data = np.delete(data, id_down, 0)
    return dim, data, id_down,
