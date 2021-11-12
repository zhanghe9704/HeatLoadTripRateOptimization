import numpy as np
import random
from random import sample

def remove(dim, data, n_down):
    id_down = sample(range(dim),n_down)
    id_down.sort()
    dim -= n_down
    data = np.delete(data, id_down, 0)
    return dim, data, id_down,
