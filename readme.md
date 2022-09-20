# Introduction

## About this package

This package includes the python files that optimize the heat load and the trip rate of CEBAF RF system simultaneously using genetic algorithm (GA) from the pygmo lib. The heat load and the trip rate are two competing objects hence a pareto front curve will be obtained, on which any point, representing a setup of the RF cavities,  is not inferior to any other points. This package can also reconstruct the pareto front efficiently when some cavities are offline, starting from the pareto front with all the cavities running. 

## List of files

```
.
├── optimize/
│   ├── __init__.py
│   └── nsga_II.py
├── savedata/
│   ├── __init__.py
│   ├── folder.py
│   └── record_pop.py
├── user_problem/
│   ├── __init__.py
│   ├── cebaf_dt_v1.py
│   └── lem_upgrad.py
└── main.py
```

optimize/nsga_II.py - This is a wrapped nsga_II optimizer from pygmo. One important difference comparing with the original one is this one takes a list of numbers for the number of generations instead of just one number. It plots the pareto-front for each of the number in the list ans shows each time cost in the legend of the plot. 

savedata/folder.py - Create a folder with time stamp and optimizer name.

savedata/record_pop.py - Save and load a population.

user_problem/cebaf_dt_v1.py - Digital twin of CEBAF with either constant Q0s or  Q0 functions as 2nd order polynomials. 

user_problem/lem_upgrad.py - Define the problem. 

## How to use the code

In the following we will explain how to use the code taking the *main.py* file as an example. 

### 0. Import libs

```
from pygmo import problem, population, unconstrain, bfe, member_bfe

import optimize.nsga_II as algo
import savedata.folder
import savedata.record_pop as sav
import user_problem.lem_upgrade as lem
import user_problem.cebaf_dt_v1 as cav
```

### 1. Create the linac

First we need to select the "North" linac or the "South" linac. Then we need to tell the name of the file that saves the cavity parameters and the name of the file that saves the q curves. With these three arguments, 

```
# Choose the linac here
linac = 'North' ## 'South' or 'North'

# cavity table file
file = 'user_problem\\cavity_table.pkl
# q curve file
file_q = 'user_problem\\q_curves_'+linac.lower()+'.pkl'

## Define the digital twin    
cavities = cav.digitalTwin(file, file_q, linac)

```

To define a digital twin with constant Q0, we can using the following code:

```
# Use empty string as the q curve file name 
file_q = ''

## Define the digital twin    
cavities = cav.digitalTwin(file, file_q, linac)
```

Or just omit the argument and set the following argument by name:

```
## Define the digital twin    
cavities = cav.digitalTwin(file, linac = linac)
```



### 2. Create the problem

In the following we create three different problems. First we create an user-defined problem (defined in lem_upgrade.py)  using the digital twin. Note that the digital is only used here. It can be disgarded once the problem is created. Then, using the user-defined problem, we can create a *pygmo* problem that a *pygmo* optimizer can work with. Specifically, the NSGA-II optimizer treats an unconstrainted problem, so we removed the constraints from the problem and create an unconstrainted one for the optimizer.  

```
# Create the user-defined problem object using the digital twin
lem_prob = lem.prbl(cavities)

# Create the pygmo problem for the optimizer     
prob = problem(lem_prob)

# Remove the constraints for the nsga_II optimizer 
prob_dth = problem(unconstrain(prob, method='death penalty')) b)
```

### 3. Run optimization

To run the optimization, a population including 128 individuals are created. The optimizer runs for 30,000 generations and the result is saved. 

```
# Create initial population
pop_size = 128
pop = population(prob_dth)
dim = prob.get_nx()
x = np.empty(dim)
for _ in range(pop_size):
    lem_prob.create_pop_w_constr(x)
    pop.push_back(x)

# Run the optimizer for 30k generations and save the result
n_gen = [30000]
pop = algo.opt(pop, n_gen, path)
sav.save_pop('pop_nsga_II_'+str(n_gen[-1])+'_'+
    cavities.getName().lower(), pop)
```

Computation of the fitness function is the bottleneck for efficiency. We can use a batch fitness function to reduce the computation time. In the following, a batch fitness function that carries out the calculation on GPU using CUPY is passed to the optimizer.

```
b = bfe(lem_prob.batch_fitness_gpu)  # Batch fitness function on GPU
pop = algo.opt(pop, n_gen, path, b)
```

### 4. Turn off a few cavities and reconstruct the pareto-front

Sometimes, not all the cavities are available. We could reconstruct the pareto-front when we have less cavities to use. The following example shows when a given number cavities are removed randomly from the linac, how we revise the original problem to create the new problem and rerun the optimization using the original pareto-front as the initial population. At the end, we put the cavities back and give zeros for their gradients.  

```
# Turn off a few cavities and reconstruct the pareto_front
n_off = 5

# Make a copy of the original user-defined problem
lem_prob_new = copy.deepcopy(lem_prob)

# Remove cavities from the problem and returns the index of 
# removed cavities.
idx_off = lem.revise_problem(lem_prob_new, n_off)  

# Create pygmo problem for the optimizer
prob_new = problem(lem_prob_new)
prob_dth_new = problem(unconstrain(prob_new, method='death penalty')) 

# Load previous result for the original problem
pop_org = population(prob_dth)
sav.load_pop('pop_nsga_II_'+str(n_gen[-1])+'_'
    +cavities.getName().lower(), pop_org)

# Delete the off cavities and use as initial population 
# for the new problem
pop_new = population(prob_dth_new)
for ind in pop_org.get_x():
    x = np.delete(ind, idx_off)
    lem_prob_new.recreate_pop_dpdg_sqr(x)
    pop_new.push_back(x)

# Run optimization up to 3000 generations
b = bfe(lem_prob_new.batch_fitness_cpu)
n_gen = [200, 500, 1000, 2000, 3000]
pop_new = algo.opt(pop_new, n_gen, path, b)

# Add the off cavities back with zero gradients
pop_recon = population(prob_dth)
for ind in pop_new.get_x():
    for idx in idx_off:
        ind = np.insert(ind, idx, 0)
    pop_recon.push_back(ind)
```
