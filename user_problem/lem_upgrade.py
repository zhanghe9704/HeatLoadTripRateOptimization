import numpy as np
import cupy as cp
import random
from scipy import constants
import csv

from random import sample
from multipledispatch import dispatch

# class lem_upgrade(base):
class lem_upgrade:        
    def __init__(self, dim, c_dim, c_ineq_dim, cavities):
        self.cavity_id = cavities.list_cavities()
        self.Q = cavities.getValues('Q0')
        self.length = cavities.getValues('length')
        self.trip_slope = cavities.getValues('trip_slope')
        self.fault_grad = cavities.getValues('trip_offset')
        self.energy = cavities.getEnergyConstraint()
        self.cnst = cavities.getValues('shunt_impedence')
        # Set bounds
        self.range_low =cavities.getMinGradients()-1e-5
   
        # Use ops_gset_max for the uppler limits of the gradients
        # When ops_gset_max gives a nan, use the max_gset value.
        self.range_up = cavities.getValues('max_gset_to_use')
        self.energy_tol =cavities.getEnergyMargin()
        # self.length[self.length == 0.5] = constants.c/1.497e9*2.5
        # self.length[self.length > 0.6] = constants.c/1.497e9*3.5

        for idx, up in enumerate(self.range_up):
            if up < self.range_low[idx]:
                self.range_up[idx] = self.range_low[idx]
        self.number_trips = 0
        self.trip_max = trip_max
        self.c_dim = c_dim
        self.c_ineq_dim = c_ineq_dim
        self.heat_max = heat_max
        self.heat_load = 0
        self.A = -10.26813067
        
        self.c2 = cavities.getValues('c2')
        self.c1 = cavities.getValues('c1')
        self.c0 = cavities.getValues('c0')
        self.dim = len(self.cavity_id)
    
    # """
    # The following function initialize the class using a cavity table, which
    # has been replaced by a digital twin.
    # """
    # def __init__(self, dim, c_dim, c_ineq_dim, cavities):
    #         self.type = cavities['type']
    #         self.Q = cavities['Q0'].values
    #         self.length = cavities['length'].values
    #         self.trip_slope = cavities['trip_slope'].values
    #         self.fault_grad = cavities['trip_offset'].values
    #         self.lowest_grad = 3.0 - 1e-5
    #         self.energy = total_energy
    #         self.cnst = cavities['shunt_impedance'].values
    #         # Set bounds
    #         self.range_low = np.empty(dim)
    #         self.range_low.fill(self.lowest_grad)
    #         self.range_low[self.type == 'C75'] = 5.0 - 1e-5
    #         self.range_low[self.type == 'C100'] = 5.0 - 1e-5
            
    #         # Use ops_gset_max for the uppler limits of the gradients
    #         # When ops_gset_max gives a nan, use the max_gset value.
    #         up =  cavities['ops_gset_max'].copy(deep=True)
    #         up[up.isna()] = cavities.max_gset[cavities.ops_gset_max.isna()]
    #         self.range_up = up.values
    #         self.energy_tol =energy_tol
    #         # self.length[self.length == 0.5] = constants.c/1.497e9*2.5
    #         # self.length[self.length > 0.6] = constants.c/1.497e9*3.5

    #         for idx, up in enumerate(self.range_up):
    #             if up < self.range_low[idx]:
    #                 self.range_up[idx] = self.range_low[idx]
    #         # self.set_bounds(self.range_low, self.range_up)
    #         self.number_trips = 0
    #         self.trip_max = trip_max
    #         self.c_dim = c_dim
    #         self.c_ineq_dim = c_ineq_dim
    #         self.heat_max = heat_max
    #         self.heat_load = 0
    #         self.A = -10.26813067
        
    def fitness(self, xx):
        x = np.array(xx)
        diff_energy = np.fabs(np.sum(self.length * x) - self.energy)
        fault = self.trip_slope * (x - self.fault_grad)
        fault_rate = np.sum(np.exp(self.A + fault[self.trip_slope > 0]))
        number_trips = 3600.0 * fault_rate
        self.number_trips = number_trips
        
        x_sqr = x * x
        q = self.c2 * x_sqr + self.c1 * x + self.c0
        # q[q<=0] = 1e-41
        q[q<=0] = -1*q[q<=0]
        heat_load = np.sum(1e12 * (x_sqr) * self.length / (q * self.cnst))
        self.heat_load = heat_load
        obj = [self.heat_load, self.number_trips]
        ci = []
        if c_dim == 1:
            ci = [diff_energy - self.energy_tol]
        elif c_dim == 2:
            ci = [diff_energy - self.energy_tol, self.number_trips - self.trip_max]
        elif c_dim == 3:
            ci = [diff_energy - self.energy_tol, self.number_trips - self.trip_max, self.heat_load - self.heat_max]
        else:
            print("Warning: Constraint number should be ONE, TWO, or THREE!")
        return obj + ci
 
    def batch_fitness(self, dvs) :
        # print('Calling bfe!')
        lv = len(self.length)
        ldvs = len(dvs)
        lp = int(ldvs/lv)
        x = np.array(dvs).reshape(lp, lv)
        diff_energy = np.fabs(np.sum(self.length*x,axis=1)-self.energy)
        
        death = diff_energy - self.energy_tol
        
        fault = self.trip_slope * (x - self.fault_grad)   
        fault_rate = np.sum(np.exp(self.A + fault[:,self.trip_slope > 0]),axis=1)
        number_trips = 3600.0 * fault_rate
        
        x_sqr = x * x
        q = self.c2 * x_sqr + self.c1 * x + self.c0
        q = np.fabs(q)
        heat_load = np.sum(1e12 * (x_sqr) * self.length / (q * self.cnst) ,axis=1)
        
        mask = death>0
        number_trips[mask] = 1e6
        heat_load[mask] = 1e6
               
        obj = [heat_load, number_trips]

        res = np.array(obj).T.reshape(lp*2)
        return res
    
    def batch_fitness_cpu(self, prob, dvs) :
        # print('Calling bfe!')
        lv = len(self.length)
        ldvs = len(dvs)
        lp = int(ldvs/lv)
        x = np.array(dvs).reshape(lp, lv)
        diff_energy = np.fabs(np.sum(self.length*x,axis=1)-self.energy)
        
        death = diff_energy - self.energy_tol
        
        fault = self.trip_slope * (x - self.fault_grad)   
        fault_rate = np.sum(np.exp(self.A + fault[:,self.trip_slope > 0]),axis=1)
        number_trips = 3600.0 * fault_rate
        
        x_sqr = x * x
        q = self.c2 * x_sqr + self.c1 * x + self.c0
        q = np.fabs(q)
        heat_load = np.sum(1e12 * (x_sqr) * self.length / (q * self.cnst) ,axis=1)
        
        mask = death>0
        number_trips[mask] = 1e6
        heat_load[mask] = 1e6
               
        obj = [heat_load, number_trips]

        res = np.array(obj).T.reshape(lp*2)

        return res    
    
    def calc_fitness(self, x):
        diff_energy = np.fabs(np.sum(self.length*x,axis=1)-self.energy)   
        death = diff_energy - self.energy_tol
               
        fault = self.trip_slope * (x - self.fault_grad)   
        fault_rate = np.sum(np.exp(self.A + fault[:,self.trip_slope > 0]),axis=1)
        number_trips = 3600.0 * fault_rate
        
        x_sqr = x * x
        q = self.c2 * x_sqr + self.c1 * x + self.c0
        q = np.fabs(q)
        heat_load = np.sum(1e12 * (x_sqr) * self.length / (q * self.cnst) ,axis=1)
        return death, heat_load, number_trips
    
    def calc_fitness_gpu(self, xx):
        
        x = cp.asarray(xx.astype('float32'))
        length = cp.asarray(self.length.astype('float32'))
        energy = cp.asarray(self.energy, dtype='float32')
        energy_tol = cp.asarray(self.energy_tol, dtype='float32')
        
        death = cp.abs(cp.sum(length*x,axis=1)-energy) - energy_tol
        death_cpu = cp.asnumpy(death).astype('float64')
        
        
        # diff_energy = np.fabs(np.sum(self.length*x,axis=1)-self.energy)   
        # death = diff_energy - self.energy_tol
        
        trip_slope = cp.asarray(self.trip_slope.astype('float32'))
        fault_grad = cp.asarray(self.fault_grad.astype('float32'))
        A = cp.asarray(self.A,dtype='float32')
        
        fault =  trip_slope * (x -  fault_grad)   
        fault_rate = cp.sum(cp.exp(A + fault[:,trip_slope > 0]),axis=1)
        number_trips = 3600.0 * fault_rate
        number_trips_cpu = cp.asnumpy(number_trips).astype('float64')
               
        # fault = self.trip_slope * (x - self.fault_grad)   
        # fault_rate = np.sum(np.exp(self.A + fault[:,self.trip_slope > 0]),axis=1)
        # number_trips = 3600.0 * fault_rate
        
        c2 = cp.asarray(self.c2.astype('float32'))
        c1 = cp.asarray(self.c1.astype('float32'))
        c0 = cp.asarray(self.c0.astype('float32'))
        cnst = cp.asarray(self.cnst.astype('float32'))
        x_sqr = x * x
        q = cp.abs(c2 * x_sqr + c1 * x + c0)
        heat_load = cp.sum(1e12 * (x_sqr) * length / (q * cnst) ,axis=1)
        heat_load_cpu = cp.asnumpy(heat_load).astype('float64')
        
        # x_sqr = x * x
        # q = self.c2 * x_sqr + self.c1 * x + self.c0
        # q = np.fabs(q)
        # heat_load = np.sum(1e12 * (x_sqr) * self.length / (q * self.cnst) ,axis=1)
        return death_cpu, heat_load_cpu, number_trips_cpu
    
    
    def batch_fitness_gpu(self, prob, dvs) :
        lv = len(self.length)
        ldvs = len(dvs)
        lp = int(ldvs/lv)
        x = np.array(dvs).reshape(lp, lv)
                
        death, heat_load, number_trips = self.calc_fitness_gpu(x)
        
        mask = death>0
        number_trips[mask] = 1e6
        heat_load[mask] = 1e6
        
        obj = [heat_load, number_trips]
        
        res = np.array(obj).T.reshape(lp*2)
        return res
        
    
    def has_batch_fitness(self):
        return True
        
    def get_nobj(self):
        return 2
    
    def get_bounds(self):
        return (self.range_low, self.range_up)
    
    def get_nic(self):
        return c_ineq_dim
    
    def get_nec(self):
        return c_dim - c_ineq_dim
    
    def get_nc(self):
        return c_dim
    
    def calc_number_trips(self, xx):
        x = np.array(xx)
        diff_energy = np.fabs(np.sum(self.length * x) - self.energy)
        fault = self.trip_slope * (x - self.fault_grad)
        fault_rate = np.sum(np.exp(self.A + fault[self.trip_slope > 0]))
        number_trips = 3600.0 * fault_rate
        return number_trips

    def calc_heat_load(self, xx):
        x = np.array(xx)
        x_sqr = x * x
        q = self.c2 * x_sqr + self.c1 * x + self.c0
        # q[q<=0] = 1e-41
        q[q<=0] = -1*q[q<=0]
        heat_load = np.sum(1e12 * x_sqr * self.length / (q * self.cnst))
        return heat_load
    
    def calc_heat_load_const_q(self, xx):
        x = np.array(xx)
        x_sqr = x * x
        heat_load = np.sum(1e12 * x_sqr * self.length / (self.Q * self.cnst))
        return heat_load

    def calc_energy(self, xx):
        x = np.array(xx)
        return  np.sum(self.length * x)
    
    def n_cavity(self):
        return self.Q.size

    def __len__(self):
        return self.Q.size

    def trip_rate_opt(self, delta_E = 0):
        x = self.range_up * 1
        idx = self.trip_slope>0
        lb = self.length[idx]/self.trip_slope[idx]
        lb_sum = np.sum(lb)
        dE = self.energy - np.sum(x[np.logical_not(idx)]*self.length[np.logical_not(idx)]) + delta_E
        lm = np.exp((dE - np.sum(lb*np.log(lb/3600.0)) + self.A * lb_sum - np.sum(self.fault_grad[idx]*self.length[idx]))/lb_sum)
        x[idx] = (np.log(lm*lb/3600.0) - self.A)/self.trip_slope[idx] + self.fault_grad[idx]
        return  x


    def heat_load_opt(self):
        lm = 2 * self.energy / np.sum(self.Q * self.cnst * self.length)
        x = 0.5* lm * self.cnst * self.Q
        return x.tolist()

    # Prepare the population
    def create_pop(self, x):
        for i, xi in enumerate(x):
            x[i] = (self.range_up[i] - self.range_low[i]) * random.random() + self.range_low[i]
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            rate = 1 + (self.energy - current_energy) / current_energy
            if (rate > 1):
                for i, xi in enumerate(x):
                    x[i] = x[i] * rate
                    if (x[i] > self.range_up[i]):
                        x[i] = self.range_up[i] * (1 - 1e-6)
            if (rate < 1):
                for i, xi in enumerate(x):
                    x[i] = x[i] * rate
                    if (x[i] < self.range_low[i]):
                        x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def create_pop_w_constr(self, x): 
        self.create_pop(x)
        if self.c_dim == 2:
            while self.calc_number_trips(x)>self.trip_max:
                self.create_pop(x)
        elif self.c_dim == 3:
            while self.calc_number_trips(x)>self.trip_max and self.calc_heat_load(x)>self.heat_max:
                self.create_pop(x)
            
        
    def recreate_pop(self, x, idx=-1):
        if (idx >= 0):
            if (x[idx] > self.range_low[idx]):
                if (0.95 * x[idx] > self.range_low[idx]):
                    x[idx] *= 0.95
                else:
                    x[idx] = self.range_low[idx]
            else:
                if (1.05 * x[idx] < self.range_up[idx]):
                    x[idx] *= 1.05
                else:
                    x[idx] = self.range_up[idx]
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            if (idx >= 0):
                rate = 1 + (self.energy - current_energy) / (current_energy - self.length[idx] * x[idx])
            else:
                rate = 1 + (self.energy - current_energy) / current_energy
            if (rate > 1):
                for i, xi in enumerate(x):
                    if (i != idx):
                        x[i] = x[i] * rate
                        if (x[i] > self.range_up[i]):
                            x[i] = self.range_up[i] #* (1 - 1e-6)
            if (rate < 1):
                for i, xi in enumerate(x):
                    if (i != idx):
                        x[i] = x[i] * rate
                        if (x[i] < self.range_low[i]):
                            x[i] = self.range_low[i] #* (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def recreate_pop_dpdg(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst*self.Q) / (self.length*x*x)
            coef_g = np.abs(coef_g)
            # coef_g = coef_g*coef_g
            coef_g = 1/coef_g
            k = energy_change / np.sum(coef_g*self.length)
            for i, xi in enumerate(x):
                x[i] += coef_g[i]*k
                if (x[i] > self.range_up[i]):
                    x[i] = self.range_up[i] * (1 - 1e-6)
                elif (x[i] < self.range_low[i]):
                    x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def recreate_pop_dpdg_sqr(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst * self.Q) / (self.length * x * x)
            coef_g = np.abs(coef_g)
            coef_g = coef_g*coef_g
            coef_g = 1 / coef_g
            k = energy_change / np.sum(coef_g * self.length)
            for i, xi in enumerate(x):
                x[i] += coef_g[i] * k
                if (x[i] > self.range_up[i]):
                    x[i] = self.range_up[i] * (1 - 1e-6)
                elif (x[i] < self.range_low[i]):
                    x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)


    def recreate_pop_dpdq_sort(self, x):
        current_energy = np.sum(self.length * x)
        energy_diff = np.fabs(current_energy - self.energy)
        coef_g = (self.cnst * self.Q) / (self.length * x)
        coef_g = np.abs(coef_g)
        index = np.argsort(coef_g)
        i = 0
        while (energy_diff > 2 and i < len(index)):
            energy_inc = (self.range_up[index[i]] - x[index[i]]) * self.length[index[i]]
            if (energy_inc < energy_diff):
                x[index[i]] = self.range_up[index[i]]
            else:
                energy_inc = energy_diff
                x[index[i]] += energy_inc/self.length[index[i]]
            energy_diff -= energy_inc
            i += 1
    def recreate_pop_dpdq_cute(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst*self.Q) / (self.length*x)
            coef_g = coef_g * coef_g * coef_g
            coef_g = 1/coef_g
            k = energy_change / np.sum(coef_g*self.length)
            for i, xi in enumerate(x):
                x[i] += coef_g[i]*k
                if (x[i] > self.range_up[i]):
                    x[i] = self.range_up[i] * (1 - 1e-6)
                elif (x[i] < self.range_low[i]):
                    x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def adjust_gradient(self,xx):
        x = np.array(xx)
        x[x>self.range_up] = self.range_up[x>self.range_up] * (1 - 1e-6)
        x[x<self.range_low] = self.range_low[x<self.range_low]  * (1 + 1e-6)
        self.recreate_pop_dpdg_sqr(x)
        return x.tolist()
    
    def cavity_list(self):
        return self.cavity_id
    
    def dim(self):
        return self.dim
    
    def remove_cavities(self, idx):
        self.cavity_id = np.delete(self.cavity_id, idx)
        self.Q = np.delete(self.Q, idx)
        self.length = np.delete(self.length, idx)
        self.trip_slope = np.delete(self.trip_slope, idx)
        self.fault_grad = np.delete(self.fault_grad, idx)
        self.cnst = np.delete(self.cnst, idx)
        self.range_low =np.delete(self.range_low, idx)
        self.range_up = np.delete(self.range_up, idx)   
        self.c2 = np.delete(self.c2, idx)
        self.c1 = np.delete(self.c1, idx)
        self.c0 = np.delete(self.c0, idx)
        self.dim = len(self.cavity_id)
        
@dispatch(lem_upgrade, int)
def revise_problem(prob, n_down):
    id_down = sample(range(prob.dim),n_down)
    id_down.sort()
    prob.remove_cavities(id_down)
    return id_down

@dispatch(lem_upgrade, list)
def revise_problem(prob, n_down):
    id_down = n_down
    id_down.sort()
    prob.remove_cavities(id_down)
    return id_down


def prbl(cav):
    prbl = lem_upgrade(cav.getTotalCavityNumber(), c_dim, c_ineq_dim, cav)
    return prbl

# filename = ""
total_energy = 1050
dim = 200
c_dim = 2
c_ineq_dim = c_dim
trip_max = 4000
heat_max = 3400
data = []
lines = []
energy_tol = 2

# nl = lem_upgrade()

if __name__ == '__main__':
    print('Define the user_problems')


