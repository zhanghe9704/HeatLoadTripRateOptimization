import numpy as np
import random
from scipy import constants
import csv

# from pygmo.problem import base

from . import cavities_off


# class lem_upgrade(base):
class lem_upgrade:
    def __init__(self, dim, c_dim, c_ineq_dim):
    # def __init__(self):
    #     super(lem_upgrade, self).__init__(dim, 0, 2, c_dim, c_ineq_dim, 0)
        # self.d2 = data[:, 1]
        self.Q = data[:, 5]
        self.length = data[:, 6]
        self.trip_slope = data[:, 4]
        self.fault_grad = data[:, 3]
        self.lowest_grad = 3.0-1e-5
        self.energy = total_energy
        self.cnst = np.empty(dim)
        self.cnst.fill(968.0)
        self.cnst[self.length == 0.5] = 960.0
        # Set bounds
        self.range_low = np.empty(dim)
        self.range_low.fill(self.lowest_grad)
        self.range_up = data[:, 1]
        self.energy_tol =energy_tol
        # self.length[self.length == 0.5] = constants.c/1.497e9*2.5
        # self.length[self.length > 0.6] = constants.c/1.497e9*3.5

        for idx, up in enumerate(self.range_up):
            if up < self.lowest_grad:
                self.range_up[idx] = self.lowest_grad
        # self.set_bounds(self.range_low, self.range_up)
        self.number_trips = 0
        self.trip_max = trip_max
        self.c_dim = c_dim
        self.c_ineq_dim = c_ineq_dim
        self.heat_max = heat_max
        self.heat_load = 0
        self.A = -10.26813067
        
    def fitness(self, xx):
        x = np.array(xx)
        diff_energy = np.fabs(np.sum(self.length * x) - self.energy)
        fault = self.trip_slope * (x - self.fault_grad)
        fault_rate = np.sum(np.exp(self.A + fault[self.trip_slope > 0]))
        number_trips = 3600.0 * fault_rate
        self.number_trips = number_trips
        heat_load = np.sum(1e12 * (x * x) * self.length / (self.Q * self.cnst))
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

    def get_nobj(self):
        return 2
    
    def get_bounds(self):
        return (self.range_low, self.range_up)
    
    def get_nic(self):
        return c_ineq_dim
    
    def get_nec(self):
        return c_dim - c_ineq_dim
    # # Define the objective function
    # def _objfun_impl(self, xx):
    #     # x = np.asarray(xx)
    #     # fault = self.trip_slope * (x - self.fault_grad)
    #     # fault_rate = np.sum(np.exp(-10.26813067 + fault[self.trip_slope > 0]))
    #     # number_trips = 3600.0 * fault_rate
    #     # self.number_trips = number_trips
    #     # heat_load = np.sum(1e12 * (x * x) * self.length / (self.Q * self.cnst))
    #     return self.heat_load, self.number_trips,

    # # Define the constraint functions
    # def _compute_constraints_impl(self, xx):
    #     x = np.array(xx)
    #     diff_energy = np.fabs(np.sum(self.length * x) - self.energy)
    #     fault = self.trip_slope * (x - self.fault_grad)
    #     fault_rate = np.sum(np.exp(self.A + fault[self.trip_slope > 0]))
    #     number_trips = 3600.0 * fault_rate
    #     self.number_trips = number_trips
    #     heat_load = np.sum(1e12 * (x * x) * self.length / (self.Q * self.cnst))
    #     self.heat_load = heat_load
    #     constr = ()
    #     if c_dim == 1:
    #         constr = constr + (diff_energy - self.energy_tol,)
    #     elif c_dim == 2:
    #         constr = constr + (diff_energy - self.energy_tol, self.number_trips - self.trip_max,)
    #     elif c_dim == 3:
    #         constr = constr + (diff_energy - self.energy_tol, self.number_trips - self.trip_max, self.heat_load - self.heat_max)
    #     else:
    #         print("Warning: Constraint number should be ONE, TWO, or THREE!")
    #     return constr

        # return diff_energy - 2, self.number_trips - self.trip_max
        # return diff_energy - 2,
    def calc_number_trips(self, xx):
        x = np.array(xx)
        diff_energy = np.fabs(np.sum(self.length * x) - self.energy)
        fault = self.trip_slope * (x - self.fault_grad)
        fault_rate = np.sum(np.exp(self.A + fault[self.trip_slope > 0]))
        number_trips = 3600.0 * fault_rate
        return number_trips

    def calc_heat_load(self, xx):
        x = np.array(xx)
        heat_load = np.sum(1e12 * (x * x) * self.length / (self.Q * self.cnst))
        return heat_load

    def calc_energy(self, xx):
        x = np.array(xx)
        return  np.sum(self.length * x)

    def trip_rate_opt(self):
        x = self.range_up * 1
        idx = self.trip_slope>0
        lb = self.length[idx]/self.trip_slope[idx]
        lb_sum = np.sum(lb)
        dE = self.energy - np.sum(x[np.logical_not(idx)]*self.length[np.logical_not(idx)])
        lm = np.exp((dE - np.sum(lb*np.log(lb/3600.0)) + self.A * lb_sum - np.sum(self.fault_grad[idx]*self.length[idx]))/lb_sum)
        x[idx] = (np.log(lm*lb/3600.0) - self.A)/self.trip_slope[idx] + self.fault_grad[idx]
        return  x

    def trip_rate_opt(self, delta_E):
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

    def recreate_pop2(self, x):
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

    def recreate_pop3(self, x):
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

    # def recreate_pop3(self, x):
    #     current_energy = np.sum(self.length * x)
    #     while np.fabs(current_energy - self.energy) > 2:
    #         energy_change = self.energy - current_energy
    #         coef_g = (self.cnst * self.Q) / (self.length * x)
    #         # coef_g = np.abs(coef_g)
    #         coef_h = coef_g * coef_g
    #         coef_t = 1 / coef_h
    #         k = 0.9
    #         kt = k * energy_change / np.sum(coef_h * self.length)
    #         kh = (1-k) * energy_change / np.sum(coef_h * self.length)
    #         for i, xi in enumerate(x):
    #             x[i] += coef_t[i]*kt + coef_h[i]*kh
    #             if x[i] > self.range_up[i]:
    #                 x[i] = self.range_up[i] * (1 - 1e-6)
    #             elif x[i] < self.range_low[i]:
    #                 x[i] = self.range_low[i] * (1 + 1e-6)
    #         current_energy = np.sum(self.length * x)

    def recreate_pop4(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst*self.Q) / (self.length*x)
            # coef_g = np.abs(coef_g)
            # coef_g = coef_g*coef_g
            coef_g = coef_g * coef_g
            coef_g = 1/coef_g
            k = energy_change / np.sum(coef_g*self.length)
            for i, xi in enumerate(x):
                x[i] += coef_g[i]*k
                if (x[i] > self.range_up[i]):
                    x[i] = self.range_up[i] * (1 - 1e-6)
                elif (x[i] < self.range_low[i]):
                    x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def recreate_pop5(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst*self.Q) / (self.length*x)
            coef_g = np.abs(coef_g)
            # coef_g = coef_g*coef_g
            # coef_g = coef_g * coef_g
            coef_g = 1/coef_g
            k = energy_change / np.sum(coef_g*self.length)
            for i, xi in enumerate(x):
                x[i] += coef_g[i]*k
                if (x[i] > self.range_up[i]):
                    x[i] = self.range_up[i] * (1 - 1e-6)
                elif (x[i] < self.range_low[i]):
                    x[i] = self.range_low[i] * (1 + 1e-6)
            current_energy = np.sum(self.length * x)

    def recreate_pop6(self, x):
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
    def recreate_pop7(self, x):
        current_energy = np.sum(self.length * x)
        while (np.fabs(current_energy - self.energy) > self.energy_tol):
            energy_change = self.energy - current_energy
            coef_g = (self.cnst*self.Q) / (self.length*x)
            # coef_g = np.abs(coef_g)
            # coef_g = coef_g*coef_g
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
        self.recreate_pop4(x)
        return x.tolist()


def revise_problem(n_off):
    global dim
    global data
    (dim, data, id_down) = cavities_off.remove(dim, data, n_off)
    return lem_upgrade(), id_down


def nl():
    global data
    global lines
    filename = 'lem_nl.csv'
    # lines = np.loadtxt(filename, dtype='str', skiprows=1)  # load data as string, first row is the cavity name
    lines = np.loadtxt(filename, dtype='str', delimiter=',', skiprows=1)  # load data as string, first row is the cavity name
    data = lines[:, 1:8].astype('float64')
    # nl = lem_upgrade()
    nl = lem_upgrade(dim, c_dim, c_ineq_dim)
    return nl


def sl():
    global data
    global lines
    filename = 'lem_sl.csv'
    # lines = np.loadtxt(filename, dtype='str', skiprows=1)  # load data as string, first row is the cavity name
    lines = np.loadtxt(filename, dtype='str', delimiter=',', skiprows=1)  # load data as string, first row is the cavity name
    data = lines[:, 1:8].astype('float64')
    # sl = lem_upgrade()
    sl = lem_upgrade(dim, c_dim, c_ineq_dim)
    return sl

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


