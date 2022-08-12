# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
# from CEBAF_opt.utils import format
import pandas as pd
import math

from gym import spaces
import gym

class cavity():
    """


    """
    def __init__(self, data, q_curves, cavity_id):
        """

        :param pathToCavityData:
        """
        # data = pd.read_pickle(path_cavity_data)
        self.cavity_id = cavity_id
        # print(cavity_id)
        # print(data)
        # print(q_curves)
        try:
            row = data[data["cavity_id"] == cavity_id]
            row_q = q_curves[q_curves["cavity_id"] == cavity_id]
        except:
            print("An exception occurred")
        # except(e):
        #     print(e)

        if len(row) < 1:
            print("Cavity-id ", self.cavity_id, " not found in the dataset...")
        if len(row_q) < 1:
            print("Cavity-id ", self.cavity_id, " curve of q vs. grad not found...")
        self.length = float(row["length"])
        self.type = str(row['type'])
        self.Q0 = float(row["Q0"])
        self.trip_slope = float(row['trip_slope'])
        self.trip_offset = float(row['trip_offset'])
        self.shunt = float(row['shunt_impedance'])
        self.max_gset = float(row['max_gset'])
        self.ops_gset_max = float(row['ops_gset_max'])
        # print(row_q['c2'].values[0])
        # print(row_q['c1'].values[0])
        # print(row_q['c0'].values[0])
        if len(row_q) < 1:
            self.q_curve = np.poly1d([self.Q0])
        else :
            self.q_curve = np.poly1d([row_q['c2'].values[0], row_q['c1'].values[0], row_q['c0'].values[0]])
        if self.q_curve == np.poly1d([]):
            self.q_curve = np.poly1d([self.Q0])
        if pd.isna(self.ops_gset_max):
            self.max_gset_to_use = self.max_gset
        else:
            self.max_gset_to_use = self.ops_gset_max
        self.min_gset = 3.0
        ## IF C100 set min gset to 5.0
        ## If C75 ? (Ask Jay)
        # self.min_gset = 3.0
        self.gradient = self.max_gset_to_use
        if self.type in ["C75", "C100"]:
            self.min_gset = 5.0
            def computeHeat():
                Q = self.q_curve(self.gradient)
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * Q)
        else:
            def computeHeat():
                Q = self.q_curve(self.gradient)
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * Q)

        self.RFheat = computeHeat
        self.chargeFraction = self.getTripRate()

    def describe(self):
        """

        :return:
        """
        print("Cavity type: ", self.cavity_id)
        print("Cavity current gradient: ", self.gradient)
        print("Cavity length: ", self.length)
        print("Cavity Q0: ", self.Q0)
        print("Cavity trip slope: ", self.trip_slope)
        print("Cavity trip offset: ", self.trip_offset)
        print("Cavity shunt: ", self.shunt)
        print("Cavity max_gset: ", self.max_gset)
        print("Cavity ops gset max: ", self.ops_gset_max)
        print("max gset to use: ", self.max_gset_to_use)



    def setGradient(self, grad):
        """

        :param gradArray:
        :return:
        """
        if type(grad) in [int, float, np.float32, np.float64]:
            if grad < self.min_gset:
                # print("Error: ", self.cavity_id, " requested gradient is lower than minimum safe gradient. Setting to min...")
                self.gradient = self.min_gset
            # elif grad > self.max_gset_to_use:
            elif grad > self.max_gset:
                # print("Error: ", self.cavity_id, " requested gradient is higher than maximum safe gradient. Setting to max...")
                self.gradient = self.max_gset_to_use
            else:
                self.gradient = grad
        else:
            print("Error: ", self.cavity_id, " gradient must be a float or integer and not ", type(grad))

        self.chargeFraction += 60*self.getTripRate()

    def getRFHeat(self):
        """

        :return:
        """
        return self.RFheat()


    def getTripRate(self):
        """

        :return:
        """
        if pd.isna(self.trip_slope) or pd.isna(self.trip_offset):
            return 0
        return math.exp(-10.268+self.trip_slope*(self.gradient - self.trip_offset))

    def getGradient(self):
        return self.gradient

    def getCavityState(self):
        # return (self.gradient - self.min_gset) / (self.max_gset_to_use - self.min_gset)
        return (self.gradient - 3.) / (15. - 3.)
    
    def getValue(self, name):
        if name == 'cavity_id':
            return self.cavity_id
        elif name == 'length':
            return self.length
        elif name == 'type':
            return self.type
        elif name == 'Q0':
            return self.Q0
        elif name == 'trip_slope':
            return self.trip_slope
        elif name == 'trip_offset': 
            return self.trip_offset
        elif name == 'shunt_impedence':
            return self.shunt
        elif name == 'max_gset':
            return self.max_gset
        elif name == 'ops_gset_max':
            return self.ops_gset_max
        elif name == 'max_gset_to_use':
            return self.max_gset_to_use
        elif name == 'min_gset':
            return self.min_gset
        elif name == 'gradient':
            return self.gradient
        elif name == 'q_curve':
            return self.q_curve
        elif name == 'c2':
            if self.q_curve.order<2:
                return 0
            return self.q_curve.coefficients[0]
        elif name == 'c1':
            if self.q_curve.order<1:
                return 0
            return self.q_curve.coefficients[1]
        elif name == 'c0':
            return self.q_curve.coefficients[-1]
        else:
            print('Cavity does not have ', name, ' !')
            return np.nan

    def getEnergy(self):
        return self.length * self.gradient

    def reset(self):
#         gset = self.max_gset_to_use - 0.95  #North linac
        gset = self.max_gset_to_use - 3.0 #1L06
        if gset < self.min_gset:
            gset = self.min_gset
        self.gradient = gset
#         print(self.getGradient())
#         print("Resetting to: ", self.gradient)
        # self.gradient = self.min_gset
        # self.gradient = 10.48
#         self.chargeFraction = self.getTripRate()

class digitalTwin():
    """

    """
    def __init__(self, path_cavity_data, path_q_data, linac="North"):
        """

        :param path_cavity_data:
        """
        data = pd.read_pickle(path_cavity_data)
        q_curves = pd.read_pickle(path_q_data)
        cavity_ids = data["cavity_id"]
        self.cavities = []
        self.cavity_order = []

        for i in range(len(cavity_ids)):
            if linac.lower() == "north" or linac.lower() == "n":
                if cavity_ids.iloc[i][0] == '1':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
            elif linac.lower() == "south" or linac.lower() == "s":
                if cavity_ids.iloc[i][0] == '2':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
            elif linac.lower() == "test1":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
            elif linac.lower() == "test2":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
            elif linac.lower() == "test4":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1', '2', '3', '4']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
            elif linac.lower() == "test8":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6': # and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))

            else:
                if cavity_ids.iloc[i][0] == '0':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
        self.name = linac

        if linac.lower() in ["north", "n", "south", "s"]:
            self.energyConstraint = 1050
            self.energyMargin = 2
        elif linac.lower() == "test1":
            self.energyConstraint = 2.5 #7.85
            self.energyMargin = 0.1
            self.target_rfHeat = 6.9
        elif linac.lower() == "test2":
            self.energyConstraint = 7.85 #7.85
            self.energyMargin = 0.5
            self.target_reward = -2.0
        elif linac.lower() == "test4":
            self.energyConstraint = 15.95 #7.85
            self.energyMargin = 0.2
        elif linac.lower() == "test8":
            self.energyConstraint = 31.8 #7.85
            self.energyMargin = 0.2
            self.target_reward = -75#-77.8
        else:
            self.energyConstraint = 126.5
            self.energyMargin = 0.22

    def list_cavities(self):
        """

        :return:
        """
        return self.cavity_order

    def describeCavity(self, cavity_id):
        """

        :param cavity_id:
        :return:
        """
        indx = self.cavity_order.index(cavity_id)
        self.cavities[indx].describe()

    def setGradients(self, grad_array):
        """

        :param grad_array:
        :return:
        """
        for i in range(len(self.cavities)):
            self.cavities[i].setGradient(grad_array[i])

    def getGradients(self):
        """

        :return:
        """
        gradients = []
        for cavity in self.cavities:
            gradients.append(cavity.getGradient())
        return np.array(gradients)

    def getState(self):
        """

        :return:
        """
        state_vars = []
        for cavity in self.cavities:
            state_vars.append(cavity.getCavityState())
        state_vars.append((self.getEnergyGain()-(self.energyConstraint - self.energyMargin))/(2*self.energyMargin))
        # state_vars.append((self.getRFHeat()-75.)/35.)
        # state_vars.append(self.getTripRates()*1e4)
        return np.array(state_vars)

    def getRFHeat(self):
        """

        :return:
        """
        heat = 0.0
        for cavity in self.cavities:
            heat += cavity.getRFHeat()
        return heat

    def getTripRates(self):
        """

        :return:
        """
        tr = 0.0
        for cavity in self.cavities:
            tr += cavity.getTripRate()
        return 3600*tr

    def getEnergyGain(self):
        """

        :return:
        """
        e = 0.0
        for cavity in self.cavities:
            e += cavity.getEnergy()
        return e

    def getMinGradients(self):
        """

        """
        min_grads = []
        for cavity in self.cavities:
            min_grads.append(cavity.min_gset)
        return np.array(min_grads)

    def getMaxGradients(self):
        """

        """
        max_grads = []
        for cavity in self.cavities:
            max_grads.append(cavity.max_gset_to_use)
        return np.array(max_grads)

    def reset(self):
        for cavity in self.cavities:
            cavity.reset()
#         print("Reset energy: ", self.getEnergyGain())

    def getEnergyConstraint(self):
        return self.energyConstraint

    def getEnergyMargin(self):
        return self.energyMargin

    def updateGradients(self, delta):
        """

        """
        # print("Updating gradients with: ", delta)
        new_grads = self.getGradients() + delta/10.
        self.setGradients(new_grads)

    def isTrip(self):
        for cavity in self.cavities:
            if cavity.chargeFraction >= 1:
                return True
        return False

    def printChargeFraction(self):
        c = []
        for cavity in self.cavities:
            c.append(cavity.chargeFraction)
        print(c)
        
    def getValues(self, name):
        """

        :return:
        """
        values = []
        for cavity in self.cavities:
            values.append(cavity.getValue(name))
        return np.array(values)
    
    def getTotalCavityNumber(self):
        return len(self.cavity_order)
    
    def getName(self):
        return self.name
    
class cryoModule(digitalTwin):
    def __init__(self, path_cavity_data, path_q_data, cryomodule, energy_constraint, energy_margin):
        """

        :param path_cavity_data:
        :param cryo_module:
        :param energy_constraint:
        :param energy_margin:
            
        """
        data = pd.read_pickle(path_cavity_data)
        q_curves = pd.read_pickle(path_q_data)
        module = data[data['cavity_id'].str.contains(cryomodule)]
        if module.empty:
            raise ValueError("Cryomodule not found!")
        cavity_ids = module["cavity_id"]
        self.cavities = []
        self.cavity_order = []
        for i in range(len(cavity_ids)):
            self.cavity_order.append(cavity_ids.iloc[i])
            self.cavities.append(cavity(data, q_curves, cavity_ids.iloc[i]))
        self.name = cryomodule
        self.energyConstraint = energy_constraint
        self.energyMargin = energy_margin

class cebaf_env_v1(gym.Env):
    def __init__(self, path_cavity_data, linac="North", trackTime=False, max_steps=100, reward_weights=[0.9, 0.1]):
        self.linac = digitalTwin(path_cavity_data, linac)
        self.nCavities = len(self.linac.list_cavities())
        self.trackTime = trackTime
        self.reward_weights = np.array(reward_weights)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nCavities,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.,
            high=1.,
            shape=(self.nCavities+1,),
            dtype=np.float32
        )

        self.states = self.linac.getGradients()
        self.beta, self.gamma, self.alpha = 1., 1e3, 100 #0.1,50.0
        self.max_steps = max_steps
        self.counter = 0
        
#         self.r_w = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001, 0.000001, 0.0000001, 0.0]
#         self.rw_counter = 0

    def _computeReward(self):
#         return - self.beta * self.linac.getRFHeat(), - self.gamma * self.linac.getTripRates()
        # return - self.reward_weights[0]*self.beta * self.linac.getRFHeat() \
            #    - self.reward_weights[1]*self.gamma * self.linac.getTripRates() \
                # - self.alpha * (abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint())) \
            # - self.alpha * max((abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) - self.linac.getEnergyMargin()), 0.) \
            #   + self.counter * 10
        return -self.linac.getRFHeat()

    def _takeAction(self, action):
        self.linac.updateGradients(action)

    def step(self, action):
        """

        """
        self._takeAction(action)
        reward = self._computeReward()
        next_state = self.linac.getState()
#         next_state = np.append(next_state, self.reward_weights)
        done = False
        if self.trackTime == True:
            done = self.linac.isTrip()
            if done == True:
                self.linac.printChargeFraction()

        if abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) >= self.linac.getEnergyMargin():
            done = True
            # print("Energy Gain: ", self.linac.getEnergyGain())
            # print("Number of Steps: ", self.counter+1)
            # print("Gradients: ", self.linac.getGradients())
            reward = -10000 
        elif abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) >= self.linac.getEnergyMargin()*0.90:
            reward = -1000

        self.counter += 1
        if self.counter >= self.max_steps:
            done = True

#         if reward >= self.linac.target_reward:
#             reward = 0
#             done = True


        return next_state, reward, done, {}

    def reset(self):
        # r_w = np.random.uniform(0, 1)
#         r_w = self.r_w[self.rw_counter]
#         self.rw_counter += 1
#         if self.rw_counter >= len(self.r_w):
#             self.rw_counter = 0
        r_w = 1.0
        self.reward_weights = np.array([r_w, 1-r_w])
        # self.reward_weights = np.array([1.0, 0.0])
        self.linac.reset()
        self.counter = 0
#         return np.append(self.linac.getState(), self.reward_weights)
        return self.linac.getState()

    def getTripRates(self):
        return self.linac.getTripRates()
    def getRFHeat(self):
        return self.linac.getRFHeat()

# dt = digitalTwin("~/LDRD/cavity_table.pkl")
# # dt.list_cavities()
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", dt.getTripRates())
# print("Energy gain: ", dt.getEnergyGain(), " MeV")
# dt.setGradients([13.]*200)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", dt.getTripRates())
# print("Energy gain: ", dt.getEnergyGain(), " MeV")
# tmp = np.array([2]*418).astype(np.float32)
# dt.setGradients(tmp)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", 3600*dt.getTripRates()*1e2)
# tmp = np.array([3]*418).astype(np.float32)
# dt.setGradients(tmp)
# print("RF heat: ", dt.getRFHeat())
# print("Trip Rate: ", 3600*dt.getTripRates()*1e2)[kishan@ifarm1901 models]$