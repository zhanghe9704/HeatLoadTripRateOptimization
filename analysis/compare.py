from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants



def gradient(filename):
    lines = np.loadtxt(filename, dtype='str', skiprows=7)  # load data as string, first row is the cavity name
    data = lines[:, 2].astype('float64')
    voltage = data[0:200]
    cavity_length = np.arange(200).astype('float64')
    cavity_length[cavity_length < 160] = constants.c/1.497e9*2.5
    cavity_length[cavity_length > 0.6] = constants.c/1.497e9*3.5
    gradient = voltage/cavity_length*1e-6
    return gradient


def compare(grad_current, grad_opt, path):
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.figure()
    plt.scatter(np.arange(grad_current.size)[grad_current>0], grad_current[grad_current>0], c='b', label='Current gradient')
    plt.scatter(np.arange(grad_opt.size)[grad_current>0], grad_opt[grad_current>0], c='r', label='Optimized gradient')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlim([-1, 201])
    plt.xlabel('index of cavity')
    plt.ylabel('gradient (MV/m)')
    plt.title('Compare the optimized gradients with current setting')
    plt.savefig(path+'/Gradients'+date_str+'.eps', format="eps")
    plt.show()

    plt.figure()
    # plt.scatter(np.arange(grad_current.size), (grad_opt-grad_current)/grad_current*100, label='Gradient change in %')
    plt.bar(np.arange(grad_current.size)[grad_current>0], ((grad_opt-grad_current)/grad_current*100)[grad_current>0], label='Gradient change')
    plt.legend()
    plt.grid()
    plt.xlim([-1, 201])
    plt.xlabel('index of cavity')
    plt.ylabel('change of gradient (%)')
    plt.title('Compare the optimized gradients with current setting')
    plt.savefig(path+'/Gradient_change' + date_str + '.eps', format="eps")
    plt.show()


def compare_max(grad_max, grad_opt, path):
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    plt.figure()
    # plt.scatter(np.arange(grad_current.size), (grad_opt-grad_current)/grad_current*100, label='Gradient change in %')
    plt.bar(np.arange(grad_max.size), grad_opt/grad_max*100, color='g', label='Gradient_opt./Gradient_max.')
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlim([-1, 201])
    plt.xlabel('index of cavity')
    plt.ylabel('Gradient_opt./Gradient_max. (%)')
    plt.title('Compare the optimized gradients with the max gradient')
    plt.savefig(path+'/Gradient_max' + date_str + '.eps', format="eps")
    plt.show()