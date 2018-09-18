from datetime import datetime
import os


def create(algo_name):
    path = os.getcwd()
    if not(os.path.exists('Data')):
        os.mkdir('Data')
    path = os.path.join(path, 'Data')
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    folder = date_str+'__'+algo_name
    path = os.path.join(path, folder)
    if not(os.path.exists(path)):
        os.mkdir(path)
    return path