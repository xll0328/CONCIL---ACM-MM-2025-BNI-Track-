import pickle
import os
from os.path import join

def read_data(data_path: str, split: str):
    """read specific data from the given data_path"""
    data_path = join(data_path, split + '.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def check_dir(directory):

    cwd = os.getcwd()
    path = join(cwd, directory)
    if not os.path.exists(path):
        os.makedirs(path)

def one_hot(label, N_classes):
    """
    convert label to one-hot label
    """
    one_hot_label = [0] * N_classes
    one_hot_label[label] = 1  
        
    return one_hot_label