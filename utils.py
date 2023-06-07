'''
provide some useful functions.
'''
import numpy as np


def read_log(logfile):
    '''
    read log file and return the run information.
    the run information is a list of tuple (dataset_name, information)
    the `information` is a dict with keys:
        'style': the network style in 'ann', 'linear', 'qnnn' and 'qnnl'
        'n_qubits': the number of qubits
        'n_layers': the number of layers
        'n_reuploads': the number of reuploads
        'n_classes': the number of classes
        
    '''
    with open(logfile, 'r') as f:
        lines = f.readlines()
    
