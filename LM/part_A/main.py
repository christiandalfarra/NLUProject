# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *
"""
    'model_arch' have different options related to the experiments:
    'RNN' : Simple RNN
    'LSTM' : Simple LSTM
    'LSTM_DOEMB_LAYER' : LSTM with dropout after embedding layer
    'LSTM_DOEMB_LAST_LAYER' : LSTM with dropout after embedding layer and before the last linear layer

    'optimizer' have two options (on default is setted to SGD):
    'SGD' : Stochastic Gradient Descent
    'AdamW' : the AdamW optimizer
    """
param ={
    'model_arch' : 'LSTM_DOEMB_LAST_LAYER',
    'emb_size': 350,
    'hidden_size': 350,
    'lr' : 1,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,

    'optimizer': 'SGD'
}

if __name__ == "__main__":
    i = 50
    training(param, experiment=f"exp{i}_{param['model_arch']}_lr{param['lr']}")