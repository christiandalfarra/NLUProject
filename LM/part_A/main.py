# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *
""" 
    'mode':
        'train' : to train the model
        'inference' : to test the model
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
    'mode': 'train',

    'model_arch' : 'LSTM_DOEMB_LAST_LAYER',
    'emb_size': 350,
    'hidden_size': 350,
    'emb_dropout': 0.2,
    'out_dropout': 0.2,

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,

    'optimizer': 'AdamW'  # 'SGD' or 'AdamW'
}

if __name__ == "__main__":
    """ 
    To perform a single training or inference run the script as is, 
    change the parameters in the param dictionary above to try different experiments.

    To perform a grid search on learning rate, emb size and hidden size uncomment the line below
    and make sure that in the param dictionary above lr, emb_size and hidden_size are lists of values.
    # grid search on learning rate emb size and hidden size
    """
    #grid_search_hyperparameters(param)

    if param['mode'] == 'train':
        training(param, experiment=f"exp700_{param['model_arch']}_lr{param['lr']}_{param['optimizer']}")
    elif param['mode'] == 'inference':
        test_ppl = testing(param, model_path=f'bin/exp700_{param["model_arch"]}_lr{param["lr"]}_{param["optimizer"]}.pt')
        print('Test ppl: ', test_ppl)