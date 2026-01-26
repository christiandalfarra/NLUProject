# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *
""" 
    'mode':
        'train' : to train the model
        'evaluate' : to test the model
    'model_arch' have different options related to the experiments:
        'RNN' : Simple RNN
        'LSTM' : Simple LSTM

    'optimizer' have two options:
        'SGD' : Stochastic Gradient Descent
        'AdamW' : the AdamW optimizer
    """
param ={
    'mode': 'train',  # 'train' or 'evaluate'

    'model_arch' : 'RNN',
    'emb_size': 350,
    'hidden_size': 350,
    'emb_dropout': 0.2,
    'out_dropout': 0.2,

    'lr' : 0.1,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,

    'optimizer': 'SGD'  # 'SGD' or 'AdamW'
}

if __name__ == "__main__":
    """ 
    To perform a single training or evaluation run the script as is, 
    change the parameters in the param dictionary above to try different experiments.

    To perform a grid search on learning rate, emb size and hidden size uncomment the line below
    and make sure that in the param dictionary above lr, emb_size and hidden_size are lists of values.
    # grid search on learning rate emb size and hidden size
    """
    #grid_search_hyperparameters(param)

    if param['mode'] == 'train':
        training(param, experiment=f"exp500_{param['model_arch']}_lr{param['lr']}_{param['optimizer']}")
    elif param['mode'] == 'evaluate':
        test_ppl = testing(param, model_path=f'bin/exp_{param["model_arch"]}_lr{param["lr"]}_{param["optimizer"]}.pt')
        print('Test ppl: ', test_ppl)
