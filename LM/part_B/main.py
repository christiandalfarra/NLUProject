# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

""" 
    'mode':
        'train' : to train the model
        'inference' : to test the model
    'model_arch' have different options related to the experiments:
        'LSTM' : Simple LSTM
    'weight_tying' : Boolean to specify if we want to use weight tying or not
    'var_dropout' : Boolean to specify if we want to use variational dropout or not

    'optimizer' have two options (on default is setted to SGD):
        'SGD' : Stochastic Gradient Descent
        'NTAvSGD' : the Non-monotonic Triggered Averaged Stochastic Gradient Descent optimizer
    """

param ={
    'mode': 'train',

    'model_arch' : 'LSTM',
    'emb_size': 350,
    'hidden_size': 350,

    'lr' : 0.5,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,

    'weight_tying': True,
    'var_dropout': False,
    'emb_dropout': 0.5,
    'out_dropout': 0.5,

    'nt_interval': 3,  # Used only if optimizer is NTAvSGD

    'optimizer': 'SGD'  # 'SGD' or 'NTAvSGD'
}

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    if param['mode'] == 'train':
        training(param, experiment=f"exp_800000_weighttie_{param['model_arch']}_lr{param['lr']}_{param['optimizer']}")
    elif param['mode'] == 'inference':
        test_ppl = testing(param, model_path=f'bin/exp_{param["model_arch"]}_lr{param["lr"]}_{param["optimizer"]}.pt')
        print('Test ppl: ', test_ppl)
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'inference'")
