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
        'WeightTiedLSTM' : Weight Tied LSTM
        'VariationalDropoutLSTM_emb' : LSTM with variational dropout after embedding layer
        'VariationalDropoutLSTM_last' : LSTM with variational dropout after last
        'VariationalDropoutLSTM_emb_last' : LSTM with variational dropout after embedding layer and before the last linear layer

    'optimizer' have two options (on default is setted to SGD):
        'SGD' : Stochastic Gradient Descent
        'NTAvSGD' : the Non-monotonic Triggered Averaged Stochastic Gradient Descent optimizer
    """

param ={
    'mode': 'train',

    'model_arch' : 'WeightTiedLSTM',
    'emb_size': 350,
    'hidden_size': 350,
    'emb_dropout': 0.5,
    'out_dropout': 0.2,

    'lr' : 0.1,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,

    'optimizer': 'SGD'  # 'SGD' or 'NTAvSGD'
}

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    if param['mode'] == 'train':
        if param['optimizer'] == 'NTAvSGD':
            raise ValueError("NTAvSGD optimizer is not implemented for training. Please use 'SGD' optimizer.")
        elif param['optimizer'] == 'SGD':
            training_SGD(param, experiment=f"exp_{param['model_arch']}_lr{param['lr']}_{param['optimizer']}")
    elif param['mode'] == 'inference':
        test_ppl = testing(param, model_path=f'bin/exp_{param["model_arch"]}_lr{param["lr"]}_{param["optimizer"]}.pt')
        print('Test ppl: ', test_ppl)
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'inference'")