# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
""" 
    'mode':
        'train' : to train the model
        'inference' : to test the model
    'model_arch' have different options related to the experiments:
        'LSTM' : Simple LSTM
        'LSTM_BIDIRECTIONAL' : Bidirectional LSTM
        'LSTM_DROPOUT' : LSTM Bidirectional with dropout embedding layer and the last linear layer
    """
param ={
    'mode': 'train',

    'model_arch' : 'LSTM',
    'emb_size': 200,
    'hidden_size': 300,
    'dropout_prob': 0.5,

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 200,
    'patience': 3,
    'multiple_runs': 5,

    'optimizer': 'AdamW'  # 'SGD' or 'AdamW'
}

if __name__ == "__main__":
    if param['mode'] == 'train':
        print("Training Mode")
        training(param, "first_experiment")
    elif param['mode'] == 'inference':
        print("Inference Mode")

