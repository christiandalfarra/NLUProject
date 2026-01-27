# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

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

    'emb_size': size of the embedding layer
    'hidden_size': size of the hidden layer
    'dropout_prob': dropout probability (only for the model with dropout)
    'lr' : learning rate
    'clip': gradient clipping value
    'patience': number of epochs to wait before early stopping

    'n_epochs': number of epochs
    'multiple_runs': number of runs with different weight initialization

    'optimizer': optimizer to use. Available optimizers: 'SGD' or 'AdamW' or 'Adam'
    """
param ={
    'mode': 'train',

    'model_arch' : 'LSTM_DROPOUT',
    'emb_size': 300,
    'hidden_size': 768,
    'dropout_prob': 0.3,

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 50,
    'patience': 3,
    'multiple_runs': 5,

    'optimizer': 'Adam'  # 'SGD' or 'AdamW' or 'Adam'
}

if __name__ == "__main__":
    if param['mode'] == 'train':
        print("Training Mode")
        training(param, "bert")
    elif param['mode'] == 'inference':
        print("Inference Mode")