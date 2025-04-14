# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 200
    emb_size = 300
    #lr = 0.05 #fixed after trials
    lr = 0.01
    clip = 5
    n_epochs = 100
    patience = 3
    result = training(250, emb_size, lr, clip, n_epochs, patience, 'RNN_hidden_size250')
    print(result)
    
    result = training(150, emb_size, lr, clip, n_epochs, patience, 'RNN_hidden_size150')
    print(result)
    
    result = training(hid_size, 250, lr, clip, n_epochs, patience, 'RNN_emb_size250')
    print(result)
    
    result = training(hid_size, 350, lr, clip, n_epochs, patience, 'RNN_emb_size350')
    print(result)
    
    #organizzati le varie iterazioni con i parametri che vuoi valutare
    #usando RNN
    # da testare ora con LSTM
