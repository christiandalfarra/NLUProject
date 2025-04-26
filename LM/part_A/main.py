# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 300
    emb_size = 350
    lr = 0.05
    clip = 5
    n_epochs = 100
    patience = 3
    result = training(hid_size, emb_size, 0.05, clip, n_epochs, patience, 'LSTM_embsize400_hidsize350_lr0.05')
    print(result)
    result = training(hid_size, emb_size, 0.1, clip, n_epochs, patience, 'LSTM_embsize400_hidsize350_lr0.1')
    print(result)