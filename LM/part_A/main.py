# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 250
    emb_size = 350
    #lr = 0.05 #fixed after trials
    lr = 0.05
    clip = 5
    n_epochs = 100
    patience = 3
    result = training(hid_size, emb_size, lr, clip, n_epochs, patience, 'RNN_embsize350_hidsize250_lr0.05.pt')
    print(result)