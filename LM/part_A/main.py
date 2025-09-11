# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 350
    emb_size = 350
    lrs = [0.05, 0.1, 0.5]
    i = 2
    for lr in lrs:
        results = training_SGD(hid_size, emb_size, lr, clip = 5 , n_epochs = 100 , patience = 3 , experiment=f'exp{i}_LSTM_embsize{emb_size}_hidsize{hid_size}_lr{lr}')
        i+=1