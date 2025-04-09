# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 200
    emb_size = 300
    #lr = 0.05 #fixed after trials
    lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    clip = 5
    n_epochs = 100
    patience = 3
    results = []
    for rate in lr:
        result = training(hid_size, emb_size, rate, clip, n_epochs, patience, f'RNN_lr_{rate}')
        results.append(result)
        print(f'lr = {rate} -> PPL: {result}')
    plt.figure(figsize=(10, 6))
    plt.plot(lr, results, marker='o', linestyle='-', color='b')
    plt.xscale('log')  # Learning rates are often better on a log scale
    plt.yticks(range(150, 700, 50))  # Y-axis ticks from 150 to 650 in steps of 50
    plt.xlabel('Learning Rate')
    plt.ylabel('Result')
    plt.title('Results vs Learning Rate')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig('plot.png', dpi=300)
    #organizzati le varie iterazioni con i parametri che vuoi valutare
    #usando RNN
    # da testare ora con LSTM
