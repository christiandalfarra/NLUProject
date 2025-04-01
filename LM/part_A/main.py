# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_size = 200
    emb_size = 300
    lr_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    clip = 5
    n_epochs = 100
    patience = 3
    results = []
    for lr in lr_values:
        print(f"Training with lr: {lr}")
        results = training(hid_size, emb_size, lr, clip, n_epochs, patience)
    print(f"Results: {results}")
    plt.plot(results)
    plt.xlabel('Learning Rate')
    plt.ylabel('PPL')
    plt.title('PPL vs Learning Rate')
    #plt.savefig('plot.png', dpi=300) to save the plot use this command
    #organizzati le varie iterazioni con i parametri che vuoi valutare
    plt.show()