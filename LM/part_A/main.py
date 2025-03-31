# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    hid_size = 200
    emb_size = 300
    lr=0.0001
    clip = 5
    n_epochs = 100
    patience = 3
    training(hid_size, emb_size, lr, clip, n_epochs, patience)