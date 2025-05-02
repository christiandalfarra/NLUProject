# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *

if __name__ == "__main__":
    hid_sizes = [350]
    emb_sizes = [350]
    lrs = [0.1, 0.5, 1, 2]
    results = grid_search_hyperparameters_RNN(hid_sizes, emb_sizes, lrs, clip=5, n_epochs=100, patience=3)