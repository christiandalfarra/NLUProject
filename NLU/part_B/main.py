# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

import argparse

# Import everything from functions.py file
from functions import *
from utils import *
param ={

    'dropout_prob': 0.3,

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 50,
    'patience': 3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--mode', type=str, default=None, help='train or inference')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD, AdamW, or Adam')
    parser.add_argument('--dropout_prob', type=float, default=None, help='dropout probability')

    args = parser.parse_args()

    if args.mode is None:
        print('Please provide a mode (train or inference)')
        exit()
    if args.mode == 'train':
        if args.lr is None:
            print('Please provide a learning rate')
            exit()
        if args.optimizer is None:
            print('Please provide an optimizer (SGD, AdamW, or Adam)')
            exit()
        if args.optimizer not in ['SGD', 'AdamW', 'Adam']:
            print('Optimizer must be one of: SGD, AdamW, Adam')
            exit()

        param['lr'] = args.lr
        param['optimizer'] = args.optimizer
        if args.dropout_prob is not None:
            param['dropout_prob'] = args.dropout_prob

        experiment = f"bert_lr{param['lr']}_{param['optimizer']}"
        print("Training Mode")
        training(param, experiment)
    elif args.mode == 'inference':
        print("Inference Mode")
        print("Inference is not implemented in part_B yet.")
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'inference'")
