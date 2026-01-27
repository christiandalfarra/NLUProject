# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import matplotlib.pyplot as plt
from functions import *
import argparse
param ={
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--mode', type=str, default=None, help='train or evaluate, for training specify also model_arch, lr and optimizer, for evaluation specify model_number')
    parser.add_argument('--model_arch', type=str, default=None, help='RNN or LSTM')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--optimizer', type=str, default=None, help='SGD or AdamW')

    parser.add_argument('--emb_dropout', type=float, default=0.0, help='embedding dropout probability')
    parser.add_argument('--out_dropout', type=float, default=0.0, help='output dropout probability')
    parser.add_argument('--emb_size', type=int, default=350, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=350, help='hidden size')

    parser.add_argument('--model_number', type=int, default=None, help='model number for evaluation')

    args = parser.parse_args()
    if args.mode is None:
        print('Please provide a mode (train or evaluate)')
        exit()
    mode = args.mode
    if mode == 'train':
        if args.model_arch is None:
            print('Please provide a model architecture (RNN or LSTM)')
            exit()
        if args.lr is None:
            print('Please provide a learning rate')
            exit()
        if args.optimizer is None:
            print('Please provide an optimizer (SGD or AdamW)')
            exit()
        param['lr'] = args.lr
        param['optimizer'] = args.optimizer
        param['model_arch'] = args.model_arch
        
        param['emb_dropout'] = args.emb_dropout
        param['out_dropout'] = args.out_dropout
        param['emb_size'] = args.emb_size
        param['hidden_size'] = args.hidden_size

    if mode == 'train':
        training(param, experiment=f"exp_{param['model_arch']}_lr{param['lr']}_{param['optimizer']}_embDO_{param['emb_dropout']}_outDO_{param['out_dropout']}")
    elif mode == 'evaluate':
        exp_number = args.model_number
        if exp_number is None:
            print('Please provide a model number to evaluate')
            exit()
        if exp_number == 1:
            param['model_arch'] = 'RNN'
            path = f'bin/RNN.pt'
        elif exp_number == 2:
            param['model_arch'] = 'LSTM'
            path = f'bin/LSTM.pt'
        elif exp_number == 3:
            param['model_arch'] = 'LSTM'
            path = f'bin/LSTM_DO.pt'
        elif exp_number == 4:
            param['model_arch'] = 'LSTM'
            path = f'bin/LSTM_DO_AdamW.pt'
        else: 
            print('Model number not recognized')
            exit()
        test_ppl = testing(model_path=path)
        print('Test ppl: ', test_ppl)
