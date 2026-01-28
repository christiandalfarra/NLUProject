# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import argparse
from functions import *
from utils import *
param ={

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 200,
    'patience': 3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--mode', type=str, default=None, help='train or evaluate, for training specify also model_arch, lr, optimizer and also the values for dropout if u want to use it, for evaluation specify model_number')
    parser.add_argument('--model_arch', type=str, default='LSTM', help='LSTM')
    parser.add_argument('--bidirectional', action='store_true', default=False, help='use bidirectional LSTM')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD, AdamW, or Adam')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='dropout probability')
    parser.add_argument('--emb_size', type=int, default=300, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size')
    parser.add_argument('--model_number', type=int, default=None, help='model number for evaluation')

    parser.add_argument('--multiple_runs', type=int, default=5, help='number of runs with different initialization')

    args = parser.parse_args()

    if args.mode is None:
        print('Please provide a mode --mode (train or evaluate)')
        exit()
    if args.mode == 'train':
        if args.model_arch is None:
            print('Please provide a model architecture --model_arch (LSTM)')
            exit()
        if args.lr is None:
            print('Please provide a learning rate --lr value')
            exit()
        if args.optimizer is None:
            print('Please provide an optimizer --optimizer (SGD, AdamW, or Adam)')
            exit()

        param['model_arch'] = args.model_arch
        param['lr'] = args.lr
        param['optimizer'] = args.optimizer

        param['bidirectional'] = args.bidirectional

        if args.dropout_prob is not None:
            param['dropout_prob'] = args.dropout_prob
        if args.emb_size is not None:
            param['emb_size'] = args.emb_size
        if args.hidden_size is not None:
            param['hidden_size'] = args.hidden_size
        if args.multiple_runs is not None:
            param['multiple_runs'] = args.multiple_runs

        experiment = f"{param['model_arch']}_{'BiDir' if param['bidirectional'] else 'UniDir'}_{'DO' if param['dropout_prob'] > 0 else 'NoDO'}_lr{param['lr']}_{param['optimizer']}"
        print("Training Mode")
        training(param, experiment)
    elif args.mode == 'evaluate':
        print("Evaluation Mode")
        exp_number = args.model_number
        if exp_number is None:
            print('Please provide a model number to evaluate --model_number value (1,2,3)')
            exit()
        if exp_number == 1:
            path = f'bin/LSTM_UniDir_NoDO_lr0.0001_Adam.pt'
        elif exp_number == 2:
            path = f'bin/LSTM_BiDir_NoDO_lr0.0001_Adam.pt'
        elif exp_number == 3:
            path = f'bin/LSTM_BiDir_DO_lr0.0001_Adam.pt'
        else: 
            print('Model number not recognized')
            exit()
        testing(path)
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'evaluate'")
