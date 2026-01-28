# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import argparse
from functions import *
from utils import *
param ={

    'emb_size': 300,
    'hidden_size': 300,
    'dropout_prob': 0.5,

    'lr' : 0.0001,
    'clip': 5,
    'n_epochs': 200,
    'patience': 3,
    'multiple_runs': 5,

    'optimizer': 'Adam'  # 'SGD' or 'AdamW' or 'Adam'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--mode', type=str, default=None, help='train or inference')
    parser.add_argument('--model_arch', type=str, default='LSTM', help='LSTM')
    parser.add_argument('--bidirectional', action='store_true', default=False, help='use bidirectional LSTM')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--optimizer', type=str, default=None, help='SGD, AdamW, or Adam')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='dropout probability (for LSTM_DROPOUT)')
    parser.add_argument('--emb_size', type=int, default=300, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=300, help='hidden size')

    parser.add_argument('--multiple_runs', type=int, default=None, help='number of runs with different initialization')
    parser.add_argument('--experiment', type=str, default=None, help='experiment name for the saved model')

    args = parser.parse_args()

    if args.mode is None:
        print('Please provide a mode (train or inference)')
        exit()
    if args.mode == 'train':
        if args.model_arch is None:
            print('Please provide a model architecture (LSTM)')
            exit()
        if args.lr is None:
            print('Please provide a learning rate')
            exit()
        if args.optimizer is None:
            print('Please provide an optimizer (SGD, AdamW, or Adam)')
            exit()

        param['model_arch'] = args.model_arch
        param['lr'] = args.lr
        param['optimizer'] = args.optimizer
        if args.dropout_prob is not None:
            param['dropout_prob'] = args.dropout_prob
        if args.emb_size is not None:
            param['emb_size'] = args.emb_size
        if args.hidden_size is not None:
            param['hidden_size'] = args.hidden_size
        if args.multiple_runs is not None:
            param['multiple_runs'] = args.multiple_runs

        experiment = f"{param['model_arch']}_lr{param['lr']}_{param['optimizer']}"
        print("Training Mode")
        training(param, experiment)
    elif args.mode == 'inference':
        print("Inference Mode")
        print("Inference is not implemented in part_A yet.")
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'inference'")
