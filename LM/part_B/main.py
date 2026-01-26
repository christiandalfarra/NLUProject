# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import argparse
param ={
    'emb_size': 350,
    'hidden_size': 350,
    'emb_dropout': 0.0,
    'out_dropout': 0.0,

    'lr' : 0.5,
    'clip': 5,
    'n_epochs': 100,
    'patience': 3,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--mode', type=str, default=None, help='train or evaluate, for training specify also model_arch, lr, optimizer adn also the values for dropout if u wnat to use it, for evaluation specify model_number')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--optimizer', type=str, default=None, help='SGD or NTAvSGD')
    parser.add_argument('--weight_tying', action='store_true', default=False, help='use weight tying or not')
    parser.add_argument('--var_dropout', action='store_true', default=False, help='use variational dropout or not')
    parser.add_argument('--emb_dropout', type=float, default=0.0, help='embedding dropout probability')
    parser.add_argument('--out_dropout', type=float, default=0.0, help='output dropout probability')
    parser.add_argument('--nt_interval', type=int, default=3, help='NT-AvSGD interval for non-monotonicity check')

    parser.add_argument('--model_number', type=int, default=None, help='model number for evaluation')

    args = parser.parse_args()

    if args.mode is None:
        print('Please provide a mode (train or evaluate)')
        exit()
    if args.mode == 'train':
        if args.lr is None:
            print('Please provide a learning rate')
            exit()
        if args.optimizer is None:
            print('Please provide an optimizer (SGD or NTAvSGD)')
            exit()
        if args.optimizer not in ['SGD', 'NTAvSGD']:
            print('Optimizer must be either "SGD" or "NTAvSGD"')
            exit()
        param['lr'] = args.lr
        param['optimizer'] = args.optimizer if args.optimizer is not None else param['optimizer']
        param['weight_tying'] = args.weight_tying
        param['var_dropout'] = args.var_dropout
        if args.var_dropout:
            param['emb_dropout'] = args.emb_dropout
            param['out_dropout'] = args.out_dropout
        if param['optimizer'] == 'NTAvSGD':
            param['nt_interval'] = args.nt_interval
            param['patience'] = 5
        training(param, experiment=f"exp_{'WT' if param['weight_tying'] else ''}_{'VD' if param['var_dropout'] else ''}_lr{param['lr']}_{param['optimizer']}")
    elif args.mode == 'evaluate':
        exp_number = args.model_number
        if exp_number is None:
            print('Please provide a model number to evaluate')
            exit()
        if exp_number == 1:
            path = f'bin/first.pt'
        elif exp_number == 2:
            path = f'bin/second.pt'
        elif exp_number == 3:
            path = f'bin/third.pt'
        elif exp_number == 4:
            path = f'bin/fourth.pt'
        else: 
            print('Model number not recognized')
            exit()
        test_ppl = testing(param, model_path=path)
        print('Test ppl: ', test_ppl)
    else:
        raise ValueError("Mode not recognized. Available modes: 'train' and 'evaluate'")
