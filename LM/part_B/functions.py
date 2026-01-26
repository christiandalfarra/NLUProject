# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import *
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def training(param, experiment):
    train_loader, dev_loader, test_loader, lang = getLoaders()
    vocab_len = len(lang.word2id)

    # Initialize the model
    model = LSTM(
        param['emb_size'], param['hidden_size'], vocab_len,pad_index=lang.word2id["<pad>"], emb_dropout=param['emb_dropout'],
        out_dropout=param['out_dropout'], tie_weights=param['weight_tying'], variational_dropout=param['var_dropout']
        ).to(DEVICE)
    
    model.apply(init_weights)
    # Initialize the optimizer, start with SGD for both cases
    optimizer = optim.SGD(model.parameters(), lr = param["lr"])
    ntavsgd_optimizer = param['optimizer'] == 'NTAvSGD'
    nt_triggered = False

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train = []
    losses_dev = []
    perplexity = []
    sampled_epochs = []
    best_ppl = math.inf

    best_model = None
    pbar = tqdm(range(1,param['n_epochs']))
    # used for NT-AvSGD
    logs = []

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, param['clip'])    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            if not ntavsgd_optimizer:
                # Standard evaluation if we are not using NT-AvSGD
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

            else:
                # NT-AvSGD specific evaluation
                # if the NT-AvSGD has been triggered we have to use the averaged weights for evaluation
                if nt_triggered:
                    temp_param = {}
                    # swap the parameters with their averaged version
                    for p in model.parameters():
                        temp_param[p] = p.data.clone()
                        p.data = optimizer.state[p]['ax'].clone()

                    # evaluate with the averaged weights
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    # swap back the parameters to continue training
                    for p in model.parameters():
                        p.data = temp_param[p].clone()
                else:
                    # standard evaluation before NT-AvSGD is triggered
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    # check if we have to trigger the NT-AvSGD
                    # t = len(logs)
                    # n = param['nt_interval']
                    if not nt_triggered and len(logs) > param['nt_interval'] and ppl_dev > min(logs[:-param['nt_interval']]):
                        # t0 = 0 because we want to start averaging right away
                        print("NT-AvSGD triggered")
                        nt_triggered = True
                        optimizer = optim.ASGD(model.parameters(), lr = param["lr"], t0=0)
                    logs.append(ppl_dev)
            
            losses_dev.append(np.asarray(loss_dev).mean())
            perplexity.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            # check the patience condition
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(DEVICE)
                patience = 3
            else:
                patience -= 1                    
            if patience <= 0:
                print("Early stopping")
                break

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)   
    print('Test ppl: ', final_ppl)
    #save weights
    path = f'bin/{experiment}.pt'
    torch.save(best_model.state_dict(), path)
    #plot the curves for the trainng models
    plot_loss(sampled_epochs, losses_train, losses_dev, f'plots/{experiment}_loss.png')
    plot_perplexity(sampled_epochs, perplexity, f'plots/{experiment}_ppl.png')

    return final_ppl

def testing(param, model_path):
    train_loader, dev_loader, test_loader, lang = getLoaders()
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]
    model = LSTM(param['emb_size'], param['hidden_size'], vocab_len, pad_index).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Set to evaluation mode (if inferencing)
    model.eval()
    print('Model loaded and set to evaluation mode.')
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)
    print('model tested', model_path)
    return final_ppl

def plot_loss(epochs, loss_train, loss_validation, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_train, label='Training Loss')
    ax.plot(epochs, loss_validation, label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)

def plot_perplexity(epochs, perplexity, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(epochs, perplexity, label='Validation PPL')
    ax.set_title('Validation PPL')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PPL')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)