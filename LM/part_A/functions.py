import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import *
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def plot_loss(epochs, loss_train, loss_validation, path):
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
    fig, ax = plt.subplots()
    ax.plot(epochs, perplexity, label='Validation PPL')
    ax.set_title('Validation PPL')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PPL')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)

def training(param, experiment):
    train_loader, dev_loader, test_loader, lang = getLoaders()
    vocab_len = len(lang.word2id)

    arch = param['model_arch']
    if arch == 'RNN':
        model = LM_RNN(
            param['emb_size'],
            param['hidden_size'],
            vocab_len,
            pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)

    elif arch == 'LSTM':
        model = LM_LSTM(
            param['emb_size'],
            param['hidden_size'],
            vocab_len,
            pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)

    elif arch == 'LSTM_DOEMB_LAYER':
        model = LM_LSTM_DROP_EMB_LAYER(
            param['emb_size'],
            param['hidden_size'],
            vocab_len,
            pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)

    elif arch == 'LSTM_DOEMB_LAST_LAYER':
        model = LM_LSTM_DROP_EMB_LAST_LAYER(
            param['emb_size'],
            param['hidden_size'],
            vocab_len,
            pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)

    else:
        raise ValueError("Architecture not recognized. Available architectures: RNN, LSTM, LSTM_DOEMB_LAYER, LSTM_DOEMB_LAST_LAYER")
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=param['lr']) if param['optimizer'] == 'AdamW' else optim.SGD(model.parameters(), lr=param['lr'])
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    losses_train = []
    losses_dev = []
    perplexity = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,param['n_epochs']))
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, param['clip'])    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            perplexity.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(DEVICE)
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:
                break

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)   
    print('Test ppl: ', final_ppl)
    #save weights
    path = f'bin/{experiment}.pt'
    torch.save(model.state_dict(), path)
    #plot the curves for the trainng models
    plot_loss(sampled_epochs, losses_train, losses_dev, f'plots/{experiment}_loss.png')
    plot_perplexity(sampled_epochs, perplexity, f'plots/{experiment}_ppl.png')

    return final_ppl

def grid_search_hyperparameters_RNN(param):
    results = []
    i = 0
    for lr in param['lr']:
        for hid_size in param['hid_size']:
            for emb_size in param['emb_size']:
                result = (param, f'exp{i}_RNN_embsize{emb_size}_hidsize{hid_size}_lr{lr}')
                results.append(result)
                print(result)
                i += 1
    return results

