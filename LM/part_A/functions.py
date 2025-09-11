import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import numpy as np
from tqdm import tqdm
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

def training_SGD(hid_size,emb_size,lr,clip,n_epochs, patience,experiment):
    train_loader, dev_loader, test_loader, lang = getLoaders()
    vocab_len = len(lang.word2id)

    model = LM_LSTM_DROP_EMB_LAYER(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    #optimizer da cambiare
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 5 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
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
    
    return final_ppl

def training_AdamW(hid_size,emb_size,lr,clip,n_epochs, patience,experiment):
    train_loader, dev_loader, test_loader, lang = getLoaders()
    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    #optimizer da cambiare
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 5 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
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
    
    return final_ppl

def grid_search_hyperparameters_RNN(hid_sizes, emb_sizes, lrs, clip, n_epochs, patience):
    results = []
    i = 0
    for lr in lrs:
        for hid_size in hid_sizes:
            for emb_size in emb_sizes:
                result = training_SGD(hid_size, emb_size, lr, clip, n_epochs, patience, f'exp{i}_RNN_embsize{emb_size}_hidsize{hid_size}_lr{lr}')
                results.append(result)
                print(result)
                i += 1
    return results

class NTAvSGD:
    def __init__(self, params, lr=30, n=5, L=1000):
        self.params = list(params)
        self.lr = lr
        self.n = n  # Non-monotone interval
        self.L = L  # Logging interval (iterations per evaluation)
        self.optimizer = torch.optim.SGD(self.params, lr=lr)
        
        # Tracking variables
        self.iteration = 0
        self.best_val_loss = float('inf')
        self.val_losses = []
        self.weights = []
        self.averaging = False
        self.trigger_count = 0
        
    def step(self, closure=None):
        self.optimizer.step(closure)
        self.iteration += 1
        
        # Store weights if we're in the averaging phase
        if self.averaging:
            self.weights.append([p.data.clone() for p in self.params])
        
    def update_val_loss(self, val_loss):
        """Call this after each validation evaluation"""
        self.val_losses.append(val_loss)
        
        # Check if we should start averaging
        if not self.averaging:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.trigger_count = 0
            else:
                self.trigger_count += 1
                
            # Non-monotonic trigger condition
            if self.trigger_count >= self.n:
                self.start_averaging()
                
    def start_averaging(self):
        """Begin weight averaging"""
        self.averaging = True
        self.weights = []
        
    def get_averaged_weights(self):
        """Returns averaged weights if averaging has been triggered"""
        if not self.averaging or len(self.weights) == 0:
            return None
            
        # Average all stored weights
        avg_weights = []
        for i in range(len(self.weights[0])):
            avg = torch.stack([w[i] for w in self.weights]).mean(0)
            avg_weights.append(avg)
            
        return avg_weights