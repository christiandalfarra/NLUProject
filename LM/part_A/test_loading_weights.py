import torch
import torch.nn as nn
import torch.nn.functional as F
from model import * 
from functions import *
from utils import getLoaders

DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def testing_RNN(emb_size, hid_size, vocab_len, pad_index, model_path, train_loader, dev_loader, test_loader):
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Set to evaluation mode (if inferencing)
    model.eval()
    print('Model loaded and set to evaluation mode.')
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)
    print('model tested', model_path)
    print('Test ppl: ', final_ppl)
    return final_ppl
def testing_LSTM(emb_size, hid_size, vocab_len, pad_index, model_path, train_loader, dev_loader, test_loader):
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Set to evaluation mode (if inferencing)
    model.eval()
    print('Model loaded and set to evaluation mode.')
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)
    print('model tested', model_path)
    print('Test ppl: ', final_ppl)
    return final_ppl

train_loader, dev_loader, test_loader, lang = getLoaders()
vocab_len = len(lang.word2id)
results = []

#testing_RNN(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp0_RNN_embsize350_hidsize350_lr0.1.pt', train_loader, dev_loader, test_loader)
#testing_RNN(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp1_RNN_embsize350_hidsize350_lr0.5.pt', train_loader, dev_loader, test_loader)
#testing_RNN(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp2_RNN_embsize350_hidsize350_lr0.05.pt', train_loader, dev_loader, test_loader)

#testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp3_LSTM_embsize350_hidsize350_lr0.1.pt', train_loader, dev_loader, test_loader)
#testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp4_LSTM_embsize350_hidsize350_lr0.5.pt', train_loader, dev_loader, test_loader)

#testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp6_LSTM_doemb0.5_lr0.1.pt', train_loader, dev_loader, test_loader)
#testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp7_LSTM_doemb0.5_lr0.5.pt', train_loader, dev_loader, test_loader)
testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp40_LSTM_DOEMB_LAST_LAYER_lr0.5.pt', train_loader, dev_loader, test_loader)
testing_LSTM(350, 350, vocab_len, lang.word2id["<pad>"], 'bin/exp50_LSTM_DOEMB_LAST_LAYER_lr1.pt', train_loader, dev_loader, test_loader)
testing_LSTM(350,350,vocab_len, lang.word2id["<pad>"], 'bin/exp50_LSTM_DOEMB_LAST_LAYER_lr0.0001_AdamW.pt', train_loader, dev_loader, test_loader)
testing_LSTM(350,350,vocab_len, lang.word2id["<pad>"], 'bin/exp50_LSTM_DOEMB_LAST_LAYER_lr0.0005_AdamW.pt', train_loader, dev_loader, test_loader)