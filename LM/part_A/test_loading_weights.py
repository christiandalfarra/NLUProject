import torch
import torch.nn as nn
import torch.nn.functional as F
from model import * 
from functions import *
from utils import getLoaders

DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def testing(emb_size, hid_size, vocab_len, pad_index, model_path, train_loader, dev_loader, test_loader):
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Set to evaluation mode (if inferencing)
    model.eval()
    print('Model loaded and set to evaluation mode.')

    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)

    print('Test ppl: ', final_ppl)
    
    
emb_size = 300     # example value
hid_size = 200 

train_loader, dev_loader, test_loader, lang = getLoaders()
vocab_len = len(lang.word2id)

testing(350, 200, vocab_len, lang.word2id['<pad>'], 'bin/RNN_embsize350_hidsize200.pt', train_loader, dev_loader, test_loader)
testing(250, 200, vocab_len, lang.word2id['<pad>'], 'bin/RNN_embsize250_hidsize200.pt', train_loader, dev_loader, test_loader)
testing(300, 150, vocab_len, lang.word2id['<pad>'], 'bin/RNN_embsize300_hidsize150.pt', train_loader, dev_loader, test_loader)
testing(300, 250, vocab_len, lang.word2id['<pad>'], 'bin/RNN_embsize300_hidsize250.pt', train_loader, dev_loader, test_loader)