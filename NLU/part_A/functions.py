# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from conll import evaluate
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import copy
from tqdm import tqdm
from model import *

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

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        utterances = sample['utterances']
        slots_len = sample['slots_len']
        intents = sample['intents']
        y_slots = sample['y_slots']
        slots, intent = model(utterances, slots_len)
        loss_intent = criterion_intents(intent, intents)
        loss_slot = criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    device = next(model.parameters()).device
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            utterances = sample['utterances'].to(device)
            slots_len = sample['slots_len'].to(device)
            intents_t = sample['intents'].to(device)
            y_slots_t = sample['y_slots'].to(device)
            slots, intents = model(utterances, slots_len)
            loss_intent = criterion_intents(intents, intents_t)
            loss_slot = criterion_slots(slots, y_slots_t)
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(param, experiment):
    train_loader, dev_loader, test_loader, lang = get_dataloaders()
    vocab_len = len(lang.word2id)
    out_slot = len(lang.slot2id)
    out_intent = len(lang.intent2id)

    epochs = param['n_epochs']
    runs = param['multiple_runs']
    patience = param['patience']
    clip = param['clip']

    slots_f1 = []
    intents_acc = []
    
    # Track best model across all runs
    best_model_overall = None
    best_f1_overall = 0

    for run in tqdm(range(0, runs)):
        print(f"\nRun {run+1}/{runs}\n")
        #Create the model
        model = ModelIAS(param['emb_size'], param['hidden_size'], out_slot, out_intent, vocab_len, 
                         bidirectional=param.get('bidirectional', False),
                         dropout_prob=param.get('dropout_prob', 0))
    
        model.apply(init_weights)
        # Optimizer
        if param['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'])
        elif param['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'])
        elif param['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
        else:
            raise ValueError("Optimizer not recognized. Available optimizers: SGD, AdamW, Adam")

        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None
        current_patience = patience  # Use a separate variable for current run's patience

        for epoch in tqdm(range(0, epochs)):
            loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)
            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss_train).mean())
                results_dev, report_intent_dev, loss_array_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_array_dev).mean())
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model)
                    current_patience = patience  # reset patience if we have a new best model
                else:
                    current_patience -= 1
                if current_patience <= 0:
                    print("Early stopping triggered\n")
                    break
        
        if best_model is None:
            best_model = model
        
        best_model.to(DEVICE)
        results_test, report_intent_test, loss_array_test = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
        
        # Get F1 score for this run
        run_f1 = results_test['total']['f']
        slots_f1.append(run_f1)
        intents_acc.append(report_intent_test['accuracy'])
        
        # Update overall best model if this run's model is better
        if run_f1 > best_f1_overall:
            best_f1_overall = run_f1
            best_model_overall = copy.deepcopy(best_model)
            print(f"New best model found in run {run+1} with F1: {best_f1_overall:.3f}")
    
    slots_f1 = np.asarray(slots_f1)
    intents_acc = np.asarray(intents_acc)
    
    # Stat print across all runs
    print('Statistics across all runs:')
    print('Slot F1:', round(slots_f1.mean(),3), '+-', round(slots_f1.std(),3))
    print('Intent Acc:', round(intents_acc.mean(), 3), '+-', round(intents_acc.std(), 3))
    print('Best F1 across all runs:', round(best_f1_overall, 3))
    
    # Final evaluation of the best model on test set
    print('Final evaluation of best model on test set:')
    if best_model_overall is not None:
        best_model_overall.to(DEVICE)
        results_test_final, report_intent_test_final, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model_overall, lang)
        
        print(f"Slot F1: {results_test_final['total']['f']:.3f}")
        print(f"Intent Accuracy: {report_intent_test_final['accuracy']:.3f}")
        
        # Save the best model
        saving_obj = {
            'model_state_dict': best_model_overall.state_dict(),
            'params': param
        }
        torch.save(saving_obj, f'bin/{experiment}.pt')
        print(f"\nBest model saved to bin/{experiment}.pt")
    else:
        print("Warning: No best model found across runs")
                
def testing(path_to_model):
    # Load data
    train_loader, dev_loader, test_loader, lang = get_dataloaders()
    vocab_len = len(lang.word2id)
    out_slot = len(lang.slot2id)
    out_intent = len(lang.intent2id)
    
    # Load the saved model
    print(f"Loading model from {model_path}")
    saved_model = torch.load(model_path, map_location=DEVICE)
    
    model_state_dict = saved_model['model_state_dict']
    saved_params = saved_model['params']
    
    # Create the model with the same architecture
    model = ModelIAS(
        saved_params['emb_size'], 
        saved_params['hidden_size'], 
        out_slot, 
        out_intent, 
        vocab_len,
        bidirectional=saved_params.get('bidirectional', False),
        dropout_prob=saved_params.get('dropout_prob', 0)
    )
    
    # Load the trained weights
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    
    # Define loss criteria
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results_test, report_intent_test, loss_array_test = eval_loop(
        test_loader, 
        criterion_slots, 
        criterion_intents, 
        model, 
        lang
    )
    
    # Print results
    print('\n' + '='*50)
    print('Test Set Results:')
    print(f"Slot F1: {results_test['total']['f']:.3f}")
    print(f"Intent Accuracy: {report_intent_test['accuracy']:.3f}")
    print('='*50 + '\n')
    
    return results_test, report_intent_test
   
