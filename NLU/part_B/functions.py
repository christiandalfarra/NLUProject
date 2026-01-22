# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from conll import evaluate
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from model import BertIAS
import numpy as np
import copy
from transformers import AutoTokenizer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['attention_mask'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 
        loss_array.append(loss.item())
        loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['attention_mask'])

            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
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
                utt_ids = sample['utterance'][id_seq].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()

                # Get the original words ids using the tokenizer
                tokens = tokenizer.convert_ids_to_tokens(utt_ids)

                # Prepare for evaluation, remove padding
                tmp_ref = []
                tmp_hyp = []

                for i, gt_id in enumerate(gt_ids):
                    if gt_id != PAD_TOKEN:
                        tmp_ref.append((tokens[i], lang.id2slot[gt_id]))
                        tmp_hyp.append((tokens[i], lang.id2slot[seq[i].item()]))
                ref_slots.extend(tmp_ref)
                hyp_slots.extend(tmp_hyp)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print("ERRORROEEEEEEEEEE")
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def training(params, experiment):
    train_loader, dev_loader, test_loader, lang = get_dataloaders()

    vocab_len = len(lang.word2id)
    out_slot = len(lang.slot2id)
    out_intent = len(lang.intent2id)

    model = BertIAS(hidden_size=params['hidden_size'], 
                    slot_out=out_slot, 
                    intent_out=out_intent, 
                    dropout_prob=params['dropout_prob']).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    epochs = params['n_epochs']
    patience = params['patience']
    clip = params['clip']

    slots_f1 = []
    intents_acc = []

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    best_model = None

    for epoch in tqdm(range(0, epochs)):
        loss_train = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        if epoch % 5 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss_train).mean())
            results_dev, report_intent_dev, loss_array_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_array_dev).mean())

            f1 = results_dev['total']['f']
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
                patience = params['patience'] # reset patience if we have a new best model
            else:
                patience -= 1
            if patience <= 0:
                print("Early stopping triggered\n")
                break
            slots_f1.append(f1)
            intents_acc.append(report_intent_dev['accuracy'])
    model.to(DEVICE)
    
    # Evaluate on the test set
    results_test, report_intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    print('Slot F1', results_test['total']['f'])
    print('Intent Acc', report_intent_test['accuracy'])
    torch.save(best_model.state_dict(), f'bin/{experiment}.pt')


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

def plot_f1_and_accuracy(epochs, f1_list, acc_list, name):
    fig, ax = plt.subplots()
    ax.plot(epochs, f1_list, label='Validation F1 score')
    ax.plot(epochs, acc_list, label='Validation accuracy')
    ax.set_title('Validation F1 and Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(name)