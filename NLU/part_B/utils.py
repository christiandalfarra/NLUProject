# Add functions or classes used for data loading and preprocessing
import json
import torch
import torch.utils.data as data
import os
from pprint import pprint
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

PAD_TOKEN = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents=None, slots=None, cutoff=0, slot2id=None, intent2id=None):
        if words is None:
            words = []
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = slot2id if slot2id is not None else self.lab2id(slots)
        self.intent2id = intent2id if intent2id is not None else self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        vocab['unk'] = len(vocab)
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    # add the tokenizer because Bert use a different tokeinization method
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.unk = unk
        
        for data in dataset:
            self.utterances.append(data['utterance'])
            self.slots.append(data['slots'])
            self.intents.append(data['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    # Map utterance and slots to their IDs 
    def mapping_seq(self, utt_list, slot_list, mapper):
        utt_ids = []
        slot_ids = []

        for utterance, slots in zip(utt_list, slot_list):
            # classic tokenization for slots and words
            word_list = utterance.split()
            slot_list_split = slots.split()
            
            token_ids = []
            slot_id_seq = []

            for word, slot_label in zip(word_list, slot_list_split):
                # Tokenize each word using BERT tokenizer
                tokensBert = self.tokenizer(word)['input_ids'][1:-1]  # Exclude [CLS] and [SEP]
                token_ids.extend(tokensBert)
                
                # first token get the slot label, other tokens get the pad token
                slot_id_seq.extend([mapper[slot_label]] + [PAD_TOKEN] * (len(tokensBert) - 1))
            
            utt_ids.append(token_ids)
            slot_ids.append(slot_id_seq)
                    
        return utt_ids, slot_ids
        

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix

        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    # Attention mask for BERT
    # 1 where there is a token, 0 where there is PAD
    attention_mask = torch.LongTensor([[1 if token != PAD_TOKEN else 0 for token in seq] for seq in src_utt])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attention_mask"] = attention_mask

    return new_item

def get_dataloaders(slot2id=None, intent2id=None):
    tmp_train_raw = load_data(os.path.join("..", "dataset", "train.json"))
    test_raw = load_data(os.path.join("..", "dataset", "test.json"))

    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw

    # Sorting makes label ids deterministic across runs.
    slots = sorted(set(sum([line['slots'].split() for line in corpus],[])))
    intents = sorted(set([line['intent'] for line in corpus]))

    lang = Lang(words, intents, slots, cutoff=0, slot2id=slot2id, intent2id=intent2id)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, lang
