import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, emb_size, hid_size, out_slot, out_int, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout_prob=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)  

        self.slot_out = nn.Linear(hid_size * (2 if bidirectional else 1), out_slot)
        self.intent_out = nn.Linear(hid_size * (2 if bidirectional else 1), out_int)

        # Dropout layer How/Where do we apply it?
        if dropout_prob > 0:
            self.dropout = True
            self.dropout_embedding = nn.Dropout(dropout_prob)
            self.dropout_output = nn.Dropout(dropout_prob)
        else:
            self.dropout = False
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # Dropout input layer if applied
        if self.dropout:
            utt_emb = self.dropout_embedding(utt_emb)
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            # If bidirectional, we need to concat the last hidden states from both directions
            last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim = 1)
        else:
            # Get the last hidden state
            last_hidden = last_hidden[-1,:,:]

        # Dropout output layer if applied
        if self.dropout:
            last_hidden = self.dropout_output(last_hidden)
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent