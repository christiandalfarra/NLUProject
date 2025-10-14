import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BertIAS(nn.Module):
    def __init__(self, bert_model_name, hid_size, out_slot, out_int, dropout=0.1):
        super(BertIAS, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hid_size = hid_size
        self.out_slot = out_slot
        self.out_int = out_int

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Get the last hidden state
        utt_encoded = outputs.last_hidden_state
        # Get the [CLS] token representation for intent classification
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        drop_utt = self.dropout(utt_encoded)
        drop_output = self.dropout(pooled_output)

        # Compute slot logits
        slots = self.slot_out(drop_utt)
        # Compute intent logits
        intent = self.intent_out(drop_output)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        
        return slots, intent
