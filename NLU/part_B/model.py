import torch.nn as nn
from transformers import BertModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

class BertIAS(nn.Module):
    def __init__(self, hidden_size, slot_out, intent_out, dropout_prob=0.1):
        super(BertIAS, self).__init__()

        #get the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.slot_out = nn.Linear(hidden_size, slot_out)
        self.intent_out = nn.Linear(hidden_size, intent_out)

        # Dropout layer as regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state
        pooled_output = output.pooler_output

        # Dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Compute slot logits
        slots = self.slot_out(sequence_output)
        # Compute intent logits
        intent = self.intent_out(pooled_output)

        slots = slots.permute(0,2,1) 

        return slots, intent