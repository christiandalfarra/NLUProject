import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)  
        self.pad_token = pad_index 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)  
        self.pad_token = pad_index 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    
class LM_LSTM_DROP_EMB_LAYER(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, emb_dropout=0.5):
        super(LM_LSTM_DROP_EMB_LAYER,self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout = nn.Dropout(emb_dropout)  # Dropout after embedding
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout(emb)  # Apply dropout
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
class LM_LSTM_DROM_EMB_LAST_LAYER(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, emb_dropout=0.5, out_dropout=0.2):
        super(LM_LSTM_DROP_EMB_LAYER, self).__init__()  # Fixed class name here
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)  # Dropout after embedding
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.dropout_out = nn.Dropout(out_dropout)  # Dropout before final linear layer
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout_out(lstm_out)  # Apply dropout before final layer
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
class WeightTiedLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Tie weights between embedding and fc layers
        self.fc.weight = self.embed.weight
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        # Sample mask once per forward/backward pass
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

class VariationalDropoutLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_p=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = VariationalDropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        x = self.dropout(x)  # Same mask applied to all timesteps
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden