import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)  
        self.pad_token = pad_index 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

class LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.0, out_dropout=0.0,
                  n_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        if emb_dropout > 0:
            self.dropout_emb = nn.Dropout(emb_dropout)  # Dropout after embedding
            self.embdropout = True
        else:
            self.embdropout = False
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) 
        if out_dropout > 0:
            self.dropout_out = nn.Dropout(out_dropout) # Dropout before final linear layer
            self.outdropout = True
        else:
            self.outdropout = False
        self.pad_token = pad_index 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.embdropout:
            emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        if self.outdropout:
            lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output