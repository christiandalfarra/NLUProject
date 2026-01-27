import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.1,
                 out_dropout=0.1, n_layers=1, tie_weights=False, variational_dropout = False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)  
        self.pad_token = pad_index 
        self.output = nn.Linear(hidden_size, output_size)

        self.var_dropout = variational_dropout

        if variational_dropout:
            self.emb_dropout = VariationalDropout(emb_dropout)
            self.out_dropout = VariationalDropout(out_dropout)

        if tie_weights:
            self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        self.lstm.flatten_parameters()
        emb = self.embedding(input_sequence)
        # apply variational dropout if specified
        if self.var_dropout:
            emb = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        # apply variational dropout if specified
        if self.var_dropout:
            lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output

class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        # Sample mask once per forward/backward pass
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)

        #expand to the full input size
        mask = mask.expand_as(x)
        return x * mask