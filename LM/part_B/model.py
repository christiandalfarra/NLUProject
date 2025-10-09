import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class WeightTiedLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
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

class VariationalDropoutLSTM_emb(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropoutemb = VariationalDropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        x = self.dropoutemb(x)  # Same mask applied to all timesteps
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden
    
class VariationalDropoutLSTM_last(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropoutlast = VariationalDropout(dropout_p)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropoutlast(x)
        x = self.fc(x)
        return x, hidden

class VariationalDropoutLSTM_emblast(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropoutemb = VariationalDropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropoutlast = VariationalDropout(dropout_p)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        x = self.dropoutemb(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropoutlast(x)
        x = self.fc(x)
        return x, hidden