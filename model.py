import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class Word_BiLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hid_dim*2, 1)  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        #embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #outputs are always from the top hidden layer
        
        hidden_result = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        result=self.fc(hidden_result.squeeze(0))
        
        return hidden,result
    
class Post_BiLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers,bidirectional=bidirectional,dropout=dropout,)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #x=[event_len,batch,hid_dimention]
        output, (hidden, cell) = self.rnn(x)
        #hidden=[n layers * n directions, batch size, hid dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #hidden=[batch size, n layers * n directions*hid dim]
        result=self.fc(hidden.squeeze(0))
        #result=[batch_size,1]
        return result
    
    

