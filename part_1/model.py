# Import necessary libraries
import torch.nn as nn
import torch.nn.functional as F
from typing import *

# Define LM_LSTM class
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, batch_first=True):
        super(LM_LSTM, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        self.output = nn.Linear(hidden_size, output_size)

        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        
        lstm_out, _ = self.lstm(emb)
        
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

# Define LM_LSTM_DROP class
class LM_LSTM_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROP, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.embedding_dropout = nn.Dropout(emb_dropout)            # define dropout for the embedding layer
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        self.output = nn.Linear(hidden_size, output_size)
        
        self.output_dropout = nn.Dropout(out_dropout)               # define dropout for the output layer
        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.embedding_dropout(emb)
        
        lstm_out, _ = self.lstm(drop1)
        
        drop2 = self.output_dropout(lstm_out)
        
        output = self.output(drop2).permute(0, 2, 1)
        return output