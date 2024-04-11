import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from weight_drop import WeightDrop
import numpy as np

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, wdrop=0):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        if wdrop:
            self.lstm = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight  # weight tying
        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
class LM_LSTM_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROP, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.embedding_dropout = nn.Dropout(emb_dropout)  # dropout embedding layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

        self.output.weight = self.embedding.weight  # weight tying
        
        self.output_dropout = nn.Dropout(out_dropout)  # dropout output layer

        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.embedding_dropout(emb)  # dropout embedding layer
        lstm_out, _ = self.lstm(drop1)
        drop2 = self.output_dropout(lstm_out)  # dropout output layer
        output = self.output(drop2).permute(0, 2, 1)
        return output
