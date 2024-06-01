# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from typing import *

# Define VariationalDropout class
class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None
        
    def forward(self, input, dropout=0.5):
        if not self.training or not dropout:
            return input
        if self.mask is None or self.mask.size() != input.size():
            m = torch.empty(input.size(), device=input.device).bernoulli_(1-dropout)
            self.mask = m/(1-dropout)
        return self.mask * input

# Define LM_LSTM class
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, batch_first=True, var_drop=False, weight_tying=False):
        super(LM_LSTM, self).__init__()

        self.var_drop = var_drop
        self.weight_tying = weight_tying
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if self.var_drop:
            self.input_drop = VariationalDropout()
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        self.output = nn.Linear(hidden_size, output_size)

        if self.var_drop:
            self.output_drop = VariationalDropout()
        
        if self.weight_tying:
            self.output.weight = self.embedding.weight
        self.pad_token = pad_index
    
    def forward(self, input_sequence):

        emb = self.embedding(input_sequence)

        if self.var_drop:
            emb = self.input_drop(emb)
        
        lstm_out, _ = self.lstm(emb)
        
        if self.var_drop:
            lstm_out = self.output_drop(lstm_out)
        
        output = self.output(lstm_out).permute(0, 2, 1)
        return output