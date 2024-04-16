# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import config 
from torch.nn.utils.rnn import PackedSequence
from typing import *

# Define VariationalDropout class
class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        # Initialize dropout rate and batch_first flag
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If not in training mode or dropout is 0, return the input as is
        if not self.training or self.dropout <= 0.:
            return x

        # Check if input is a PackedSequence
        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            # If it is, unpack it
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            # If it's not, set batch_sizes to None and get the batch size from x
            batch_sizes = None
            max_batch_size = x.size(0)

        # Apply the same dropout mask across the entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        # If input was a PackedSequence, return it as such
        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

# Define LM_LSTM class
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, batch_first=True):
        super(LM_LSTM, self).__init__()
        
        # Define embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # If variational dropout is enabled in the config, initialize VariationalDropout
        if config.var_drop:
            self.input_drop = VariationalDropout(emb_dropout, batch_first=batch_first)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # Define output layer
        self.output = nn.Linear(hidden_size, output_size)
        # If variational dropout is enabled in the config, initialize VariationalDropout for output
        if config.var_drop:
            self.output_drop = VariationalDropout(out_dropout, batch_first=batch_first)
        
        # Tie the weights of the output and embedding layers
        self.output.weight = self.embedding.weight
        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        # Pass the input through the embedding layer
        emb = self.embedding(input_sequence)
        # If variational dropout is enabled, apply it
        if config.var_drop:
            emb = self.input_drop(emb)
        
        # Pass the output of the embedding layer through the LSTM
        lstm_out, _ = self.lstm(emb)
        
        # If variational dropout is enabled, apply it to the LSTM output
        if config.var_drop:
            lstm_out = self.output_drop(lstm_out)
        
        # Pass the LSTM output through the output layer and permute the dimensions
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

# Define LM_LSTM_DROP class
class LM_LSTM_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROP, self).__init__()
        # Define embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Define dropout for the embedding layer
        self.embedding_dropout = nn.Dropout(emb_dropout)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # Define output layer
        self.output = nn.Linear(hidden_size, output_size)
        # Tie the weights of the output and embedding layers
        self.output.weight = self.embedding.weight
        
        # Define dropout for the output layer
        self.output_dropout = nn.Dropout(out_dropout)
        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        # Pass the input through the embedding layer and apply dropout
        emb = self.embedding(input_sequence)
        drop1 = self.embedding_dropout(emb)
        
        # Pass the output of the embedding layer through the LSTM
        lstm_out, _ = self.lstm(drop1)
        
        # Apply dropout to the LSTM output
        drop2 = self.output_dropout(lstm_out)
        
        # Pass the LSTM output through the output layer and permute the dimensions
        output = self.output(drop2).permute(0, 2, 1)
        return output