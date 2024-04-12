import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import config
from torch.nn.utils.rnn import PackedSequence
from typing import *

class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, batch_first=True):
        super(LM_LSTM, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        if config.var_drop:
            self.input_drop = VariationalDropout(emb_dropout, batch_first=batch_first)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        self.output = nn.Linear(hidden_size, output_size)
        if config.var_drop:
            self.output_drop = VariationalDropout(out_dropout, batch_first=batch_first)
        
        self.output.weight = self.embedding.weight  # weight tying
        self.pad_token = pad_index
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if config.var_drop:
            emb = self.input_drop(emb)
        
        lstm_out, _ = self.lstm(emb)
        
        if config.var_drop:
            lstm_out = self.output_drop(lstm_out)
        
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
