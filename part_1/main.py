import torch
from model import LM_LSTM, LM_LSTM_DROP
from functions import train_loop, eval_loop, init_weights, create_next_test_folder, save_results
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn 
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import config
from functools import partial
import os
import time
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # it can be changed with 'cpu' if you do not have a gpu

train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

vocab = get_vocab(train_raw, ["<pad>", "<eos>"])            # vocab is computed only on training set, add two special tokens end of sentence and padding

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config.dev_batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

if not config.adam:
    lr = config.sgd_lr                          # SGD = 2.5, AdamW = 0.001
else:
    lr = 0.001

lr_initial = lr
clip = 5

vocab_len = len(lang.word2id)

if config.drop:
    model = LM_LSTM_DROP(config.emb_size, config.hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
else:
    model = LM_LSTM(config.emb_size, config.hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

model.apply(init_weights)

if config.adam:
    optimizer = optim.AdamW(model.parameters(), lr=lr)
elif config.sgd:
    optimizer = optim.SGD(model.parameters(), lr=lr)

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

create_next_test_folder("tests")

n_epochs = config.n_epochs
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1, n_epochs))
ppls_train = []               # logs
ppls_dev = []               # logs
best_loss_dev = []
stored_loss = 100000000

for epoch in pbar:
    epoch_start_time = time.time()
    ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
    ppls_train.append(ppl_train)
    losses_train.append(loss_train)
    sampled_epochs.append(epoch)
    losses_train.append(np.asarray(loss_train).mean())
    
    if epoch % 1 == 0:
        
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        string = "Playing with normal SGD"

        if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

        best_loss_dev.append(loss_dev)
        
        ppls_dev.append(ppl_dev)
        losses_dev.append(loss_dev)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        pbar.set_description(string + " - PPL: %f" % ppl_dev)


best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

ppls_dev.append(final_ppl)
ppls_train.append(final_ppl)

print('Best ppl: ', final_ppl)

save_results(ppls_dev, ppls_train, lr_initial, lr, epoch,  sampled_epochs, losses_dev, losses_train)

print("TEST " + str(dir) + ":")

print(final_ppl)
