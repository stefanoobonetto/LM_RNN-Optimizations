import torch
from model import LM_LSTM, LM_LSTM_DROP
from functions import read_file, Lang, PennTreeBank, create_next_test_folder, save_results, get_vocab, collate_fn
from utils import train_loop, eval_loop, init_weights
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import config
from functools import partial
import os
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # it can be changed with 'cpu' if you do not have a gpu
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' # it can be changed with 'cpu' if you do not have a gpu

train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

# Vocab is computed only on training set, add two special tokens end of sentence and padding
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

# Dataloader instantiation
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

if not config.adam:
    lr = 2.5                          # SGD = 2.5, AdamW = 0.001
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
else:
    if config.asgd:
        optimizer = optim.ASGD(model.parameters(), lr=lr) 
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

create_next_test_folder("tests")

L = config.n_epochs
n = 5
k = 0
t = 0
T = 0
n_epochs = config.n_epochs
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1, n_epochs))
ppls = []               # logs

for epoch in pbar:
    loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
    if epoch % L == 0 and T == 0:
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        losses_dev.append(np.asarray(loss_dev).mean())
        pbar.set_description("PPL: %f" % ppl_dev)
        
        if t > n and ppl_dev > min(ppls[max(0, t-n-1):t]):
            T = k
        ppls.append(ppl_dev)
        t += 1

        if  ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0:
            break
    
    k += 1


best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

ppls.append(final_ppl)

print('Test ppl: ', final_ppl)

save_results(ppls, lr_initial, lr, epoch,  sampled_epochs, losses_dev, losses_train)

print("TEST " + str(dir) + ":")
# print(ppls)
print(final_ppl)

# print("Learning Rate: " + str(lr))
# plt.xlabel("Epoch")
# plt.ylabel("PPL")
# plt.grid()
# plt.plot(ppls)