import torch
from model import LM_LSTM, LM_LSTM_DROP
from functions import *
from utils import * 
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
from functools import partial
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)

HID_SIZE = 300
EMB_SIZE = 300
N_EPOCHS = 100

# these flags can be changed to test different configurations

DROP = True
SGD = True
ADAM = False

SGD_LR = 1.5
ADAM_LR = 0.001

# batch sizes have improved performances with these values

TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 64 
TEST_BATCH_SIZE = 64

train_raw = read_file(os.path.join('dataset','PennTreeBank','ptb.train.txt'))
dev_raw = read_file(os.path.join('dataset','PennTreeBank','ptb.valid.txt'))
test_raw = read_file(os.path.join('dataset','PennTreeBank','ptb.test.txt'))

vocab = get_vocab(train_raw, ["<pad>", "<eos>"])            # vocab is computed only on training set, add two special tokens end of sentence and padding

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


# based on the optimizer we're using, we set the learning rate
if not ADAM:
    lr = SGD_LR
else:
    lr = ADAM_LR

clip = 5

vocab_len = len(lang.word2id)

# select the model (the one with dropout or the vanilla one)
if DROP:
    model = LM_LSTM_DROP(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
else:
    model = LM_LSTM(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

model.apply(init_weights)

# set the optimizer
if ADAM:
    optimizer = optim.AdamW(model.parameters(), lr=lr)
elif SGD:
    optimizer = optim.SGD(model.parameters(), lr=lr)

# cost function
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1, N_EPOCHS))
ppls_train = []               # logs
ppls_dev = []                 # logs
best_loss_dev = []

for epoch in pbar:
    ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
    ppls_train.append(ppl_train)
    losses_train.append(loss_train)
    sampled_epochs.append(epoch)
    losses_train.append(np.asarray(loss_train).mean())
    
    if epoch % 1 == 0:
        
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                # os.makedirs('bin', exist_ok=True)
                # torch.save(best_model.state_dict(), os.path.join('bin','best_model.pt'))
                patience = 3
        else:
            patience -= 1
            
        if patience <= 0: 
            break 

        best_loss_dev.append(loss_dev)
        
        ppls_dev.append(ppl_dev)
        losses_dev.append(loss_dev)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        pbar.set_description("PPL: %f" % ppl_dev)

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

ppls_dev.append(final_ppl)
ppls_train.append(final_ppl)

print('Best ppl: ', final_ppl)

# if we want to save results in csv and plot data, uncomment the two rows below

# create_next_test_folder("tests")
# save_results(ppls_dev, ppls_train, lr, epoch,  sampled_epochs, losses_dev, losses_train, DROP, ADAM)

print("TEST " + str(dir) + ":")

print(final_ppl)
