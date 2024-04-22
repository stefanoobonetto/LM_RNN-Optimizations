global drop
global adam
global n_epochs
global emb_size
global hid_size

# wdrop = 0.5
hid_size = 300
emb_size = 300
n_epochs = 100
nonmono = 3             # n var in the paper

drop = False
sgd = True
adam = False
asgd = True
weight_tying = True
var_drop = True

train_batch_size = 32
dev_batch_size = 64 
test_batch_size = 64
