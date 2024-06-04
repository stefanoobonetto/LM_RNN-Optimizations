import torch
import matplotlib.pyplot as plt
import csv
import os
import math
import torch.nn as nn

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() 
        output = model(sample['source'])
        loss = criterion(output, sample['target'])                   # compute loss on the training set (prediction, target)
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() 

    # calculate PPL
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))

    return ppl, sum(loss_array)/sum(number_of_tokens)           # return PPL and avg loss

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])          # compute loss on the validation set (prediction, target) 
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    # calculate PPL
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))      

    return ppl, sum(loss_array) / sum(number_of_tokens)         # return ppl and avg loss

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# functions to save results (csv and plots)

def create_next_test_folder(base_dir):
    global n, dir  

    if os.path.exists(base_dir):
        print("Folder exists:", base_dir)
    else:
        os.makedirs(base_dir)
        print("Created folder:", base_dir)

    existing_folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    last_number = 0
    for folder in existing_folders:
        if folder.startswith("test_"):
            try:
                number = int(folder.split("_")[1])
                last_number = max(last_number, number)
            except ValueError:
                pass

    n = last_number

    next_folder = os.path.join(base_dir, f"test_{last_number + 1}")
    os.makedirs(next_folder, exist_ok=True)
    dir = next_folder
    print(f"Created folder: {next_folder}")

def plot_line_graph(data1, data2, sampled_epochs, losses_dev, losses_train, filename, filename1):

    y1 = data1[:-1]  # last val is best_ppl, don't want to plot it 
    y2 = data2[:-1]  
    
    x1 = list(range(1, len(y1) + 1))  # indx + 1
    x2 = list(range(1, len(y2) + 1))  # indx + 1
    

    plt.plot(x1, y1, label='PPL valuation')
    plt.plot(x2, y2, label='PPL training')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.title('PPLs for each epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

    y1 = losses_dev  # last val is best_ppl, don't want to plot it 
    y2 = losses_train

    x1 = list(range(1, len(y1) + 1))  # indx + 1
    x2 = list(range(1, len(y2) + 1))  # indx + 1
    

    plt.plot(x1, y1, label='Validation Loss')
    plt.plot(x2, y2, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses for each epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename1)
    plt.close()

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'PPL'])  
        for idx, value in enumerate(data):
            writer.writerow([idx + 1, value])

def save_results(ppls_dev, ppls_train, lr, epoch,  sampled_epochs, losses_dev, losses_train, drop, adam):
    test = "[LSTM_"
    if drop:
        test += "drop_"
    if adam:
        test += "adam_"
    test += str(lr) + "_" + str(epoch) + "]"
    plot_line_graph(ppls_dev, ppls_train, sampled_epochs, losses_dev, losses_train, os.path.join(dir, "ppls_" + test + ".png"), os.path.join(dir, "training_loss_" + test + ".png"))
    save_to_csv(ppls_dev, os.path.join(dir, "ppls_dev_" + test + ".csv"))
    save_to_csv(ppls_train, os.path.join(dir, "ppls_train_" + test + ".csv"))

    plot_line_graph(losses_dev, losses_train, sampled_epochs, losses_dev, losses_train, os.path.join(dir, "ppls_" + test + ".png"), os.path.join(dir, "training_loss_" + test + ".png"))
    save_to_csv(losses_dev, os.path.join(dir, "val_loss_" + test + ".csv"))
    save_to_csv(losses_train, os.path.join(dir, "train_loss_" + test + ".csv"))

    print("Experiment stopped at epoch: ", epoch, " with lr: ", lr, "[drop: ", drop, ", adam: ", adam, "]")


