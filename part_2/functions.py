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
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))

    return ppl, sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

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

def create_next_test_folder(base_dir):
    global n, dir  # Declare as global to update the global variables
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

    y1 = data1[:-1]  # l'ultimo valore è il best_ppl, non voglio plottarlo 
    y2 = data2[:-1]  
    
    x1 = list(range(1, len(y1) + 1))  # Indici incrementati di 1
    x2 = list(range(1, len(y2) + 1))  # Indici incrementati di 1
    

    plt.plot(x1, y1, label='PPL valuation')
    plt.plot(x2, y2, label='PPL training')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.title('PPLs for each epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

    y1 = losses_dev  # l'ultimo valore è il best_ppl, non voglio plottarlo 
    y2 = losses_train

    x1 = list(range(1, len(y1) + 1))  # Indici incrementati di 1
    x2 = list(range(1, len(y2) + 1))  # Indici incrementati di 1
    

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
        writer.writerow(['Index', 'PPL'])  # Scrivi l'intestazione delle colonne
        for idx, value in enumerate(data):
            writer.writerow([idx + 1, value])

def save_results(ppls_dev, ppls_train, lr_initial, lr, epoch,  sampled_epochs, losses_dev, losses_train):
    test = "[LSTM_"
    # if DROP:
    #     test += "drop_"
    # if config.adam:
    #     test += "adam_"
    test += str(lr_initial) + "_" + str(epoch) + "]"
    plot_line_graph(ppls_dev, ppls_train, sampled_epochs, losses_dev, losses_train, os.path.join(dir, "ppls_" + test + ".png"), os.path.join(dir, "training_loss_" + test + ".png"))
    save_to_csv(ppls_dev, os.path.join(dir, "ppls_dev_" + test + ".csv"))
    save_to_csv(ppls_train, os.path.join(dir, "ppls_train_" + test + ".csv"))

    plot_line_graph(losses_dev, losses_train, sampled_epochs, losses_dev, losses_train, os.path.join(dir, "ppls_" + test + ".png"), os.path.join(dir, "training_loss_" + test + ".png"))
    save_to_csv(losses_dev, os.path.join(dir, "val_loss_" + test + ".csv"))
    save_to_csv(losses_train, os.path.join(dir, "train_loss_" + test + ".csv"))

    print("Experiment stopped at epoch: ", epoch, " with lr: ", lr, " and initial lr: ", lr_initial)


