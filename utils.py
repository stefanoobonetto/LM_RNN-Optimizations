import torch
import config
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import copy
import csv
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # it can be changed with 'cpu' if you do not have a gpu
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' # it can be changed with 'cpu' if you do not have a gpu

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# This class computes and stores our vocab
# Word to ids and ids to word
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]): 
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

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

def plot_line_graph(data, sampled_epochs, losses_dev, losses_train, filename, filename1):
    x = list(range(1, len(data) + 1))  # Indici incrementati di 1
    y = data  # Valori della lista

    plt.plot(x, y)
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.title('PPLs for each epoch')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    plt.plot(sampled_epochs, losses_train, label='Training Loss')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename1)
    plt.close()

    

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'PPL'])  # Scrivi l'intestazione delle colonne
        for idx, value in enumerate(data):
            writer.writerow([idx + 1, value])

def save_results(ppls, lr_initial, lr, epoch,  sampled_epochs, losses_dev, losses_train):
    test = "[LSTM_"
    if config.drop:
        test += "drop_"
    if config.adam:
        test += "adam_"
    test += str(lr_initial) + "_" + str(lr) + "_" + str(epoch) + "]"
    plot_line_graph(ppls, sampled_epochs, losses_dev, losses_train, os.path.join(dir, "ppls_" + test + ".png"), os.path.join(dir, "training_loss_" + test + ".png"))
    save_to_csv(ppls, os.path.join(dir, "ppls_" + test + ".csv"))
    save_to_csv(losses_dev, os.path.join(dir, "val_loss_" + test + ".csv"))
    save_to_csv(losses_train, os.path.join(dir, "train_loss_" + test + ".csv"))

    print("Experiment stopped at epoch: ", epoch, " with lr: ", lr, " and initial lr: ", lr_initial, "[drop: ", config.drop, ", adam: ", config.adam, "]")


