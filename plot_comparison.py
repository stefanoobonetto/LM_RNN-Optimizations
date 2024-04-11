import matplotlib.pyplot as plt
import pandas as pd

dir = "comparison_train_loss"
type = "Training Loss"
df = [None, None, None]

def plot_comparison_PPL(folder, type):
    for i in range(3):
        df[i] = pd.read_csv(f"{folder}/{i+1}.csv")

    plt.plot(df[0]["PPL"][:-1], label="LSTM", color="blue")
    plt.plot(df[1]["PPL"][:-1], label="+ dropout", color="red")
    plt.plot(df[2]["PPL"][:-1], label="+ AdamW", color="green")
    plt.grid(True)
    plt.xlabel("Epoch", weight='bold')
    plt.ylabel(type, weight='bold')
    plt.legend(fontsize='large')
    plt.xticks(range(0, max(len(df[0]), len(df[1]), len(df[2]))-1, 1))
    if type == "PPL":
        plt.text(36, 220, "LSTM best PPL: " + str(round(df[0]["PPL"][len(df[0])-1], 2)), fontsize=12, color="blue", weight='bold')
        plt.text(36, 200, "+ dropout best PPL: " + str(round(df[1]["PPL"][len(df[1])-1], 2)), fontsize=12, color="red", weight='bold')
        plt.text(36, 180, "+ AdamW best PPL: " + str(round(df[2]["PPL"][len(df[2])-1], 2)), fontsize=12, color="green", weight='bold')
    plt.show()

plot_comparison_PPL(dir, type)

