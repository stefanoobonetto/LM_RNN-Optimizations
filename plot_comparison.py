import matplotlib.pyplot as plt
import pandas as pd

dir = "comparison_PPL"
type = "PPL"
df = [None, None, None, None, None]

def plot_comparison_PPL(folder, type):
    for i in range(5):
        df[i] = pd.read_csv(f"{folder}/{i+1}.csv")

    plt.plot(df[0]["PPL"][:-1], label="LSTM", color="blue")
    plt.plot(df[1]["PPL"][:-1], label="LSTM + dropout", color="red")
    plt.plot(df[2]["PPL"][:-1], label="LSTM + dropout + AdamW", color="green")
    plt.plot(df[3]["PPL"][:-1], label="LSTM + weight_tying", color="orange")
    plt.plot(df[4]["PPL"][:-1], label="LSTM + weight_tying + VarDropout", color="brown")

    plt.grid(True)
    plt.xlabel("Epoch", weight='bold')
    plt.ylabel(type, weight='bold')
    plt.legend(fontsize='large')
    max_tick = max(len(df[0]), len(df[1]), len(df[2]), len(df[3]), len(df[4])) - 1
    plt.xticks(range(0, max_tick, 5))

    if type == "PPL":
        plt.text(57, 260, "1.1 best PPL: " + str(round(df[0]["PPL"][len(df[0])-1], 2)), fontsize=12, color="blue", weight='bold')
        plt.text(57, 240, "1.2 best PPL: " + str(round(df[1]["PPL"][len(df[1])-1], 2)), fontsize=12, color="red", weight='bold')
        plt.text(57, 220, "1.3 best PPL: " + str(round(df[2]["PPL"][len(df[2])-1], 2)), fontsize=12, color="green", weight='bold')
        plt.text(57, 200, "1.4 best PPL: " + str(round(df[3]["PPL"][len(df[3])-1], 2)), fontsize=12, color="orange", weight='bold')
        plt.text(57, 180, "1.5 best PPL: " + str(round(df[4]["PPL"][len(df[4])-1], 2)), fontsize=12, color="brown", weight='bold')
    plt.show()

plot_comparison_PPL(dir, type)

