import matplotlib.pyplot as plt
import os
import seaborn



class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.png')
    plt.clf()


def plot_attention(model, layers, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    for layer in range(layers):  # layers
        fig, (axs, axs1) = plt.subplots(1, 2, figsize=(10, 4))
        heatmap = seaborn.heatmap(model.transformer_encoder1.layers[layer].att[0].data.cpu(), ax=axs)
        heatmap.set_title("Local_attention", fontsize=10)
        heatmap = seaborn.heatmap(model.transformer_encoder2.layers[layer].att[0].data.cpu(), ax=axs1)
        heatmap.set_title("Global_attention", fontsize=10)
    heatmap.get_figure().savefig(f'plots/{folder}/attention-score.png')
    plt.clf()


def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def getresults2(df, result):  # all dims-sum & mean
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall', 'ROC/AUC']:
        results2[a] = df2[a]
    results2['f1_mean'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2
