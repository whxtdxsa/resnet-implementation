import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
folder_name = "experiments"
file_name = "metrics.csv"
model_dirs = {
    "mini_resnet_v1": f"{folder_name}/MiniResNet_v1_bs64_ep10_lr0.05/{file_name}",
    "mini_resnet_v2": f"{folder_name}/MiniResNet_v2_bs64_ep10_lr0.05/{file_name}",
    "mini_resnet_v3": f"{folder_name}/MiniResNet_v3_bs64_ep10_lr0.05/{file_name}",
    "mini_resnet_v4": f"{folder_name}/MiniResNet_v4_bs64_ep10_lr0.05/{file_name}",
    "mini_resnet_v5": f"{folder_name}/MiniResNet_v5_bs64_ep10_lr0.05/{file_name}",
    "mini_resnet_v6": f"{folder_name}/MiniResNet_v6_bs64_ep10_lr0.05/{file_name}"
}

color_map = {
    "mini_resnet_v1": "#0072B2",
    "mini_resnet_v2": "#E69F00",
    "mini_resnet_v3": "#009E73",
    "mini_resnet_v4": "#D55E00",
    "mini_resnet_v5": "#CC79A7",
    "mini_resnet_v6": "#999999"
}

plt.figure(figsize=(10, 6))

for model_name, path in model_dirs.items():

    df = pd.read_csv(path)
    color = color_map[model_name]

    plt.plot(df["epoch"], df["train_loss"], label=f"{model_name}_train", color=color, linewidth=1.0)
    plt.plot(df["epoch"], df["test_loss"], label=f"{model_name}_test", color=color, linewidth=3.0)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss (All Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_loss.png")
