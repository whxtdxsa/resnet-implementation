import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

model_dirs = {
    "mini_resnet_v1": {
        "train": "mini_resnet_v1_lr-0.05_bs-64_ep-10/Loss_train",
        "test": "mini_resnet_v1_lr-0.05_bs-64_ep-10/Loss_test"
    },
    "mini_resnet_v2": {
        "train": "mini_resnet_v2_lr-0.05_bs-64_ep-10/Loss_train",
        "test": "mini_resnet_v2_lr-0.05_bs-64_ep-10/Loss_test"
    },
    "mini_resnet_v3": {
        "train": "mini_resnet_v3_lr-0.05_bs-64_ep-10/Loss_train",
        "test": "mini_resnet_v3_lr-0.05_bs-64_ep-10/Loss_test"
    },
    "mini_resnet_v4": {
        "train": "mini_resnet_v4_lr-0.05_bs-64_ep-10/Loss_train",
        "test": "mini_resnet_v4_lr-0.05_bs-64_ep-10/Loss_test"
    },
}

# 색상 정의
color_map = {
    "mini_resnet_v1": ("#0072B2","123"),
    "mini_resnet_v2": ("#E69F00", "#ffbb78"),
    "mini_resnet_v3": ("#009E73", "#98df8a"),
    "mini_resnet_v4": ("#D55E00", "#ff9896"),
}

plt.figure(figsize=(10, 6))

for model_name, paths in model_dirs.items():
    # 각각의 로그 불러오기
    train_reader = SummaryReader(paths["train"])
    test_reader  = SummaryReader(paths["test"])

    train_df = train_reader.scalars
    test_df = test_reader.scalars

    # 색상 할당
    train_color, test_color = color_map[model_name]

    # 그래프 그리기
    plt.plot(train_df["step"], train_df["value"], label=f"{model_name} train", color=train_color, linewidth=1.0)
    plt.plot(test_df["step"], test_df["value"], label=f"{model_name} test", color=train_color, linewidth=3.0)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss (All Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_loss.png")
