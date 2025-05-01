import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_metrics(epoch, train_loss, test_acc, log_path="log.txt"):
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Acc = {test_acc:. 4f}\n")

    
import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, test_accuracies, save_path="learning_curve.png"):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color='tab:blue')
    ax2.plot(epochs, test_accuracies, label='Test Accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title('Training Loss and Test Accuracy')

    plt.savefig(save_path)
    print(f"그래프 저장됨: {save_path}")

# utils/misc.py

from torch.utils.tensorboard import SummaryWriter

def get_writer(log_dir="runs/exp1"):
    return SummaryWriter(log_dir=log_dir)

def log_tensorboard(writer, epoch, train_loss, test_acc):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)
