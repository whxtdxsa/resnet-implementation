from models.mini_resnet import MiniResNet as MyModel 
from data.loader import get_kmnist_dataloaders 
from train.trainer import train_one_epoch, evaluate

import torch
import torch.nn as nn
import torch.optim as optim

import yaml
from utils.misc import set_seed, plot_learning_curve, get_writer, log_tensorboard


def main():
    writer = get_writer("runs/exp")
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    print(cfg)
    set_seed(cfg["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    train_loader, test_loader = get_kmnist_dataloaders(batch_size=cfg["batch_size"]) 
    
    network = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), cfg["lr"])

    train_losses = []
    test_accuracies = []
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{cfg["epochs"]}")
        train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        test_acc = evaluate(network, test_loader, device)
        test_accuracies.append(test_acc)
        log_tensorboard(writer, epoch, train_loss, test_acc)
    writer.close()
    print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")
    plot_learning_curve(train_losses, test_accuracies)

main()

#from plot import plotting
#path = "/data/data/com.termux/files/home/storage/dcim/Graph"
#file_name = "result"
#plotting(train_loss_list, train_acc_list, test_acc_list, file_name, path)
