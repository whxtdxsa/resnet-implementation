from models.mini_resnet import MiniResNet_v1 as MyModel 
from data.loader import get_kmnist_dataloaders 
from train.trainer import train_one_epoch, test_one_epoch, evaluate

import torch
import torch.nn as nn
import torch.optim as optim

import yaml
from utils.misc import set_seed, get_writer, log_tensorboard


def main():
    ### Import init_setting

    # Get hyperparams
    with open("config/default.yaml") as f: 
        cfg = yaml.safe_load(f)
        print(cfg)
        batch_size, epochs, lr, seed,  = cfg["batch_size"], cfg["epochs"], cfg["lr"], cfg["seed"]
    
    # Set Seed
    set_seed(seed)

    # Set logger
    model_name = "mini_resnet_v1"
    writer = get_writer(f"experiments/{model_name}_lr-{lr}_bs-{batch_size}_ep-{epochs}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    ### Define train procedure

    # Get loader
    train_loader, test_loader = get_kmnist_dataloaders(batch_size=batch_size) 
    
    # Set Model, Criterion, Optimizer
    network = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr)

    train_losses = []
    test_losses = []

    # Training
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        test_loss = test_one_epoch(network, test_loader, criterion, device)
        test_losses.append(test_loss)

        log_tensorboard(writer, epoch, train_loss, test_loss)
        print(f"Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}")

    writer.close()
    test_acc = evaluate(network, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")

main()


