from data.loader import get_kmnist_dataloaders 
from train.trainer import train_one_epoch, evaluate_loss, evaluate_accuracy

import torch
import torch.nn as nn
import torch.optim as optim

import yaml
import importlib
from utils.misc import set_seed, init_csv_log, log_to_csv, get_amp_components


def main():
    ### Import init_setting

    # Get hyperparams
    with open("config/default.yaml") as f: 
        cfg = yaml.safe_load(f)
        print(cfg)
        model_name, class_name, batch_size, epochs, lr, seed = cfg["model_name"], cfg["class_name"], cfg["batch_size"], cfg["epochs"], cfg["lr"], cfg["seed"]
    
    # Set Seed
    set_seed(seed)

    # Set Model
    module = importlib.import_module(f"models.{model_name}")
    MyModel = getattr(module, class_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_context, scaler = get_amp_components(device)


    ### Define train procedure

    # Get data loader
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
        train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device, amp_context, scaler)
        train_losses.append(train_loss)

        test_loss = evaluate_loss(network, test_loader, criterion, device, amp_context)
        test_losses.append(test_loss)

        print(f"Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}")

    test_acc = evaluate_accuracy(network, test_loader, device, amp_context)
    print(f"Test Acc: {test_acc:.4f}")
    experiment_name = f"{class_name}_bs{batch_size}_ep{epochs}_lr{lr}"
    log_path = f"experiments/{experiment_name}/metrics.csv"
    init_csv_log(log_path, ["epoch", "train_loss", "test_loss"])
    # Logging
    for epoch in range(len(train_losses)):
        log_to_csv(log_path, {
            "epoch": epoch,
            "train_loss": train_losses[epoch],
            "test_loss": test_losses[epoch]
        })

if __name__ == "__main__":
    main()
