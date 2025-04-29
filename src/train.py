from tqdm import tqdm
from data.kmnist import load_kmnist
from torch.utils.data import DataLoader, Subset

import torch
import torch.nn as nn
import torch.optim as optim

# from model.resnet import ResNet
from model.mini_resnet import MiniResNet as MyModel
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.05
epochs = 10
batch_size = 64

train_dataset, test_dataset = load_kmnist()
# train_dataset, test_dataset = Subset(train_dataset, range(30000)), Subset(test_dataset, range(1000))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

network = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    network.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for x_batch, t_batch in pbar:
        x_batch, t_batch = x_batch.to(device), t_batch.to(device)
        optimizer.zero_grad()
        y = network(x_batch) 
        loss = criterion(y, t_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss_list.append(running_loss / len(train_loader))

    train_acc = evaluate(network, train_loader, device)
    test_acc = evaluate(network, test_loader, device)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch {epoch}: train_loss = {running_loss/len(train_loader):.4f}, train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}")

from plot import plotting
path = "/data/data/com.termux/files/home/storage/dcim/Graph"
file_name = "result"
plotting(train_loss_list, train_acc_list, test_acc_list, file_name, path)
