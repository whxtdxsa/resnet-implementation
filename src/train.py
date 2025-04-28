from tqdm import tqdm
from data.kmnist import load_kmnist
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

from model import CustomCNN
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.1
epochs = 10
batch_size = 256

train_dataset, test_dataset = load_kmnist()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

network = CustomCNN().to(device)
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
        with autocast():
            y = network(x_batch) 
            loss = criterion(y, t_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    train_loss_list.append(running_loss / len(train_loader))

    train_acc = evaluate(network, train_loader, device)
    test_acc = evaluate(network, test_loader, device)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch {epoch}: train_loss = {running_loss/len(train_loader):.4f}, train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}")


import numpy as np
import matplotlib.pyplot as plt
path = "/data/data/com.termux/files/home/storage/dcim/Graph"
# path = "."
x1 = np.arange(len(train_loss_list))
x2 = np.arange(len(train_acc_list))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)  
plt.plot(x1, train_loss_list, label='train loss')
plt.xlabel("iters")
plt.ylabel("loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2) 
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()  
plt.savefig(path + '/CustomCNN_cnn.png')
plt.clf()

