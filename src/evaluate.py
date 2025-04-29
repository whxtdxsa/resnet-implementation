import torch
def evaluate(network, data_loader, device):
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, t_batch in data_loader:
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            y = network(x_batch)
            y = y.argmax(dim=1)

            correct += (y == t_batch).sum().item()
            total += t_batch.size(0)
        acc = correct / total
    return acc
