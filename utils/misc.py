import random
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_writer(log_dir="experiments"):
    return SummaryWriter(log_dir=log_dir)


def log_tensorboard(writer, epoch, train_loss, test_loss):
    writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)


def log_metrics(epoch, train_loss, test_acc, log_path="log.txt"):
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Acc = {test_acc:. 4f}\n")


from contextlib import nullcontext

def get_amp_components(device):
    """
    device (torch.device): 'cuda' or 'cpu'
    return:
        amp_context: autocast context or nullcontext
        scaler: GradScaler or None
    """
    if device.type == 'cuda':
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
    else:
        amp_context = nullcontext()
        scaler = None
    return amp_context, scaler
