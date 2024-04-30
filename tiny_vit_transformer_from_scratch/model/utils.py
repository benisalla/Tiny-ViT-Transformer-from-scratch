import torch
from torch.nn import nn
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import math


def lr_schedule(epoch, learning_rate, warmup_iters, total_iters, min_lr):
    if epoch < warmup_iters:
        return min_lr + (learning_rate - min_lr) * epoch / warmup_iters
    elif epoch > total_iters:
        return min_lr
    else:
        decay_ratio = (epoch - warmup_iters) / (total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    
def save_checkpoints(model, optimizer, save_ckpt_path, epoch):
    model_args = model.get_init_args() 
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'epoch': epoch,
    }
    torch.save(checkpoint, save_ckpt_path)
    print("\033[94mCheckpoints Saved Successfully :)\033[0m")

    
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


def live_plot_dual(data_dict, figsize=(12,5), title=''):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(data_dict['Train Loss'], 'r-', label='Train Loss')
    ax1.plot(data_dict['Val Loss'], 'b-', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    if data_dict['Train Loss']:
        ax1.annotate(f"{data_dict['Train Loss'][-1]:.4f}", xy=(1, data_dict['Train Loss'][-1]), xytext=(8,0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
    if data_dict['Val Loss']:
        ax1.annotate(f"{data_dict['Val Loss'][-1]:.4f}", xy=(1, data_dict['Val Loss'][-1]), xytext=(8,0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    ax2.plot(data_dict['Train Acc'], 'r-', label='Train Accuracy')
    ax2.plot(data_dict['Val Acc'], 'b-', label='Val Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    if data_dict['Train Acc']:
        ax2.annotate(f"{data_dict['Train Acc'][-1]:.4f}", xy=(1, data_dict['Train Acc'][-1]), xytext=(8,0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
    if data_dict['Val Acc']:
        ax2.annotate(f"{data_dict['Val Acc'][-1]:.4f}", xy=(1, data_dict['Val Acc'][-1]), xytext=(8,0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    
    plt.suptitle(title)
    plt.show()