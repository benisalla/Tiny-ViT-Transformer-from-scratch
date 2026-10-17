import torch
import torch.nn as nn
from core.config import Config, VitConfig
from data.vit_data_processor import DataPreprocessor
from model.utils import live_plot_dual, lr_schedule, save_checkpoints, train_epoch, validate_epoch
from model.vit_transformer import VisionTransformer
import torch.optim as optim

def main():
    # build configs
    vitconfig = VitConfig()
    config = Config(batch_size=vitconfig.batch_size, im_size=vitconfig.im_size, n_class=vitconfig.n_class)

    # hyperparameters
    lr_rate = config.lr_rate
    beta1, beta2 = config.beta1, config.beta2
    eps = config.eps
    w_decay = config.w_decay
    # amsgrad = config.amsgrad
    num_epochs = config.num_epochs
    min_lr = config.min_lr
    # warmup_iters = config.warmup_iters
    # total_iters = config.total_iters
    device = config.device
    label_smoothing = config.label_smoothing

    # vit model
    model = VisionTransformer(vitconfig)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_rate,             
        betas=(beta1, beta2),   
        eps=eps,                
        weight_decay=w_decay    
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # load data
    data_preprocessor = DataPreprocessor(config)
    train_loader, val_loader, _ = data_preprocessor.create_dataloaders()
    

    # start training
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    model.to(device)

    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        # Update plots with the new function
        data_dict = {
            'Train Loss': train_losses,
            'Val Loss': val_losses,
            'Train Acc': train_accs,
            'Val Acc': val_accs
        }
        live_plot_dual(data_dict, title=f'Training & Validation Metrics [lr={curr_lr:.7f}]')

        if epoch % 100 == 0 and epoch > 0:
            save_checkpoints(model, optimizer, config.save_ckpt_path)

if __name__ == '__main__':
    main()
