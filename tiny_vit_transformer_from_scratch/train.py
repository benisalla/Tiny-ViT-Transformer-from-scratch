import torch
from torch.nn import nn
from tiny_vit_transformer_from_scratch.core.config import Config, VitConfig
from tiny_vit_transformer_from_scratch.data.vit_data_processor import DataPreprocessor
from tiny_vit_transformer_from_scratch.model.utils import live_plot_dual, lr_schedule, save_checkpoints, train_epoch, validate_epoch
from tiny_vit_transformer_from_scratch.model.vit_transformer import VisionTransformer
import torch.optim as optim


# build configs
vitconfig = VitConfig()
config = Config(batch_size=vitconfig.batch_size, image_size=vitconfig.image_size)

# hyperparameters
lr_rate = config.lr_rate
beta1, beta2 = config.beta1, config.beta2
eps = config.eps
w_decay = config.w_decay
amsgrad = config.amsgrad
num_epochs = config.num_epochs
min_lr = config.min_lr
warmup_iters = config.warmup_iters
total_iters = config.total_iters
device = config.device
label_smoothing = config.label_smoothing


# vit model
model = VisionTransformer(vitconfig)

# Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# Optimizer
adam_optimizer = optim.Adam(
    model.parameters(),
    lr=lr_rate,
    betas=(beta1, beta2),
    eps=eps,
    weight_decay=w_decay,
    amsgrad=amsgrad,
)

adamw_optimizer = optim.AdamW(
    model.parameters(),
    lr=lr_rate,             
    betas=(beta1, beta2),   
    eps=eps,                
    weight_decay=w_decay    
)

# Scheduler
lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(adamw_optimizer,lr_lambda=lambda epoch: lr_schedule(epoch, lr_rate, warmup_iters, total_iters, min_lr))
step_cheduler = torch.optim.lr_scheduler.StepLR(adamw_optimizer, step_size=2000, gamma=0.000001)
cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adamw_optimizer, T_max=num_epochs, eta_min=min_lr)
exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(adamw_optimizer, gamma=0.999)


# load data
data_preprocessor = DataPreprocessor(config)
train_loader, val_loader = data_preprocessor.create_dataloaders()


# start training
train_losses, train_accs = [], []
val_losses, val_accs = [], []

model.to(device)
optimizer = adamw_optimizer
scheduler = cos_scheduler
curr_lr = lr_rate

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
    live_plot_dual(data_dict, title=f'Taining & Validation Metrics [lr={curr_lr:.7f}]')
    
    if epoch % 100 == 0 and epoch > 0:
        save_checkpoints(model, optimizer, config.save_ckpt_path)