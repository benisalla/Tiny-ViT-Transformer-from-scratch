import torch


class VitConfig:
    def __init__(self):
        self.batch_size = 16
        self.embedding_dim = 512  # TODO: Adjust 
        self.num_blocks = 12      # TODO: Adjust 
        self.hidden_size = 32     # TODO: Adjust 
        self.patch_size = 16
        self.channel_dim = 3
        self.img_size = 256     # TODO: Adjust
        self.num_classes = 10
        self.dropout_rate = 0.0   # Use 0.1 in fine-tuning
        self.use_bias = False
        self.head_dim = 6 * self.embedding_dim  

    def __repr__(self):
        return  f"<Config batch_size={self.batch_size}, embedding_dim={self.embedding_dim}, num_blocks={self.num_blocks}, " \
                f"hidden_size={self.hidden_size}, patch_size={self.patch_size}, channel_dim={self.channel_dim}, " \
                f"img_size={self.img_size}, num_classes={self.num_classes}, dropout_rate={self.dropout_rate}, " \
                f"use_bias={self.use_bias}, head_dim={self.head_dim}>"


class Config:
    def __init__(self, batch_size, img_size) -> None:
        self.save_ckpt_path = "./vit_chpts.pth"
        self.load_chpt_path = "./vit_chpts.pth"
        self.lr_rate = 2e-4  # 3e-3 could be added as a comment or in documentation
        self.w_decay = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.amsgrad = False  
        self.train_size = 2000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.total_iters = 3000000
        self.batch_size = batch_size
        self.num_epochs = (self.total_iters * self.batch_size) // self.train_size
        self.warmup_epochs = 300
        self.warmup_iters = self.warmup_epochs * (self.train_size // self.batch_size)
        self.min_lr = 1e-6
        self.label_smoothing = 0.1
        
        self.classes = [
            "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
            "Tomato Septoria leaf spot", "Tomato Spider mites Two spotted spider mite", "Tomato Target Spot",
            "Tomato healthy", "Potato Early blight", "Potato Late blight", "Tomato Tomato mosaic virus", "Potato healthy"
        ]

        self.valid_size = 0.2
        self.img_size = img_size
        self.num_workers = 1
        self.pin_memory = True
        self.shuffle = True
        self.data_dir = "./morocco-pests-and-diseases-dataset"
        self.max_img_cls = 250              # to None after sanity check
        self.max_cls = 5                    # to None after sanity check
        self.is_balanced = False
        
    def __repr__(self):
        return (f"<Config save_ckpt_path={self.save_ckpt_path}, load_chpt_path={self.load_chpt_path}, "
                f"lr_rate={self.lr_rate}, w_decay={self.w_decay}, beta1={self.beta1}, beta2={self.beta2}, "
                f"eps={self.eps}, amsgrad={self.amsgrad}, total_iters={self.total_iters}, "
                f"num_epochs={self.num_epochs}, warmup_epochs={self.warmup_epochs}, device={self.device}, "
                f"warmup_iters={self.warmup_iters}, min_lr={self.min_lr}, label_smoothing={self.label_smoothing}>")
