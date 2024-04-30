class Config:
    def __init__(self):
        self.batch_size = 16
        self.embedding_dim = 512  # TODO: Adjust as necessary
        self.num_blocks = 12      # TODO: Adjust as necessary
        self.hidden_size = 32     # TODO: Adjust as necessary
        self.patch_size = 16
        self.channel_dim = 3
        self.image_size = 256     # TODO: Adjust as necessary
        self.num_classes = 10
        self.dropout_rate = 0.0   # Use 0.1 in fine-tuning
        self.use_bias = False
        self.head_dim = 6 * self.embedding_dim  # Calculated based on embedding dimension

    def __repr__(self):
        return  f"<Config batch_size={self.batch_size}, embedding_dim={self.embedding_dim}, num_blocks={self.num_blocks}, " \
                f"hidden_size={self.hidden_size}, patch_size={self.patch_size}, channel_dim={self.channel_dim}, " \
                f"image_size={self.image_size}, num_classes={self.num_classes}, dropout_rate={self.dropout_rate}, " \
                f"use_bias={self.use_bias}, head_dim={self.head_dim}>"

# Create an instance of the config
config = Config()