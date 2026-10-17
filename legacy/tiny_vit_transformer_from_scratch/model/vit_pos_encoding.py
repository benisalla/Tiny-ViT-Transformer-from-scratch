import torch
import torch.nn as nn

class ViTPosEncoding(nn.Module):
    """
    Implements the positional encoding component of the Vision Transformer architecture.

    Positional encoding adds information about the relative or absolute position of the image
    patches in the sequence of input embeddings, which is crucial for the model to interpret
    image patterns correctly in the absence of recurrent or convolutional structures.

    Attributes:
        ppe (nn.Parameter): Learnable positional embeddings added to the input embeddings.
    
    Methods:
        forward (x): Applies positional encoding to the input tensor.
    """
    def __init__(self, n_embd, p_size, im_size):
        """
        Initializes the ViTPosEncoding module.

        Args:
            n_embd (int): Dimensionality of the embeddings (feature dimension).
            p_size (int): Size of each patch into which the images are divided.
            im_size (int): Size of the input images along one dimension (assumed square).

        Note:
            The positional embeddings are initialized as learnable parameters with a size
            calculated based on the number of patches ((im_size // p_size) ** 2). An extra
            position is added for the [CLS] token typically used in Vision Transformers.
        """
        super(ViTPosEncoding, self).__init__()

        # Calculate the total number of patches and initialize the positional embeddings.
        # +1 in the size calculation to accommodate the class token in transformers.
        self.ppe = nn.Parameter(torch.randn((im_size // p_size) ** 2 + 1, n_embd))

    def forward(self, x):
        """
        Adds positional embeddings to the input feature embeddings.

        Args:
            x (Tensor): The input embeddings tensor of shape (batch_size, num_patches + 1, embedding_dim),
                        where `num_patches + 1` accounts for the patches and an optional [CLS] token.

        Returns:
            Tensor: The embeddings tensor with added positional encodings.
        """
        x += self.ppe  # Element-wise addition of positional embeddings to the input embeddings
        return x
