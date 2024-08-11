# References: https://github.com/lucidrains/vit-pytorch

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    """
    Convert a single value to a pair.
    If t is already a pair, return it as is.
    """
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    """
    Implements a feed-forward network with GELU activation and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Implements multi-head self-attention mechanism.

    Args:
        dim (int): The input dimension
            Note: The dim shared between the transformer blocks and between attention and feedforward sub-blocks
            The d_model in transformer terminology would be inner_dim = dim_head * heads
            (dim => inner_dim => ... => inner_dim => dim)
            When dim == inner_dim, then standard transformer
        heads (int): Number of attention heads (default 8)
        dim_head (int): Dimension of each attention head (default 64)
        dropout (float): Dropout probability
        vis (bool): Whether to visualize attention weights (default False)

    Shape:
        - Input: (batch_size, seq_len, dim)
        - Output: (batch_size, seq_len, dim), attention weights (optional)
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., vis = False):
        super().__init__()
        inner_dim = dim_head * heads # d_model of transformer
        project_out = not (heads == 1 and dim_head == dim)

        self.vis = vis
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Single linear layer to compute query, key, and value
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim)
            torch.Tensor: Attention weights if vis is True, else None
        """
        x = self.norm(x) # pre_norm with shape: b n input_dim

        # Project input into query, key, and value representations with different transformation
        # Rearrange q, k, v to separate the heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 3 b n (h d)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # d_model=hxd=headxhead_dim; n=hxw=num_patches_heightxnum_patches_width

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h n d x b h d n -> b h n n

        # Apply softmax and masking to get attention weights
        attn = self.attend(dots)
        weights = attn if self.vis else None
        attn = self.dropout(attn)

        # Apply attention weights to values and concatenate heads
        out = torch.matmul(attn, v) # b h n n x b h n d -> b h n d
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), weights

class Transformer(nn.Module):
    """
    Implements a transformer block with multiple layers of attention and feed-forward networks.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., vis = False):
        super().__init__()
        self.vis = vis
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, vis = vis),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        attn_weights = []
        for attn, ff in self.layers:
            attn_out, weights = attn(x)
            x = attn_out + x  # Attention with residual connection
            x = ff(x) + x    # Feed-forward with residual connection
            if self.vis:
                attn_weights.append(weights)

        return self.norm(x), attn_weights

class ViT(nn.Module):
    """
    Implementation of the Vision Transformer (ViT) model.

    Args:
        image_size (int, tuple): Image size. If you have rectangular images, make sure your image size is the maximum of the width and height.
        patch_size (int, tuple): Size of patches. `image_size` must be divisible by `patch_size`.
                          The number of patches is: `n = (image_size // patch_size) ** 2` and `n` must be greater than 16.
        num_classes (int): Number of classes to classify.
        dim (int): The dimensionality that is shared across the transformer blocks and shared between attention and linear sub-blocks.
            Within each attention sub-block, the dimension dim is first transformed into dim_head * heads for processing,
            and then converted back to dim after the attention operations are completed.
            If dim equals dim_head * heads, the implementation behaves as a standard transformer.
        depth (int): Number of Transformer blocks.
        heads (int): Number of heads in Multi-head Attention layer. (d_model = dim_head * heads)
        mlp_dim (int): Dimension of the MLP (FeedForward) layer.
        pool (str): Either 'cls' token pooling or 'mean' pooling, used for projection head processing the transformer output.
        channels (int, optional): Number of image's channels. Default is 3.
        dim_head (int, optional): Dimension of each attention head. Default is 64. (d_model = dim_head * heads)
        dropout (float, optional): Dropout rate. Default is 0.
        emb_dropout (float, optional): Embedding dropout rate. Default is 0.
        vis (bool, optional): Whether to visualize attention weights. Default is False.
            When `vis` is True, the forward pass returns a list of attention weight tensors, with a length equal to `num_blocks`, for visualization;
            each tensor in the list has the shape [batch_size, num_heads, seq_len, seq_len].
            If `vis` is False, the forward function returns None for attention weights.

    Attributes:
        to_patch_embedding (nn.Sequential): Converts image to patch embeddings.
        pos_embedding (nn.Parameter): Positional embedding for patches.
        cls_token (nn.Parameter): Learnable classification token.
        dropout (nn.Dropout): Dropout layer.
        transformer (Transformer): Transformer encoder.
        pool (str): Pooling type ('cls' or 'mean').
        to_latent (nn.Identity): Identity layer for latent representation.
        mlp_head (nn.Linear): Final classification head.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., vis = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.vis = vis
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # Flatten
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, vis)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        Forward pass of the Vision Transformer.

        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
            list or None: A list of attention weight tensors if `vis` is True; otherwise, None.
                The list has a length equal to `num_blocks`, with each element having shape
                [batch_size, num_heads, seq_len, seq_len].
        """
        x = self.to_patch_embedding(img) # Flatten 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] # Learnable
        x = self.dropout(x)

        x, attn_weights = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), attn_weights if self.vis else (self.mlp_head(x), None)
