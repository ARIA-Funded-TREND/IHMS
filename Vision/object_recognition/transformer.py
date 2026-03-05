"""
Vision Transformer Block Module
Contains the transformer block with MHSA, modulated attention, MLP, and layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple , List, Union
import math
from attention import ModulatedMultiHeadAttention, MultiHeadAttention

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for Vision Transformer
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension (default: 4 * in_features)
            out_features: Output feature dimension (default: in_features)
            activation: Activation function
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = activation()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class HybridTransformerBlock(nn.Module):
    """
    Transformer Block that can use either Modulated or Standard Attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attention_type: str = 'modulated',  # 'modulated' or 'standard'
        modulate_v: bool = False
    ):
        """
        Args:
            dim: Input embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias to qkv projection
            dropout: Dropout rate for MLP
            attn_dropout: Dropout rate for attention
            drop_path: Stochastic depth rate
            activation: Activation function for MLP
            norm_layer: Normalization layer
            attention_type: Type of attention ('modulated' or 'standard')
            modulate_v: Whether to modulate V in modulated attention
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Select attention mechanism
        if attention_type == 'modulated':
            self.attn = ModulatedMultiHeadAttention(
                dim=dim,
                num_latents=1,
                num_heads=1, #fixing num heads to 1 for modulated attention
                qkv_bias=qkv_bias,
                dropout=attn_dropout,
                modulate_v=modulate_v
            )
        elif attention_type == 'standard':
            self.attn = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout=attn_dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Use 'modulated' or 'standard'")
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout=dropout
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        # Attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = self.drop_path(self.attn(self.norm1(x)))
        
        # MLP block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        x_norm = self.norm1(x)
        return self.attn.get_attention_weights(x_norm) if hasattr(self.attn, 'get_attention_weights') else None
    
    def extra_repr(self):
        return f'attention_type={self.attention_type}, dim={self.dim}, num_heads={self.num_heads}'

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for regularization.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob: Probability of dropping a path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'