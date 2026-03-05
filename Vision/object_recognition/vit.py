import torch
import torch.nn as nn
from typing import Optional, Union, List
from transformer import HybridTransformerBlock


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding
    Converts images into patches and linearly embeds them
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True
    ):
        """
        Args:
            image_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
            bias: Whether to use bias in projection layer
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        # Patch embedding using convolution
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

        # Layer normalization (optional, but often helpful)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape

        # Verify input dimensions
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}, {W}) doesn't match model ({self.image_size}, {self.image_size})"
        assert C == self.in_channels, \
            f"Input channels ({C}) doesn't match model ({self.in_channels})"

        # Extract patches and flatten
        x = self.proj(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Apply layer normalization
        x = self.norm(x)

        return x

    def get_patch_info(self):
        """Return patch embedding information"""
        return {
            'num_patches': self.num_patches,
            'grid_size': self.grid_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        }


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for Vision Transformer
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.0):
        """
        Args:
            num_patches: Number of patches in the image
            embed_dim: Embedding dimension
            dropout: Dropout rate for positional embeddings
        """
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Learnable positional embeddings (including CLS token)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Patch embeddings of shape (B, num_patches + 1, embed_dim)
        Returns:
            Embeddings with positional encoding of same shape
        """
        return self.dropout(x + self.pos_embed)


class ClassificationHead(nn.Module):
    """
    Classification head for Vision Transformer
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        use_norm: bool = True
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            dropout: Dropout rate before final linear layer
            use_norm: Whether to apply layer norm before classification
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize with smaller weights for better training stability
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: CLS token features of shape (B, embed_dim)
        Returns:
            Class logits of shape (B, num_classes)
        """
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


class HybridVisionTransformer(nn.Module):
    """
    Vision Transformer with Hybrid Attention Architecture
    Supports mixing modulated and standard attention layers
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        attention_pattern: Union[str, List[str]] = 'modulated',
        modulate_v: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.GELU
    ):
        """
        Args:
            image_size: Input image size
            patch_size: Patch size for patch embedding
            in_channels: Number of input channels
            num_classes: Number of classification classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to add bias to qkv projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            attention_pattern: Pattern of attention types. Options:
                - 'modulated': All layers use modulated attention
                - 'standard': All layers use standard attention
                - 'alternating': Alternates between modulated and standard
                - 'early_modulated': First half modulated, second half standard
                - 'late_modulated': First half standard, second half modulated
                - List[str]: Explicit list of attention types per layer
            modulate_v: Whether to modulate V in modulated attention layers
            norm_layer: Normalization layer
            activation: Activation function
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Determine attention pattern for each layer
        attention_types = self._parse_attention_pattern(attention_pattern, depth)
        self.attention_types = attention_types
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Build transformer blocks with hybrid attention
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
                activation=activation,
                norm_layer=norm_layer,
                attention_type=attention_types[i],
                modulate_v=modulate_v
            )
            for i in range(depth)
        ])
        
        # Classification head
        self.head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _parse_attention_pattern(self, pattern: Union[str, List[str]], depth: int) -> List[str]:
        """
        Parse attention pattern into a list of attention types per layer
        
        Args:
            pattern: Attention pattern specification
            depth: Number of layers
        
        Returns:
            List of attention types for each layer
        """
        if isinstance(pattern, list):
            assert len(pattern) == depth, f"Pattern list length {len(pattern)} must match depth {depth}"
            return pattern
        
        if pattern == 'modulated':
            return ['modulated'] * depth
        elif pattern == 'standard':
            return ['standard'] * depth
        elif pattern == 'alternating':
            return ['modulated' if i % 2 == 0 else 'standard' for i in range(depth)]
        elif pattern == 'early_modulated':
            mid = depth // 2
            return ['modulated'] * mid + ['standard'] * (depth - mid)
        elif pattern == 'late_modulated':
            mid = depth // 2
            return ['standard'] * mid + ['modulated'] * (depth - mid)
        else:
            raise ValueError(f"Unknown attention pattern: {pattern}")
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x):
        """
        Forward pass through patch embedding, positional encoding, and transformer blocks
        
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Feature tokens of shape (B, num_patches + 1, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return x
    
    def forward(self, x):
        """
        Full forward pass
        
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Class logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits
    
    def get_attention_maps(self, x, block_idx: Optional[int] = None):
        """
        Extract attention maps for visualization
        
        Args:
            x: Input images
            block_idx: Which block to extract attention from (None for all)
        Returns:
            Attention maps with layer information
        """
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_embed(x)
        
        attention_maps = []
        
        for i, block in enumerate(self.blocks):
            if block_idx is None or i == block_idx:
                attn_info = block.get_attention_weights(x)
                if attn_info is not None:
                    attention_maps.append({
                        'layer': i,
                        'attention_type': self.attention_types[i],
                        'weights': attn_info[0],
                        'modulation_info': attn_info[1] if len(attn_info) > 1 else None
                    })
            x = block(x)
        
        return attention_maps if len(attention_maps) > 1 else attention_maps[0] if attention_maps else None
    
    def get_model_info(self):
        """Get comprehensive model information including attention pattern"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count attention types
        modulated_count = sum(1 for t in self.attention_types if t == 'modulated')
        standard_count = sum(1 for t in self.attention_types if t == 'standard')
        
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_patches': self.patch_embed.num_patches,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_classes': self.num_classes,
            'attention_pattern': self.attention_types,
            'modulated_layers': modulated_count,
            'standard_layers': standard_count,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2
        }
    
    def print_architecture(self):
        """Print a visual representation of the architecture"""
        print("=" * 70)
        print(f"Hybrid Vision Transformer Architecture")
        print("=" * 70)
        info = self.get_model_info()
        print(f"Image Size: {info['image_size']}x{info['image_size']}")
        print(f"Patches: {info['num_patches']} ({self.patch_embed.grid_size}x{self.patch_embed.grid_size})")
        print(f"Embedding Dim: {info['embed_dim']}")
        print(f"Total Layers: {info['depth']}")
        print(f"  - Modulated Attention: {info['modulated_layers']}")
        print(f"  - Standard Attention: {info['standard_layers']}")
        print(f"Parameters: {info['total_params']:,} ({info['model_size_mb']:.2f} MB)")
        print("\nLayer-by-Layer Attention Types:")
        print("-" * 70)
        for i, attn_type in enumerate(self.attention_types):
            symbol = "🔷" if attn_type == "modulated" else "⬜"
            print(f"  Layer {i:2d}: {symbol} {attn_type.capitalize()}")
        print("=" * 70)


def create_hybrid_vit(
    model_size: str = 'tiny',
    num_classes: int = 10,
    image_size: int = 32,
    modulate_v: bool = True,
    dropout: float = None,
    drop_path_rate: float = None
):
    """
    Factory function to create Hybrid Vision Transformer models
    
    Args:
        model_size: Size of the model ('tiny', 'small', 'base', 'large')
        num_classes: Number of classes for classification
        image_size: Input image size
        attention_pattern: Pattern of attention types
        modulate_v: Whether to modulate V in modulated layers
        dropout: Dropout rate (overrides model size default if provided)
        drop_path_rate: Stochastic depth rate (overrides model size default if provided)
    
    Returns:
        Hybrid Vision Transformer model
    """
    configs = {
        'hybrid_base': {
            'embed_dim': 768,
            'depth': 6,
            'num_heads': 12,
            'patch_size': 16,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'drop_path_rate': 0.1,
            'attention_pattern': ['modulated', 'modulated' ,'standard', 'modulated','modulated', 'standard']
        },
        'co4_base': {
            'embed_dim': 768,
            'depth': 6,
            'num_heads': 1,
            'patch_size': 16,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'drop_path_rate': 0.1,
            'attention_pattern': ['modulated', 'modulated' ,'modulated', 'modulated','modulated', 'modulated']
        }

    }
    
    assert model_size in configs, f"Model size {model_size} not supported"
    config = configs[model_size]
    
    # Use provided values or fall back to config defaults
    final_dropout = dropout if dropout is not None else config['dropout']
    final_drop_path_rate = drop_path_rate if drop_path_rate is not None else config['drop_path_rate']
    
    return HybridVisionTransformer(
        image_size=image_size,
        patch_size=config['patch_size'],
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=final_dropout,
        drop_path_rate=final_drop_path_rate,
        attention_pattern=config['attention_pattern'],
        modulate_v=modulate_v
    )