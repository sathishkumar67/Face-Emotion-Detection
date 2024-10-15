from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    image_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    patch_size: int
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    head_dim: int = None 
    num_image_tokens: int = None 

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_image_tokens = (self.image_size // self.patch_size) ** 2


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class EncoderEmbeddings(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Convolve the image into patches of size `patch_size`
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            padding="valid", # This indicates no padding is added
        )

        # Positional embedding for image tokens
        self.position_embedding = nn.Embedding(self.config.num_image_tokens, self.config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.config.num_image_tokens).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Num_Channels, Image_Height, Image_Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.patch_embedding(pixel_values).flatten(2).transpose(1, 2) + self.position_embedding(self.position_ids)


class EncoderAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0, f"`hidden_size`: {config.hidden_size} is not divisible by `num_attention_heads`: {config.num_attention_heads}."

        # key, query, value projection for all heads, but in a batch
        self.attention = nn.Linear(config.hidden_size, 3 * config.hidden_size)

        # output projection
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projection.scale_init = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, 3 * Embed_Dim]
        qkv = self.attention(hidden_states)
        
        # split q, k, v
        q, k, v = qkv.split(self.config.hidden_size, dim=2)

        # changes shapes of q, k, v
        q = q.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2)
        k = k.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2)
        v = v.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2)

        # perform scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.config.attention_dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.output_projection(y)

        return y
        

class EncoderMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(hidden_states), approximate="tanh"))


class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = EncoderAttention(config)
        self.layer_norm1 = RMSNorm(self.config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EncoderMLP(config)
        self.layer_norm2 = RMSNorm(self.config.hidden_size, eps=config.rms_norm_eps)

    # Ignore copy
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.embeddings = EncoderEmbeddings(config)
        self.encoder = EncoderBlock(config)
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Channels, Image_Height, Image_Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.post_layernorm(self.encoder(inputs_embeds=self.embeddings(pixel_values)))


class EncoderModel(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.vision_model = Encoder(config)

    def forward(self, pixel_values) -> torch.Tensor:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 