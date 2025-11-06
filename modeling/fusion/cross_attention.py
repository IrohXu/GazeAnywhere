import logging
from itertools import repeat
from typing import Literal, Union, Iterable
from functools import partial
import torch
import torch.nn as nn
from detectron2.modeling import Backbone

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_ON = True
except ImportError:
    XFORMERS_ON = False

def to_2tuple(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        return_softmax_attn=True,
        use_proj=True,
        patch_token_offset=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) if use_proj else nn.Identity()

        self.return_softmax_attn = return_softmax_attn

        self.patch_token_offset = patch_token_offset

    def forward(self, x, return_attention=False, extra_token_offset=None):
        B, L, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, L, -1).unbind(0)

        if return_attention or not XFORMERS_ON:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if return_attention and not self.return_softmax_attn:
                out_attn = attn
            attn = attn.softmax(dim=-1)
            if return_attention and self.return_softmax_attn:
                out_attn = attn
            x = attn @ v
        else:
            x = memory_efficient_attention(q, k, v, scale=self.scale)

        x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        x = self.proj(x)

        if return_attention:
            out_attn = out_attn.reshape(B, self.num_heads, L, -1)
            out_attn = out_attn[
                :,
                :,
                self.patch_token_offset : extra_token_offset,
                self.patch_token_offset : extra_token_offset,
            ]
            return x, out_attn
        else:
            return x
    
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        return_softmax_attn=True,
        use_proj=True,
        patch_token_offset=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) if use_proj else nn.Identity()
        
        self.return_softmax_attn = return_softmax_attn
        self.patch_token_offset = patch_token_offset

    def forward(self, x, y):
        B, L, _ = x.shape
        _, L_y, _ = y.shape
        
        q = self.linear_q(x).view(B, L, self.num_heads, -1).transpose(1, 2)
        k = self.linear_k(y).view(B, L_y, self.num_heads, -1).transpose(1, 2)
        v = self.linear_v(y).view(B, L_y, self.num_heads, -1).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        x = self.proj(x)

        return x
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        init_values=None,
        return_softmax_attn=True,
        attention_map_only=False,
        patch_token_offset=0,
    ):
        super().__init__()
        self.attention_map_only = attention_map_only
        self.norm1_x = norm_layer(dim)
        self.norm1_y = norm_layer(dim)
        
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            return_softmax_attn=return_softmax_attn,
            use_proj=return_softmax_attn or not attention_map_only,
            patch_token_offset=patch_token_offset,
        )

        if attention_map_only:
            return

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, inputs, return_attention=False, extra_token_offset=None):
        x_input, y_input = inputs
        shortcut = x_input
        x = self.norm1_x(x_input)
        y = self.norm1_y(y_input)
                
        x = self.attn(x, y)

        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        
        return x, y_input


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma