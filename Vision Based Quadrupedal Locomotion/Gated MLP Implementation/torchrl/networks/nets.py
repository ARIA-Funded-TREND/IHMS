import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.networks.init as init
import torchrl.networks.base as base



class MultiheadAttention(nn.Module):
    """
    Keeps the same class name, but implements the Spatial Gating Unit (SGU)
    instead of multi-head attention.
    """
    def __init__(self, embed_dim, num_heads, seq_len=17, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(embed_dim // 2)
        # Token mixing
        self.spatial_proj = nn.Conv1d(
            seq_len,
            seq_len,
            kernel_size=1
        )
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_mask=None,
        key_padding_mask=None,
    ):
        """
        query : (B, N, D_ff)
        """
        u, v = query.chunk(2, dim=-1)
        v = self.norm(v)
        # (B, N, D/2)
        v = self.spatial_proj(v)
        out = u * v
        return out

###############################################################################

class MLP(nn.Module):
    """
    gMLP
    """
    def __init__(
        self,
        in_features,
        seq_len,
        hidden_features=None,
        out_features=None,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.channel_proj1 = nn.Linear(
            in_features,
            hidden_features
        )
        self.activation = nn.GELU()
        # Uses the class name MultiheadAttention
        self.sgu = MultiheadAttention(
            embed_dim=hidden_features,
            num_heads=1,
            seq_len=seq_len
        )
        self.channel_proj2 = nn.Linear(
            hidden_features // 2,
            out_features
        )
    def forward(self, x):
        x = self.channel_proj1(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x


###############################################################################

class TransformerEncoderLayer(nn.Module):
    """
    Same class name, but internally behaves like a gMLP block.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=False,
        seq_len=17,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.norm = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            seq_len=seq_len,
            hidden_features=dim_feedforward,
            out_features=d_model,
        )
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):

        if not self.batch_first:
            src = src.transpose(0, 1)
        residual = src
        x = self.norm(src)
        x = self.mlp(x)
        src = residual + self.dropout(x)
        if not self.batch_first:
            src = src.transpose(0, 1)
        return src


# class SpatialGatingUnit(nn.Module):
#     def __init__(self, d_ffn, seq_len):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_ffn //2 )
#         self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
#         nn.init.constant_(self.spatial_proj.bias, 1.0)

#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = self.spatial_proj(v)
#         out = u * v
#         return out


# class MLP(nn.Module):
#     """
#     Multi-Layer Perceptron (Feed-Forward Network) for Vision Transformer
#     """

#     # FIX 2: Add seq_len argument
#     def __init__(self, in_features, seq_len, hidden_features=None, out_features=None, activation=None, dropout=None):

#         super().__init__()

#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features * 4

        
#         self.norm = nn.LayerNorm(in_features)
#         self.channel_proj1 = nn.Linear(in_features, hidden_features)
#         self.channel_proj2 = nn.Linear(hidden_features // 2, out_features) # Note: SGU reduces dim by half
        
#         # FIX 4: Pass seq_len correctly
#         self.sgu = SpatialGatingUnit(hidden_features, seq_len)


#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (B, N, C)
#         Returns:
#             Output tensor of shape (B, N, C)
#         """
#         residual = x
#         x = self.norm(x)
#         x = F.gelu(self.channel_proj1(x))
#         x = self.sgu(x)
#         x = self.channel_proj2(x)
#         return out

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
#         super().__init__()
#         self.batch_first = batch_first
        
#         # 1. Multi-Head Attention from scratch
#         self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

#         mlp_hidden_dim = dim_feedforward
#         self.mlp = MLP(
#             in_features=d_model,
#             seq_len=17,  # Pass it down
#             hidden_features=mlp_hidden_dim,
#             activation=nn.GELU,
#             dropout=0.0
#         )

#         # 2. Feed-Forward Network
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         # 3. Normalization and Dropout layers (Post-LN architecture, PyTorch default)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)


#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         # Standard PyTorch default is (Sequence, Batch, Dim). 
#         # If batch_first=False, we transpose to (Batch, Sequence, Dim) for our MHA math.
#         if not self.batch_first:
#             src = src.transpose(0, 1)

#         # Block 1: Self-Attention + Add & Norm
#         attn_out = self.self_attn(
#             query=src, 
#             key=src, 
#             value=src, 
#             attn_mask=src_mask, 
#             key_padding_mask=src_key_padding_mask
#         )
#         src = src + self.dropout1(attn_out)
#         src = self.norm1(src)

#         # Block 2: Feed-Forward + Add & Norm
#         ffn_out = self.mlp(src)
#         src = src + self.dropout2(ffn_out)
#         src = self.norm2(src)

#         # Transpose back if standard PyTorch format was used
#         if not self.batch_first:
#             src = src.transpose(0, 1)

#         return src


# class MultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
        
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         # Linear projections for Query, Key, and Value
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
        
#         # Output projection
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
        
#         self.attn_dropout = nn.Dropout(dropout)

#         # Added for Bilal Gen
#         # self.adaLN_modulation = nn.Sequential(
#         # nn.SiLU(),
#         # nn.Linear(embed_dim, 3 * embed_dim, bias=True)
#         # )


#     def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
#         # We assume inputs are (Batch, Sequence, Dim) for the math below.
#         B, L_target, _ = query.shape
#         _, L_source, _ = key.shape

#         # 1. Project Q, K, V
#         q = self.q_proj(query) # (B, L_target, D)
#         k = self.k_proj(key)   # (B, L_source, D)
#         v = self.v_proj(value) # (B, L_source, D)

#         # 2. Reshape for Multi-Head: (B, Seq, Heads, Head_Dim) -> (B, Heads, Seq, Head_Dim)
#         q = q.view(B, L_target, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(B, L_source, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(B, L_source, self.num_heads, self.head_dim).transpose(1, 2)


#         # q = F.relu(q + q*k)
#         # k = F.relu(k + q*k)
#         # # # v = v**2 + 2*v + k*q*(1+torch.abs(v))
#         # # # v = v**2 + 2*v + k*(1+torch.abs(v))
#         # v = F.relu(v + v*k)

#         q = q + q*k#F.relu(q + q*k)
#         k = k + q*k#F.relu(k + q*k)
#         v = v + v*k#F.relu(v + v*k)

#         out = v

#         out = out.transpose(1, 2).contiguous().view(B, L_target, self.embed_dim)

#         # 8. Final Output Projection
#         out = self.out_proj(out)
#         return out


class ZeroNet(nn.Module):
  def forward(self, x):
    return torch.zeros(1)


class Net(nn.Module):
  def __init__(
      self,
      output_shape,
      base_type,
      append_hidden_shapes=[],
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      **kwargs):
    super().__init__()
    self.base = base_type(
      activation_func=activation_func,
      add_ln=add_ln,
      **kwargs)

    self.add_ln = add_ln
    self.activation_func = activation_func
    append_input_shape = self.base.output_shape
    self.append_fcs = []
    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(nn.LayerNorm(next_shape))
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

  def forward(self, x):
    out = self.base(x)
    out = self.seq_append_fcs(out)
    return out


class FlattenNet(Net):
  def forward(self, input):
    out = torch.cat(input, dim=-1)
    return super().forward(out)


class QNet(Net):
  def forward(self, input):
    assert len(input) == 2, "Q Net only get observation and action"
    state, action = input
    x = torch.cat([state, action], dim=-1)
    out = self.base(x)
    out = self.seq_append_fcs(out)
    return out


class BootstrappedNet(nn.Module):
  def __init__(
      self,
      output_shape,
      base_type, head_num=10,
      append_hidden_shapes=[],
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      **kwargs):
    super().__init__()
    self.base = base_type(
      activation_func=activation_func,
      add_ln=add_ln
      ** kwargs)
    self.add_ln = add_ln
    self.activation_func = activation_func

    self.bootstrapped_heads = []

    append_input_shape = self.base.output_shape

    for idx in range(head_num):
      append_input_shape = self.base.output_shape
      append_fcs = []
      for next_shape in append_hidden_shapes:
        fc = nn.Linear(append_input_shape, next_shape)
        append_hidden_init_func(fc)
        append_fcs.append(fc)
        append_fcs.append(self.activation_func())
        if self.add_ln:
          append_fcs.append(nn.LayerNorm(next_shape))
        # set attr for pytorch to track parameters( device )
        append_input_shape = next_shape

      last = nn.Linear(append_input_shape, output_shape)
      net_last_init_func(last)
      append_fcs.append(last)
      head = nn.Sequential(*append_fcs)
      self.__setattr__(
        "head{}".format(idx),
        head)
      self.bootstrapped_heads.append(head)

  def forward(self, x, head_idxs):
    output = []
    feature = self.base(x)
    for idx in head_idxs:
      output.append(self.bootstrapped_heads[idx](feature))
    return output


class FlattenBootstrappedNet(BootstrappedNet):
  def forward(self, input, head_idxs):
    out = torch.cat(input, dim=-1)
    return super().forward(out, head_idxs)


class NatureEncoderProjNet(nn.Module):
  def __init__(
      self,
      encoder,
      output_shape,
      visual_input_shape,
      append_hidden_shapes=[],
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach

    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.append_fcs = []

    append_input_shape = self.encoder.output_dim

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    self.normalizer = None

  def forward(self, x):

    visual_input = x.view(
      torch.Size(x.shape[:-1] + self.visual_input_shape)
    )

    out = self.encoder(
      visual_input,
      detach=self.detach
    )

    out = self.seq_append_fcs(out)
    return out


class ImpalaEncoderProjNet(nn.Module):
  def __init__(
    self,

    encoder,

    output_shape,

    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],

      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach

    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.append_fcs = []

    append_input_shape = self.encoder.base.output_shape + self.encoder.visual_dim

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    self.normalizer = None

  def forward(self, state):

    state_input = state[..., :self.state_input_shape]
    visual_input = state[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )

    visual_out, state_out = self.encoder(
      visual_input, state_input,
      detach=self.detach
    )

    out = torch.cat([visual_out, state_out], dim=-1)

    out = self.seq_append_fcs(out)
    return out


class ImpalaEncoderProjResidualActor(nn.Module):
  def __init__(
    self,
    encoder,
    projector,

    output_shape,

    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],
    state_hidden_shapes=[],

      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      visual_scale=0.3,
      **kwargs):
    super().__init__()
    self.encoder = encoder
    self.projector = projector

    self.add_ln = add_ln
    self.visual_scale = visual_scale
    self.detach = detach
    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.base = networks.MLPBase(
      input_shape=state_input_shape,
      hidden_shapes=state_hidden_shapes,
      add_ln=add_ln, activation_func=activation_func, **kwargs
    )

    self.append_fcs = []

    # assert self.projector.output_dim == self.state_base.output_shape
    # add_proj_linear = nn.Linear(
    #     self.projector.output_dim, self.base.output_shape
    # )
    # net_last_init_func(add_proj_linear)
    # # torch.nn.init.zeros_(add_proj_linear.weight)
    # # torch.nn.init.zeros_(add_proj_linear.bias)

    # self.additional_proj = nn.Sequential(
    #     add_proj_linear,
    #     nn.LayerNorm(
    #         self.base.output_shape
    #     ),
    #     nn.Sigmoid()
    # )

    append_input_shape = self.base.output_shape

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    visual_append_input_shape = self.projector.output_dim

    self.visual_append_fcs = []
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None

  # def forward(self, x, detach=False):
  def forward(self, x):
    state_input = x[..., :self.state_input_shape]
    state_out = self.base(state_input)
    state_out = self.seq_append_fcs(state_out)

    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )
    out = self.encoder(visual_input, detach=self.detach)
    out = self.projector(out)
    out = self.visual_seq_append_fcs(out)

    # out = self.additional_proj(out)
    # out = out * self.visual_scale

    # out = torch.cat([out, state_out], dim=-1)
    # out = out + state_out
    return out + state_out


class ImpalaFuseResidualActor(nn.Module):
  def __init__(
    self,
    encoder,

    output_shape,

    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],

    append_hidden_init_func=init.basic_init,
    net_last_init_func=init.uniform_init,
    activation_func=nn.ReLU,
    add_ln=False,
    detach=False,
    state_detach=False,
    # visual_scale=0.3,
    displacement_dim=7,
    history=3,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach
    self.state_detach = state_detach

    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape
    self.displacement_dim = displacement_dim
    self.history = history
    self.activation_func = activation_func

    self.append_fcs = []

    append_input_shape = self.encoder.base.output_shape

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    visual_append_input_shape = self.encoder.visual_dim + \
      self.encoder.base.output_shape

    self.visual_append_fcs = []
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(visual_last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None

  # def forward(self, x, detach=False):
  def forward(self, x):
    state_input = x[..., :self.state_input_shape]
    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )

    visual_out, state_out = self.encoder(
      visual_input, state_input,
      detach=self.detach
    )

    if self.state_detach:
      out = torch.cat(
        [visual_out, state_out.detach()],
        dim=-1
      )
    else:
      out = torch.cat([visual_out, state_out], dim=-1)
    out = self.visual_seq_append_fcs(out)

    state_out = self.seq_append_fcs(state_out)

    return out + state_out

  def forward_and_compute_aux_loss(self, x):
    state_input = x[..., :self.state_input_shape]
    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )
    dispalcement_gt = state_input[..., :self.history * self.displacement_dim].view(
      x.shape[0], -1, self.displacement_dim)
    visual_out, state_out, visual_sub_out = self.encoder.forward_with_sub_vec(
      visual_input, state_input,
      detach=self.detach
    )

    if self.state_detach:
      out = torch.cat(
        [visual_out, state_out.detach()],
        dim=-1
      )
    else:
      out = torch.cat([visual_out, state_out], dim=-1)
    out = self.visual_seq_append_fcs(out)

    state_out = self.seq_append_fcs(state_out)
    aux_loss = F.mse_loss(dispalcement_gt, visual_sub_out)
    return out + state_out, aux_loss


class ImpalaWeightedFuseResidualActor(nn.Module):
  def __init__(
    self,
    encoder,

    output_shape,

    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],

    append_hidden_init_func=init.basic_init,
    net_last_init_func=init.uniform_init,
    activation_func=nn.ReLU,
    add_ln=False,
    detach=True,
    state_detach=False,
    # visual_scale=0.3,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach
    self.state_detach = state_detach

    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.append_fcs = []

    append_input_shape = self.encoder.base.output_shape

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    visual_append_input_shape = self.encoder.visual_dim + \
      self.encoder.base.output_shape

    self.visual_append_fcs = []
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(visual_last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None
    self.k = nn.Parameter(torch.zeros(1))

  # def forward(self, x, detach=False):
  def forward(self, x):
    state_input = x[..., :self.state_input_shape]
    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )

    visual_out, state_out = self.encoder(
      visual_input, state_input,
      detach=self.detach
    )

    if self.state_detach:
      out = torch.cat(
        [visual_out, state_out.detach()],
        dim=-1
      )
    else:
      out = torch.cat([visual_out, state_out], dim=-1)
    out = self.visual_seq_append_fcs(out)

    state_out = self.seq_append_fcs(state_out)

    return self.k * out + state_out


class ImpalaMixResidualActor(nn.Module):
  def __init__(
    self,
    encoder,
    projector,

    output_shape,

    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],
    state_hidden_shapes=[],

    append_hidden_init_func=init.basic_init,
    net_last_init_func=init.uniform_init,
    activation_func=nn.ReLU,
    add_ln=False,
    detach=True,
    # visual_scale=0.3,
      **kwargs):
    super().__init__()
    self.encoder = encoder
    self.projector = projector

    self.add_ln = add_ln
    # self.visual_scale = visual_scale
    self.detach = detach
    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.base = networks.MLPBase(
      input_shape=state_input_shape,
      hidden_shapes=state_hidden_shapes,
      add_ln=add_ln, activation_func=activation_func,
      last_activation_func=nn.Tanh, **kwargs
    )

    self.append_fcs = []

    append_input_shape = self.base.output_shape

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    visual_append_input_shape = self.projector.output_dim + self.base.output_shape

    self.visual_append_fcs = []
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None

  # def forward(self, x, detach=False):
  def forward(self, x):
    state_input = x[..., :self.state_input_shape]
    state_out = self.base(state_input)

    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )
    out = self.encoder(visual_input, detach=self.detach)
    out = self.projector(out)
    out = torch.cat([out, state_out], dim=-1)
    out = self.visual_seq_append_fcs(out)

    state_out = self.seq_append_fcs(state_out)

    return out + state_out


class VisualNet(nn.Module):
  def __init__(
    self,
    encoder,
    output_shape,
    state_input_shape,
    visual_input_shape,

    append_hidden_shapes=[],

      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach

    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape

    self.activation_func = activation_func

    self.append_fcs = []

    append_input_shape = self.encoder.visual_dim

    for next_shape in append_hidden_shapes:
      fc = nn.Linear(append_input_shape, next_shape)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      append_input_shape = next_shape

    last = nn.Linear(append_input_shape, output_shape)
    net_last_init_func(last)

    self.append_fcs.append(last)
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    self.normalizer = None

  def forward(self, state, return_state=False):

    state_input = state[..., :self.state_input_shape]

    visual_input = state[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )

    visual_out = self.encoder(visual_input, detach=self.detach)

    out = self.seq_append_fcs(visual_out)
    if return_state:
      return out, state_input
    return out


class Transformer(nn.Module):
  def __init__(
      self,
      encoder,
      output_shape,
      visual_input_shape,
      transformer_params=[],
      append_hidden_shapes=[],
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      state_detach=False,
      max_pool=False,
      token_norm=False,
      use_pytorch_encoder=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach
    self.state_detach = state_detach

    self.visual_input_shape = visual_input_shape
    self.activation_func = activation_func

    self.max_pool = max_pool
    visual_append_input_shape = self.encoder.visual_dim

    self.token_norm = token_norm
    if self.token_norm:
      self.token_ln = nn.LayerNorm(self.encoder.visual_dim)
      self.state_token_ln = nn.LayerNorm(self.encoder.visual_dim)

    self.use_pytorch_encoder = use_pytorch_encoder
    if not self.use_pytorch_encoder:
      self.visual_append_layers = nn.ModuleList()
      for n_head, dim_feedforward in transformer_params:
        visual_att_layer = TransformerEncoderLayer(
          visual_append_input_shape, n_head, dim_feedforward,
          dropout=0
        )
        self.visual_append_layers.append(visual_att_layer)
    else:
      encoder_layer = TransformerEncoderLayer(
        self.encoder.visual_dim, transformer_params[0][0], transformer_params[0][1],
        dropout=0
      )
      encoder_norm = nn.LayerNorm(self.encoder.visual_dim)
      self.visual_trans_encoder = TransformerEncoder(
        encoder_layer, len(transformer_params), encoder_norm
      )
    # self.visual_atts = nn.Sequential(*self.visual_append_layers)

    self.per_modal_tokens = self.encoder.per_modal_tokens
    if self.encoder.in_channels == 4 or self.encoder.in_channels == 12:
      self.second = False
    else:
      self.second = True

    self.visual_append_fcs = []
    visual_append_input_shape = visual_append_input_shape
    if self.second:
      visual_append_input_shape += self.encoder.visual_dim
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(visual_last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None

  def forward(self, x):
    visual_input = x.view(
      torch.Size(x.shape[:-1] + self.visual_input_shape)
    )

    visual_out = self.encoder(
      visual_input,
      detach=self.detach
    )
    out = visual_out
    if self.token_norm:
      out = self.token_ln(out)
    if not self.use_pytorch_encoder:
      for att_layer in self.visual_append_layers:
        out = att_layer(out)
    else:
      out = self.visual_trans_encoder(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    if self.max_pool:
      out_first = out[0: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[0: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_first]
    if self.second:
      out_second = out[self.per_modal_tokens: 2 * self.per_modal_tokens, ...]
      if self.max_pool:
        out_second = out_second.max(dim=0)[0]
      else:
        out_second = out_second.mean(dim=0)
      out_list.append(out_second)

    # out_depth = out_depth

    out = torch.cat(out_list, dim=-1)
    # (Batch_size, Feature Dim * 3)
    out = self.visual_seq_append_fcs(out)

    return out


class LocoTransformer(nn.Module):
  def __init__(
      self,
      encoder,
      output_shape,
      state_input_shape,
      visual_input_shape,
      transformer_params=[],
      append_hidden_shapes=[],
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      add_ln=False,
      detach=False,
      state_detach=False,
      max_pool=False,
      token_norm=False,
      use_pytorch_encoder=False,
      **kwargs):
    super().__init__()
    self.encoder = encoder

    self.add_ln = add_ln
    self.detach = detach
    self.state_detach = state_detach

    self.state_input_shape = state_input_shape
    self.visual_input_shape = visual_input_shape
    self.activation_func = activation_func

    self.max_pool = max_pool
    visual_append_input_shape = self.encoder.visual_dim

    self.token_norm = token_norm
    if self.token_norm:
      self.token_ln = nn.LayerNorm(self.encoder.visual_dim)
      self.state_token_ln = nn.LayerNorm(self.encoder.visual_dim)

    self.use_pytorch_encoder = use_pytorch_encoder
    if not self.use_pytorch_encoder:
      self.visual_append_layers = nn.ModuleList()
      for n_head, dim_feedforward in transformer_params:
        visual_att_layer = TransformerEncoderLayer(
          visual_append_input_shape, n_head, dim_feedforward,
          dropout=0
        )
        self.visual_append_layers.append(visual_att_layer)
    else:
      encoder_layer = TransformerEncoderLayer(
        self.encoder.visual_dim, transformer_params[0][0], transformer_params[0][1],
        dropout=0
      )
      encoder_norm = nn.LayerNorm(self.encoder.visual_dim)
      self.visual_trans_encoder = TransformerEncoder(
        encoder_layer, len(transformer_params), encoder_norm
      )
    # self.visual_atts = nn.Sequential(*self.visual_append_layers)

    self.per_modal_tokens = self.encoder.per_modal_tokens
    if self.encoder.in_channels == 4 or self.encoder.in_channels == 12:
      self.second = False
    else:
      self.second = True

    self.visual_append_fcs = []
    visual_append_input_shape = visual_append_input_shape * 2
    if self.second:
      visual_append_input_shape += self.encoder.visual_dim
    for next_shape in append_hidden_shapes:
      visual_fc = nn.Linear(visual_append_input_shape, next_shape)
      append_hidden_init_func(visual_fc)
      self.visual_append_fcs.append(visual_fc)
      self.visual_append_fcs.append(self.activation_func())
      if self.add_ln:
        self.visual_append_fcs.append(
          nn.LayerNorm(next_shape)
        )
      visual_append_input_shape = next_shape

    visual_last = nn.Linear(visual_append_input_shape, output_shape)
    net_last_init_func(visual_last)

    self.visual_append_fcs.append(visual_last)
    self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)

    self.normalizer = None

  def forward(self, x):
    state_input = x[..., :self.state_input_shape]
    visual_input = x[..., self.state_input_shape:].view(
      torch.Size(state_input.shape[:-1] + self.visual_input_shape)
    )

    # print("Visual Input Mean: ",visual_input.mean().item())
    # print("Visual Input Max: ",visual_input.max().item())
    # print("Visual Input Min: ",visual_input.min().item())

    visual_out, state_out = self.encoder(
      visual_input, state_input,
      detach=self.detach
    )
    out = visual_out
    if self.token_norm:
      out = self.token_ln(out)
    if not self.use_pytorch_encoder:
      for att_layer in self.visual_append_layers:
        out = att_layer(out)
    else:
      out = self.visual_trans_encoder(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    # rgb_depth_channels = (self.encoder.channel_count - 1) // 2
    # out_rgb = out[1: rgb_depth_channels + 1, ...]
    if self.max_pool:
      out_first = out[1: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[1: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_state, out_first]
    if self.second:
      out_second = out[1 + self.per_modal_tokens: 1 +
                       2 * self.per_modal_tokens, ...]
      if self.max_pool:
        out_second = out_second.max(dim=0)[0]
      else:
        out_second = out_second.mean(dim=0)
      out_list.append(out_second)

    # out_depth = out_depth

    out = torch.cat(out_list, dim=-1)
    # (Batch_size, Feature Dim * 3)
    out = self.visual_seq_append_fcs(out)

    return out
