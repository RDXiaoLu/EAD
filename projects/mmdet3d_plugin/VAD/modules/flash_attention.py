import torch
from torch import nn
from torch.nn import functional as F
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv import ConfigDict, deprecated_api_warning
from mmcv.runner.base_module import BaseModule
from mmcv.runner import auto_fp16

from torch.nn.init import xavier_uniform_, constant_

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.nn.functional import linear
from einops import rearrange
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_unpadded_kvpacked_func')
except:
    from flash_attn.flash_attn_interface import  flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_varlen_kvpacked_func')
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

def identity(x, *args, **kwargs):
    return x

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None

def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size))==1).transpose(0,1)
    return (torch.triu(torch.ones(dim_k, input_size))==1).transpose(0,1)

def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class Residual(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    However, I do postnorm instead of prenorm.
    """
    def __init__(self, fn, input_channels=0, output_channels=0):
        super(Residual, self).__init__()
        self.fn = fn
        self.resample = nn.Linear(input_channels, output_channels) if input_channels != output_channels else None
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, tensor, **kwargs):
        if self.resample is not None:
            tensor = self.resample(tensor) + self.fn(tensor, **kwargs)
            tensor = self.norm(tensor)
            return tensor
        tensor = tensor + self.fn(tensor, **kwargs)
        tensor = self.norm(tensor)
        return tensor

class PositionalEmbedding(nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer sequence lengths.
    """
    def __init__(self, channels):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1. / (100000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        pos = torch.arange(tensor.shape[1], device=tensor.device).type(self.inv_freq.type())
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb[None,:,:]

class ProjectInOut(nn.Module):
    """
    Impelemenation taken from https://github.com/lucidrains/sinkhorn-transformer/blob/73da02958965e1a690cb301292c0a3c549687d44/sinkhorn_transformer/sinkhorn_transformer.py#L218
    """
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super(ProjectInOut, self).__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, tensor, **kwargs):
        tensor = self.project_in(tensor)
        tensor = self.fn(tensor, **kwargs)
        tensor = self.project_out(tensor)
        return tensor

class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, input_channels, output_channels, ff_dim, dropout, activation="gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        if self.activation is not None:
            tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor

def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        torch.cuda.empty_cache()  
        
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        print(f"q shape: {q.shape}, kv shape: {kv.shape}")
        if kv.shape[0] != q.shape[0]: 
            kv = kv.mean(dim=0, keepdim=True)
        print(f"q 2 shape: {q.shape}, kv 2 shape: {kv.shape}")    
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)         
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)  
            
            # 示例性检查  
            assert cu_seqlens_k.shape[0] == batch_size + 1, "cu_seqlens_k should have shape (batch_size + 1)"
            
            print(f"1q: {q.shape}", f"kv: {kv.shape}")  
            print(f"cu_seqlens_q: {cu_seqlens_q}")   
            print(f"cu_seqlens_k: {cu_seqlens_k}")
            print(f"max_sq: {max_sq}")  
            print(f"max_sk: {max_sk}")  
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            print(f"output shape: {output.shape}") 
        else:
            # nheads = kv.shape[-2]
            # q = rearrange(q, 'b s ... -> (b s) ...')
            # max_sq = seqlen_q
            # cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
            #                         device=q.device)
            # x = rearrange(kv, 'b s two h d -> b s (two h d)')
            # # 确保 key_padding_mask 有效  
            # key_padding_mask = key_padding_mask.view(batch_size, -1)  # 或其他方式调整形状以匹配  

            # # 检查有效 token  
            # # assert key_padding_mask.shape == (batch_size, seqlen_k), "key_padding_mask shape should match (batch_size, seqlen)"  
            # print(f"key_padding_mask shape: {key_padding_mask.shape}, effective tokens: {key_padding_mask.sum()}")  
            # effective_tokens = key_padding_mask.sum() 
            # if effective_tokens == 0:  
            #     print("Warning: No effective tokens found in key_padding_mask.")  
            #     # 在没有有效 token 的情况下，我们必须做出合理的处理决策。  
            #     # 例如，可以选择返回零填充的输出等。  

            #     # 在只处理没有有效 token 的情况下返回一个形状正确的零张量  
            #     # output_shape =   # output shape should match expected  
            #     return None, None    
            
            # x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            # if x_unpad.shape[0] == 0:  
            #      x_unpad = torch.zeros((q.shape[0], x.shape[-1]), device=x.device, dtype=torch.float16)
            #      print(f"x_unpad 0 -> rechange shape: {x_unpad.shape}")

            # x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            # # print(f"x_unpad shape: {x_unpad.shape}") 
            # # print(f"x: {x.shape}")  
            # # print(f"key_padding_mask: {key_padding_mask.shape}")  
            # # print(f"batch_size: {batch_size}")  
            # # print(f"cu_seqlens_q: {cu_seqlens_q}")  
            # # print(f"cu_seqlens_k: {cu_seqlens_k}")  
            # # print(f"q shape: {q.shape}, x_unpad shape: {x_unpad.shape}")   
            # # print(f"seqlen_k shape: {seqlen_k}")
            
           
            # seqlen_k = torch.full((batch_size,), seqlen_k, device='cuda', dtype=torch.int32)  
            # cu_seqlens_k = torch.zeros(len(seqlen_k) + 1, dtype=torch.int32, device='cuda')  
            # cu_seqlens_k[1:] = torch.cumsum(seqlen_k, dim=0)  # 正确计算累计和             
          
            
            # q = q.unsqueeze(0)
            # x_unpad = x_unpad.unsqueeze(0)
            # q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            
            # print(f"2q: {q.shape}", f"x_unpad: {x_unpad.shape}")  
            # print(f"cu_seqlens_q: {cu_seqlens_q}")   
            # print(f"cu_seqlens_k: {cu_seqlens_k}")
            # print(f"max_sq: {max_sq}")  
            # print(f"max_sk: {max_sk}") 
            # output_unpad = flash_attn_unpadded_kvpacked_func(
            #     q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
            #     self.dropout_p if self.training else 0.0,
            #     softmax_scale=self.softmax_scale, causal=causal
            # )
            # output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)
            
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            
            key_padding_mask = key_padding_mask.permute(0, 1).contiguous()  # 确保为 (batch_size, seqlen)  
            effective_lengths = key_padding_mask.sum(dim=1)  # (batch_size, )  
            max_sk = effective_lengths.max().item()           # 获取最大有效长度  
            cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=kv.device)  
            
            for i in range(batch_size):  
                cu_seqlens_k[i + 1] = cu_seqlens_k[i] + effective_lengths[i]  
            
            cu_seqlens_k = cu_seqlens_q
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, _, max_sk = unpad_input(x, key_padding_mask)
            
            
            # 确保 x_unpad 相关的 Tensor 在适当状态  
            if x_unpad.shape[0] == 0:  
                print(f"x_unpad is empty, initializing with zeros.")  
                x_unpad = torch.zeros((batch_size,  x.shape[-1]), device=x.device, dtype=x.dtype)  # 用零填充  
    
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads).contiguous()  
            print(f"x_unpad shape: {x_unpad.shape}") 
            print(f"x: {x.shape}")  
            print(f"key_padding_mask: {key_padding_mask.shape}")  
            print(f"batch_size: {batch_size}")  
            print(f"cu_seqlens_q: {cu_seqlens_q}")  
            print(f"cu_seqlens_k: {cu_seqlens_k}")  
            print(f"q shape: {q.shape}, x_unpad shape: {x_unpad.shape}")   
            print(f"seqlen_k shape: {seqlen_k}")
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q.contiguous(), x_unpad.contiguous(), cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
                max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)

        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights

@ATTENTION.register_module()
class MultiEfficientAttention(BaseModule):  
    def __init__(self,  
                 embed_dims,  
                 num_heads,
                 attn_drop=0.,  
                 proj_drop=0.,  
                 dropout_layer=dict(type='Dropout', drop_prob=0.),  
                 init_cfg=None,  
                 batch_first=True,

                 **kwargs):  
        super(MultiEfficientAttention, self).__init__(init_cfg)  
        
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
            
        self.embed_dims = embed_dims  
        self.num_heads = num_heads
        self.batch_first = True  
        # E_proj = nn.Linear(embed_dims, embed_dims)  # 假设输入输出维度相同  
        # F_proj = nn.Linear(embed_dims, embed_dims)  # 假设输入输出维度相同
        # 直接创建 LinearAttentionHead，没有 num_heads 参数  
        self.attn = FlashMHA(
            embed_dim=embed_dims, 
            num_heads=num_heads, 
            attention_dropout=attn_drop, 
            dtype=torch.float16, 
            device='cuda',
            **kwargs
        )

        self.proj_drop = nn.Dropout(proj_drop)  
        self.dropout_layer = build_dropout(  
            dropout_layer) if dropout_layer else nn.Identity()  
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,  
                query,  
                key=None,  
                value=None,  
                identity=None,  
                query_pos=None,  
                key_pos=None,  
                attn_mask=None,  
                key_padding_mask=None,  
                **kwargs):  
        
        if key is None:  
            key = query  
        if value is None:  
            value = key  
        if identity is None:  
            identity = query  
            
        if key_pos is None:  
            if query_pos is not None:  
                if query_pos.shape == key.shape:  
                    key_pos = query_pos  
                else:  
                    warnings.warn(f'Position encoding of key is missing in {self.__class__.__name__}.')  

        if query_pos is not None:  
            query = query + query_pos  
        if key_pos is not None:  
            key = key + key_pos  

        # Adjust tensor shapes for LinearAttentionHead  
        if not self.batch_first:  
            query = query.transpose(0, 1)  
            key = key.transpose(0, 1)  
            value = value.transpose(0, 1)  

        # 调用自定义的注意力头  
  
        out = self.attn(q = query, k=key, v=value, key_padding_mask=key_padding_mask)[0]  
        
        
        if not self.batch_first:  
            out = out.transpose(0, 1)  
        
        # print(f"Output shape: {out.shape}, type: {type(out)}")
        # print(f"identity shape: {identity.shape}, type: {type(identity)}")
        return identity + self.dropout_layer(self.proj_drop(out))

# @ATTENTION.register_module()
# class MultiEfficientAttention(BaseModule):
#     """A wrapper for ``torch.nn.MultiheadAttention``.

#     This module implements MultiheadAttention with identity connection,
#     and positional encoding  is also passed as input.

#     Args:
#         embed_dims (int): The embedding dimension.
#         num_heads (int): Parallel attention heads.
#         attn_drop (float): A Dropout layer on attn_output_weights.
#             Default: 0.0.
#         proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
#             Default: 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): When it is True,  Key, Query and Value are shape of
#             (batch, n, embed_dim), otherwise (n, batch, embed_dim).
#              Default to False.
#     """

#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  attn_drop=0.,
#                  proj_drop=0.,
#                  dropout_layer=dict(type='Dropout', drop_prob=0.),
#                  init_cfg=None,
#                  batch_first=False,
#                  **kwargs):
#         super(MultiEfficientAttention, self).__init__(init_cfg)
#         if 'dropout' in kwargs:
#             warnings.warn('The arguments `dropout` in MultiheadAttention '
#                           'has been deprecated, now you can separately '
#                           'set `attn_drop`(float), proj_drop(float), '
#                           'and `dropout_layer`(dict) ')
#             attn_drop = kwargs['dropout']
#             dropout_layer['drop_prob'] = kwargs.pop('dropout')

#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.batch_first = batch_first

#         self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
#                                           **kwargs)

#         self.proj_drop = nn.Dropout(proj_drop)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else nn.Identity()

#     @deprecated_api_warning({'residual': 'identity'},
#                             cls_name='MultiheadAttention')
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_mask=None,
#                 key_padding_mask=None,
#                 **kwargs):
#         """Forward function for `MultiheadAttention`.

#         **kwargs allow passing a more general data flow when combining
#         with other operations in `transformerlayer`.

#         Args:
#             query (Tensor): The input query with shape [num_queries, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#                 If None, the ``query`` will be used. Defaults to None.
#             value (Tensor): The value tensor with same shape as `key`.
#                 Same in `nn.MultiheadAttention.forward`. Defaults to None.
#                 If None, the `key` will be used.
#             identity (Tensor): This tensor, with the same shape as x,
#                 will be used for the identity link.
#                 If None, `x` will be used. Defaults to None.
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `x`. If not None, it will
#                 be added to `x` before forward function. Defaults to None.
#             key_pos (Tensor): The positional encoding for `key`, with the
#                 same shape as `key`. Defaults to None. If not None, it will
#                 be added to `key` before forward function. If None, and
#                 `query_pos` has the same shape as `key`, then `query_pos`
#                 will be used for `key_pos`. Defaults to None.
#             attn_mask (Tensor): ByteTensor mask with shape [num_queries,
#                 num_keys]. Same in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
#                 Defaults to None.

#         Returns:
#             Tensor: forwarded results with shape
#                 [num_queries, bs, embed_dims]
#                 if self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#         """

#         if key is None:
#             key = query
#         if value is None:
#             value = key
#         if identity is None:
#             identity = query
#         if key_pos is None:
#             if query_pos is not None:
#                 # use query_pos if key_pos is not available
#                 if query_pos.shape == key.shape:
#                     key_pos = query_pos
#                 else:
#                     warnings.warn(f'position encoding of key is'
#                                   f'missing in {self.__class__.__name__}.')
#         if query_pos is not None:
#             query = query + query_pos
#         if key_pos is not None:
#             key = key + key_pos

#         # Because the dataflow('key', 'query', 'value') of
#         # ``torch.nn.MultiheadAttention`` is (num_query, batch,
#         # embed_dims), We should adjust the shape of dataflow from
#         # batch_first (batch, num_query, embed_dims) to num_query_first
#         # (num_query ,batch, embed_dims), and recover ``attn_output``
#         # from num_query_first to batch_first.
#         if self.batch_first:
#             query = query.transpose(0, 1)
#             key = key.transpose(0, 1)
#             value = value.transpose(0, 1)

#         out = self.attn(
#             query=query,
#             key=key,
#             value=value,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask)[0]

#         if self.batch_first:
#             out = out.transpose(0, 1)

#         return identity + self.dropout_layer(self.proj_drop(out))