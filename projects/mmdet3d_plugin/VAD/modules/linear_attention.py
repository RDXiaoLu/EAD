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
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        print(f"q shape: {q.shape}, kv shape: {kv.shape}")
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
            
            print(f"batch_size: {batch_size}")  
            print(f"cu_seqlens_q: {cu_seqlens_q}")  
            print(f"cu_seqlens_k: {cu_seqlens_k}")  
            print(f"q shape: {q.shape}")   
            print(f"seqlen_k shape: {seqlen_k}")
            
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)

            print(f"batch_size: {batch_size}")  
            print(f"cu_seqlens_q: {cu_seqlens_q}")  
            print(f"cu_seqlens_k: {cu_seqlens_k}")  
            print(f"q shape: {q.shape}, x_unpad shape: {x_unpad.shape}")   
            print(f"seqlen_k shape: {seqlen_k}")
            
           
            seqlen_k = torch.full((batch_size,), seqlen_k, device='cuda', dtype=torch.int32)  
            cu_seqlens_k = torch.zeros(len(seqlen_k) + 1, dtype=torch.int32, device='cuda')  
            cu_seqlens_k[1:] = torch.cumsum(seqlen_k, dim=0)  # 正确计算累计和  
            
            print(f"batch_size: {batch_size}")  
            print(f"cu_seqlens_q: {cu_seqlens_q}")  
            print(f"cu_seqlens_k: {cu_seqlens_k}")  
            print(f"q shape: {q.shape}, x_unpad shape: {x_unpad.shape}")   
            print(f"seqlen_k shape: {seqlen_k}")
            
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
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


# class LinearAttentionHead(nn.Module):
#     """
#     Linear attention, as proposed by the linformer paper
#     """
#     def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
#         super(LinearAttentionHead, self).__init__()
#         self.E = E_proj
#         self.F = F_proj
#         self.dim = dim
#         self.dropout = nn.Dropout(dropout)
#         self.P_bar = None
#         self.full_attention = full_attention
#         self.causal_mask = causal_mask
#         self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

#     def forward(self, Q, K, V, **kwargs):
#         """
#         Assume Q, K, V have same dtype
#         E, F are `nn.Linear` modules
#         """
#         input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
#         embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

#         # Instead of classic masking, we have to do this, because the classic mask is of size nxn
#         if input_mask is not None:
#             # This is for k, v
#             mask = input_mask[:,:,None]
#             K = K.masked_fill_(~mask, 0.0)
#             V = V.masked_fill_(~mask, 0.0)
#             del mask

#         if embeddings_mask is not None:
#             mask = embeddings_mask[:,:,None]
#             Q = Q.masked_fill_(~mask, 0.0)
#             del mask

#         K = K.transpose(1,2)
#         if not self.full_attention:
#             if self.is_proj_tensor:
#                 self.E = self.E.to(K.device)
#                 K = torch.matmul(K, self.E)
#             else:
#                 K = self.E(K)
#         Q = torch.matmul(Q, K)

#         P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
#         if self.causal_mask is not None:
#             self.causal_mask = self.causal_mask.to(Q.device)
#             P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
#         P_bar = P_bar.softmax(dim=-1)

#         # Only save this when visualizing
#         if "visualize" in kwargs and kwargs["visualize"] == True:
#             self.P_bar = P_bar

#         P_bar = self.dropout(P_bar)

#         if not self.full_attention:
#             V = V.transpose(1,2)
#             if self.is_proj_tensor:
#                 self.F = self.F.to(V.device)
#                 V = torch.matmul(V, self.F)
#             else:
#                 V = self.F(V)
#             V = V.transpose(1,2)
#         out_tensor = torch.matmul(P_bar, V)

#         return out_tensor

# class LinearAttentionHead(nn.Module):  
#     def __init__(self, dim, dropout, E_proj, F_proj, causal_mask=None, full_attention=False):  
#         super(LinearAttentionHead, self).__init__()  
#         self.E = E_proj  
#         self.F = F_proj  
#         self.dim = dim  
#         self.dropout_layer = nn.Dropout(dropout)  # 定义 dropout 
#         self.full_attention = full_attention  
#         self.causal_mask = causal_mask  

#     def forward(self, Q, K, V, attn_mask=None):  # 接受 attn_mask  
#         K = K.transpose(1, 2)  # 转置 K 的尺寸  

#         # 打印 K 的形状用于调试  
#         print(f"K shape before E projection: {K.shape}")  

#         # 处理 K 的投影  
#         if not self.full_attention:  
#             K = K.reshape(-1, 4)  # 重新调整 K 的形状为 (batch_size * 256, 4)  
#             K = self.E(K)  # 投影 K  
#             # 这里需要恢复 K 的形状为 (batch_size, 256, features)  
#             K = K.view(-1, 256, 256)  

#         print(f"K shape after E projection: {K.shape}")   
        
#         # 处理 Q 的投影  
#         Q = Q.reshape(-1, 4)  # Q 的形状变化  
#         Q = self.F(Q)  # 投影 Q  
#         Q = Q.view(-1, 256, 256)  # 假设 Q 的形状应与 K 的形状一致  
#         print(f"Q shape after F projection: {Q.shape}")  

#         # 在这里进行 Q 和 K 的矩阵乘法  
#         QK = torch.matmul(Q, K.transpose(-2, -1))  # 转置 K 使其形状为 (batch_size, 256, 256)  
#         QK = QK / (self.dim ** 0.5)  # 确保放缩  
        
#         # 应用注意力掩码  
#         if attn_mask is not None:  
#             QK = QK.masked_fill(attn_mask == 0, float('-inf'))  

#         attn_weights = F.softmax(QK, dim=-1)  
        
#         attn_weights = self.dropout_layer(attn_weights)  # 只对 attn_weights 应用 dropout 

#         # 投影 V  
#         if not self.full_attention:  
#             V = V.reshape(-1, 4)  # 处理 V 的形状变化  
#             V = self.F(V)  # 投影 V  
#             V = V.view(-1, 256, 256)  # 假设 V 最后也成为 (batch_size, 256, 256)  

#         # 计算输出  
#         out_tensor = torch.matmul(attn_weights, V)  
#         return out_tensor, attn_weights  # 返回输出和注意力权重  

# class LinearAttentionHead(nn.Module):  
#     """  
#     Linear attention, as proposed by the linformer paper  
#     """  
#     def __init__(self, dim, dropout, E_proj, F_proj, causal_mask=None, full_attention=False):  
#         super(LinearAttentionHead, self).__init__()  
#         self.E = E_proj  # K 的线性投影层  
#         self.F = F_proj  # V 的线性投影层  
#         self.dim = dim  
#         self.dropout_layer = nn.Dropout(dropout)  # 定义 dropout   
#         self.full_attention = full_attention  
#         self.causal_mask = causal_mask  

#     def forward(self, Q, K, V, attn_mask=None):  
#         K = K.transpose(1, 2)  # 转置 K 的尺寸  
        
#         print(f"Q shape before E projection: {Q.shape}")  
#         print(f"K shape before E projection: {K.shape}")  
#         print(f"V shape before E projection: {V.shape}")  

#         # 处理 K 和 V 的投影  
#         if not self.full_attention:  
#             K = self.E(K)  # 投影 K，K 的形状会变为 (batch_size, features, 256)  
#             V = self.F(V.transpose(1, 2)).transpose(1, 2)  # 先转置 V，投影 V 后再转置回来，使得 V 的形状变为 (batch_size, seq_length, out_features)  
        
#         print(f"K shape after E projection: {K.shape}")   
#         print(f"V shape after F projection: {V.shape}")  

#         # 进行 Q 和 K 的矩阵乘法  
#         QK = torch.matmul(Q, K.transpose(-2, -1))  # 计算注意力分数，确保 Q 是 (batch_size, seq_length, features)  
#         QK = QK / (self.dim ** 0.5)  # 确保放缩  
        
#         # 应用注意力掩码  
#         if attn_mask is not None:  
#             QK = QK.masked_fill(attn_mask == 0, float('-inf'))  

#         attn_weights = F.softmax(QK, dim=-1)  
#         attn_weights = self.dropout_layer(attn_weights)  # 只对 attn_weights 应用 dropout  
        
#         # 计算输出  
#         out_tensor = torch.matmul(attn_weights, V)  # 输出应为 (batch_size, seq_length, out_features)  
        
#         return out_tensor, attn_weights  # 返回输出和注意力权重 

class LinearAttentionHead(nn.Module):  
    """  
    Linear attention, as proposed by the linformer paper  
    """  
    def __init__(self, dim, dropout,  causal_mask=None, full_attention=False):  
        super(LinearAttentionHead, self).__init__()  

        self.dim = dim  
        self.dropout_layer = nn.Dropout(dropout)  # 定义 dropout   
        self.full_attention = full_attention  
        self.causal_mask = causal_mask  
        
    def adjust_dim(self, tensor, target_size):  
        """ 调整tensor的第一维度以匹配目标大小 """  
        current_size = tensor.shape[0]  
        if current_size < target_size:  
            # 如果当前尺寸小于目标，使用填充  
            padding = torch.zeros(target_size - current_size, *tensor.shape[1:]).to(tensor.device)  
            adjusted_tensor = torch.cat((tensor, padding), dim=0)  
        elif current_size > target_size:  
            # 如果当前尺寸大于目标，进行切片  
            adjusted_tensor = tensor[:target_size]  
        else:  
            # 尺寸相同  
            adjusted_tensor = tensor  
        
        return adjusted_tensor
    def forward(self, Q, K, V, attn_mask=None, key_padding_mask=None):  
        K = K.transpose(1, 2)  # 转置 K 的尺寸  
        
        # 根据输入的维度动态创建 E 和 F  
        self.E = nn.Linear(K.shape[-1], K.shape[-2])  # K的线性投影层  
        self.F = nn.Linear(V.shape[-1], V.shape[-2])  # V的线性投影层  

        self.E = self.E.to(K.device)
        self.F = self.F.to(K.device)
        
        # 自动处理 K 和 V 的批量维度不匹配  
        if K.shape[0] != Q.shape[0]:  
            # 使用线性层进行维度匹配  
            K = self.adjust_dim(K, Q.shape[0])  
            V = self.adjust_dim(V, Q.shape[0])  
        # 如果 K 和 V 的形状不匹配，进行线性投影  
        if K.shape[-1] != self.E.in_features:  
            K = self.E(K)  # 线性投影 K  
            
        if V.shape[-1] != self.F.in_features:  
            V = self.F(V)  # 线性投影 V  
        
        try:  
            QK = torch.matmul(Q, K)  
        except RuntimeError as e:  
            print(f"Error during matrix multiplication: {e}")    
        
        QK = QK / (self.dim ** 0.5)  
        
        # 应用注意力掩码  
        if attn_mask is not None:  
            
            QK = QK.masked_fill(attn_mask == 0, float('-inf')) 
            
        # No support key_padding_mask
        # if key_padding_mask is not None:  
        #     # 将 key_padding_mask 的维度扩展到与 QK 相同  
        #     # key_padding_mask 应该是 (batch_size, seq_length)  
        #     # key_padding_mask = key_padding_mask.unsqueeze(1)  # (batch_size, 1, seq_length) 
        #     if key_padding_mask.shape[-1]!=QK.shape[0]:
        #         key_padding_mask = key_padding_mask.float() 
        #         key_padding_mask =  key_padding_mask.max(dim=1, keepdim=True).values
        #     print(f"QK: {QK.shape}") 
        #     print(f"key_padding_mask: {key_padding_mask.shape}")
        #     QK = QK.masked_fill(key_padding_mask == 0, float('-inf'))  

        attn_weights = F.softmax(QK, dim=-1)  
        attn_weights = self.dropout_layer(attn_weights)  # 只对 attn_weights 应用 dropout  

        # 计算输出  
        out_tensor = torch.matmul(attn_weights, V)  # 输出应为 (batch_size, seq_length, features)   
       
        return out_tensor, attn_weights  # 返回输出和注意力权重  

@ATTENTION.register_module()
class LinearAttentionhead(BaseModule):  
    def __init__(self,  
                 embed_dims,  
                 attn_drop=0.,  
                 proj_drop=0.,  
                 dropout_layer=dict(type='Dropout', drop_prob=0.),  
                 init_cfg=None,  
                 batch_first=False,
                 **kwargs):  
        super(LinearAttentionhead, self).__init__(init_cfg)  
        
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
            
        self.embed_dims = embed_dims  
        
        self.batch_first = batch_first  
        # E_proj = nn.Linear(embed_dims, embed_dims)  # 假设输入输出维度相同  
        # F_proj = nn.Linear(embed_dims, embed_dims)  # 假设输入输出维度相同
        # 直接创建 LinearAttentionHead，没有 num_heads 参数  
        self.attn = LinearAttentionHead(
            dim=embed_dims, 
            dropout=attn_drop,
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
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # 调用自定义的注意力头  
  
        out = self.attn(Q=query, K=key, V=value, attn_mask=attn_mask)[0]  
        
        
        if self.batch_first:
            out = out.transpose(0, 1)
            
        # print(f"Output shape: {out.shape}, type: {type(out)}")
        # print(f"identity shape: {identity.shape}, type: {type(identity)}")
        return identity + self.dropout_layer(self.proj_drop(out))

@ATTENTION.register_module()
class MultiEfficientAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(MultiEfficientAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

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
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))