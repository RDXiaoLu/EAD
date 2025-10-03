import torch
from torch import nn
from torch.nn import functional as F
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv import ConfigDict, deprecated_api_warning
from mmcv.runner.base_module import BaseModule



# class EfficientAttention(nn.Module):
    
#     def __init__(self, in_channels, key_channels, head_count, value_channels):
#         super(EfficientAttention, self).__init__()
#         self.in_channels = in_channels
#         self.key_channels = key_channels
#         self.head_count = head_count
#         self.value_channels = value_channels

#         self.keys = nn.Conv2d(in_channels, key_channels, 1)
#         self.queries = nn.Conv2d(in_channels, key_channels, 1)
#         self.values = nn.Conv2d(in_channels, value_channels, 1)
#         self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

#     def forward(self, query, key=None, value=None, **kwargs):
        
#         input_ = query
#         print("Input shape:", input_.shape)
#         # n, _, h, w = input_.size()
#         n, _, l = input_.size()  
#         h = int(l ** 0.5)  
#         w = h  
#         keys = self.keys(input_).reshape((n, self.key_channels, h * w))
#         queries = self.queries(input_).reshape(n, self.key_channels, h * w)
#         values = self.values(input_).reshape((n, self.value_channels, h * w))
#         head_key_channels = self.key_channels // self.head_count
#         head_value_channels = self.value_channels // self.head_count
        
#         attended_values = []
#         for i in range(self.head_count):
#             key = f.softmax(keys[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=2)
#             query = f.softmax(queries[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=1)
#             value = values[
#                 :,
#                 i * head_value_channels: (i + 1) * head_value_channels,
#                 :
#             ]
#             context = key @ value.transpose(1, 2)
#             attended_value = (
#                 context.transpose(1, 2) @ query
#             ).reshape(n, head_value_channels, h, w)
#             attended_values.append(attended_value)

#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.reprojection(aggregated_values)
#         attention = reprojected_value + input_

#         return attention

class EfficientAttention(nn.Module):  
    def __init__(self, in_channels, key_channels, head_count, value_channels):  
        super(EfficientAttention, self).__init__()  
        self.in_channels = in_channels  
        self.key_channels = key_channels  
        self.head_count = head_count  
        self.value_channels = value_channels  

        # 使用一维卷积，因为输入可能是三维的  
        self.keys = nn.Conv1d(in_channels, key_channels, kernel_size=1)  
        self.queries = nn.Conv1d(in_channels, key_channels, kernel_size=1)  
        self.values = nn.Conv1d(in_channels, value_channels, kernel_size=1)  
        self.reprojection = nn.Conv1d(value_channels, in_channels, kernel_size=1)  

    def forward(self, query, key=None, value=None, **kwargs):   
        
        input_ = query
        n, _, w = input_.size()  # 期望是 (batch_size, channels, length)  
        
        keys = self.keys(input_).reshape((n, self.key_channels, w))  
        queries = self.queries(input_).reshape(n, self.key_channels, w)  
        values = self.values(input_).reshape((n, self.value_channels, w))  
        
        head_key_channels = self.key_channels // self.head_count  
        head_value_channels = self.value_channels // self.head_count  
        
        attended_values = []  
        for i in range(self.head_count):  
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)  
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)  
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]  
            
            context = key @ value.transpose(1, 2)  # 注意力上下文  
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, w)  
            attended_values.append(attended_value)  

        aggregated_values = torch.cat(attended_values, dim=1)  
        reprojected_value = self.reprojection(aggregated_values)  
        attention = reprojected_value + input_  

        return attention      

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

        # self.attn = # 用 EfficientAttention 替换 nn.MultiheadAttention  
        self.attn = EfficientAttention(  
            in_channels=embed_dims,  # 为了与原 embed_dims 对应  
            key_channels=embed_dims,  # 根据需要调整  
            head_count=num_heads,  # 对应 num_heads  
            value_channels=embed_dims  # 一般与 embed_dims 相同，此处可以根据需求调整  
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