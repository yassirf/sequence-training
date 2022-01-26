import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from .attention import MultiMultiheadAttention
from fairseq.modules import TransformerEncoderLayer
from fairseq.models.transformer import TransformerConfig


class NaiveMimoTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super(NaiveMimoTransformerEncoderLayer, self).__init__(args)

    def build_self_attention(self, embed_dim, args):
        cfg = TransformerConfig.from_namespace(args)
        return MultiMultiheadAttention(
            args.mimo_num_heads,
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(
        self,
        x,
        encoder_padding_mask: List[Optional[Tensor]],
        attn_mask: List[Optional[Tensor]] = None,
    ):
        """
        Args:
            x (Tensor): input to List layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask List[(ByteTensor)]: binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask Tuple[(ByteTensor)]: binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        # Ensure the number of masks match the number of heads in the system
        if encoder_padding_mask is not None: assert len(encoder_padding_mask) == self.args.mimo_num_heads
        if attn_mask is not None:
            assert len(attn_mask) == self.args.mimo_num_heads

            # Update all masks within the list
            attn_mask = [am.masked_fill(
                am.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            ) for am in attn_mask]

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Perform attention across n different layers
        # Note that each layer has its own mask
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x