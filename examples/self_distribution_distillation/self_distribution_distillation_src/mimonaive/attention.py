
import torch
import torch.nn as nn
import torch.Tensor as Tensor

from typing import List, Tuple, Union, Optional, Dict

from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import MultiheadAttention


@with_incremental_state
class MultiMultiheadAttention(nn.Module):
    def __init__(
        self,
        num,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        # Number of separate attention mechanisms
        self.num = num

        # Build attentions
        self.attns = nn.ModuleList([
            MultiheadAttention(
                embed_dim = embed_dim,
                num_heads = num_heads,
                kdim = kdim,
                vdim = vdim,
                dropout = dropout,
                bias = bias,
                add_bias_kv = add_bias_kv,
                add_zero_attn = add_zero_attn,
                self_attention = self_attention,
                encoder_decoder_attention = encoder_decoder_attention,
                q_noise = q_noise,
                qn_block_size = qn_block_size,
            ) for _ in range(num)
        ])

        self.reset_parameters()
        self.encoder_decoder_attention = encoder_decoder_attention
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        for attn in self.attns:
            attn.reset_parameters()

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: List[Optional[Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Union[Tensor, List[Tensor]]]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: List[Optional[Tensor]] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:

        if key_padding_mask is not None: assert len(key_padding_mask) == self.num
        if attn_mask is not None: assert len(attn_mask) == self.num

        # Get the output from each attention layer
        # Disabling the incremental state for efficient prototyping
        res = [
            layer(
                query = query,
                key = key,
                value = value,
                key_padding_mask = key_padding_mask[i] if key_padding_mask is not None else None,
                incremental_state = None,
                need_weights = need_weights,
                static_kv = static_kv,
                attn_mask = attn_mask[i] if attn_mask is not None else None,
                before_softmax = before_softmax,
                need_head_weights = need_head_weights,
            ) for i, layer in enumerate(self.attns)
        ]

        # Process the result in the desired format
        attn, attn_weights = sum(r[0] for r in res), [r[1] for r in res]

        # We assume that we never use the 'before_softmax' setting
        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Union[Tensor, List[Tensor]]]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break

                    # The selection could be based on tensors or lists of tensors
                    if isinstance(input_buffer_k, list):
                        input_buffer[k] = [q.index_select(0, new_order) for q in input_buffer_k]
                    else:
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)

            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Union[Tensor, List[Tensor]]]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Union[Tensor, List[Tensor]]]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Union[Tensor, List[Tensor]]]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
