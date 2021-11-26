import torch
from typing import Any, Dict, List, Optional
from torch import Tensor

from self_distribution_distillation_src.modules.noise import MultiplicativeGaussianLayer
from fairseq.models.transformer import TransformerDecoder


class SelfDirichletTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super().__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            # output_projection = output_projection
        )
        self.stochasticity = MultiplicativeGaussianLayer(args.uniform_gauss_a, args.uniform_gauss_b, use_gpu = True)
        self.num_passes = args.num_passes

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        v, extra = super(SelfDirichletTransformerDecoder, self).forward(
            prev_output_tokens = prev_output_tokens,
            encoder_out = encoder_out,
            incremental_state = incremental_state,
            features_only = True,
            full_context_alignment = full_context_alignment,
            alignment_layer = alignment_layer,
            alignment_heads = alignment_heads,
            src_lengths = src_lengths,
            return_all_hiddens = return_all_hiddens
        )

        # In standard forward pass setting
        if features_only:
            return v, extra

        # Stochastic free last layer
        z = self.output_layer(v)

        # Do not perform subsequent code if in evaluation mode
        if not self.training or self.num_passes < 1:
            return z, extra

        # Stochastic last layer
        vs = v.unsqueeze(1).repeat(1, self.num_passes, 1, 1)
        zs = self.output_layer(self.stochasticity(vs))

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach()
        extra['student_predictions_dir'] = z

        # Normalise stochastic teacher predictions and train those
        z = torch.mean(zs, dim = 1)

        # Return prediction and extra
        return z, extra