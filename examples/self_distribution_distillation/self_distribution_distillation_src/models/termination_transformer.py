
import numpy
import torch
from typing import Optional

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big
)
from self_distribution_distillation_src.modules.termination_decoder import (
    TerminationTransformerDecoder
)


@register_model('termination_transformer')
class TerminationTransformerModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--termination-probability', type=float, default=0.0)
        parser.add_argument('--bias', type=int, default=0)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TerminationTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=args.no_cross_attention,
            output_projection=None,
            bias=args.bias
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        # In stochastic termination mode
        if self.training: return decoder_out

        # In evaluation model average all predictions
        _, extra = decoder_out

        # Get all output predictions
        v = [self.decoder.manual_forward_output(op) for op in extra['inner_states']]
        v = self.decoder.output_layer(torch.stack(v, dim=1))
        v = torch.log_softmax(v, dim = -1)

        # Ensemble the predictions
        x = torch.logsumexp(v, dim = 1) - numpy.log(v.size(1))

        # Save predictions
        extra['intermediate_predictions'] = v

        return x, extra


def termination_get_attributes(args):
    args.termination_probability = getattr(args, 'termination_probability', 0.0)
    args.bias = getattr(args, 'bias', 0)


@register_model_architecture('termination_transformer', 'termination_transformer')
def self_dirichlet_transformer(args):
    base_architecture(args)
    termination_get_attributes(args)


@register_model_architecture('termination_transformer', 'termination_transformer_wmt_en_de_big')
def self_dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    termination_get_attributes(args)