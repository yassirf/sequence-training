
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    tiny_architecture,
    base_architecture,
    transformer_wmt_en_de_big
)
from self_distribution_distillation_src.modules.decoder import (
    SelfGaussianTransformerDecoder
)


@register_model('self_gaussian_transformer')
class SelfGaussianTransformerModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--uniform-gauss-a', type=float, default=0.0)
        parser.add_argument('--uniform-gauss-b', type=float, default=0.0)
        parser.add_argument('--num-passes', type=int, default=0)
        parser.add_argument('--bias', type=int, default=0)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SelfGaussianTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=args.no_cross_attention,
            output_projection=None,
            bias=args.bias
        )


def self_get_attributes(args):
    args.uniform_gauss_a = getattr(args, 'uniform_gauss_a', 0.1)
    args.uniform_gauss_b = getattr(args, 'uniform_gauss_b', 0.1)
    args.num_passes = getattr(args, 'num_passes', 5)
    args.bias = getattr(args, 'bias', 0)


@register_model_architecture('self_gaussian_transformer', 'self_gaussian_transformer_tiny')
def self_gaussian_transformer_tiny(args):
    tiny_architecture(args)
    self_get_attributes(args)


@register_model_architecture('self_gaussian_transformer', 'self_gaussian_transformer')
def self_gaussian_transformer(args):
    base_architecture(args)
    self_get_attributes(args)


@register_model_architecture('self_gaussian_transformer', 'self_gaussian_transformer_wmt_en_de_big')
def self_gaussian_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    self_get_attributes(args)