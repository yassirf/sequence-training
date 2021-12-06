
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big
)
from self_distribution_distillation_src.modules.decoder import (
    MimoTransformerDecoder
)


@register_model('mimo_transformer')
class MimoTransformerModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--num-heads', type=int, default=2)
        parser.add_argument('--bias', type=int, default=0)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MimoTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=args.no_cross_attention,
            output_projection=None,
            bias=args.bias,
            num_heads=args.num_heads,
        )


def mimo1_get_attributes(args):
    args.num_heads = 1
    args.bias = 1


def mimo2_get_attributes(args):
    args.num_heads = 2
    args.bias = 1


def mimo3_get_attributes(args):
    args.num_heads = 3
    args.bias = 1


@register_model_architecture('mimo_transformer', 'mimo1_transformer')
def self_dirichlet_transformer(args):
    base_architecture(args)
    mimo1_get_attributes(args)


@register_model_architecture('mimo_transformer', 'mimo1_transformer_wmt_en_de_big')
def self_dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo1_get_attributes(args)


@register_model_architecture('mimo_transformer', 'mimo2_transformer')
def self_dirichlet_transformer(args):
    base_architecture(args)
    mimo2_get_attributes(args)


@register_model_architecture('mimo_transformer', 'mimo2_transformer_wmt_en_de_big')
def self_dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo2_get_attributes(args)


@register_model_architecture('mimo_transformer', 'mimo3_transformer')
def self_dirichlet_transformer(args):
    base_architecture(args)
    mimo3_get_attributes(args)


@register_model_architecture('mimo_transformer', 'mimo3_transformer_wmt_en_de_big')
def self_dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo3_get_attributes(args)