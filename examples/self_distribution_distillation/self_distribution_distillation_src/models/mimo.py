
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
        parser.add_argument('--naive-mimo', type=int, default=0)
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
            naive=args.naive_mimo,
        )


def mimo_get_attributes(args, heads, bias, naive):
    args.num_heads = heads
    args.bias = bias
    args.naive_mimo = naive


@register_model_architecture('mimo_transformer', 'mimo1_transformer')
def mimo1_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 1, False, False)


@register_model_architecture('mimo_transformer', 'mimo2_transformer')
def mimo2_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 2, False, False)


@register_model_architecture('mimo_transformer', 'mimo3_transformer')
def mimo3_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 3, False, False)


@register_model_architecture('mimo_transformer', 'mimo1_naive_transformer')
def mimo1_naive_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 1, True, True)


@register_model_architecture('mimo_transformer', 'mimo2_naive_transformer')
def mimo2_naive_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 2, True, True)


@register_model_architecture('mimo_transformer', 'mimo3_naive_transformer')
def mimo3_naive_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, 3, True, True)


