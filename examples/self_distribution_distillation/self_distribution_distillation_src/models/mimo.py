from fairseq import utils
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
from self_distribution_distillation_src.modules.embedding import MimoEmbedding
from self_distribution_distillation_src.modules.decoder import MimoTransformerDecoder


@register_model('mimo_transformer')
class MimoTransformerModel(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--num-heads', type=int, default = 2)
        parser.add_argument('--padding', type=int, default = 1)
        parser.add_argument('--bias', type=int, default = 0)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)

        # Make padding a learnable embedding
        padding_idx = dictionary.pad() if args.padding else None

        emb = MimoEmbedding(num_embeddings, embed_dim, args.num_heads, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

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


def mimo_get_attributes(args, heads, bias):
    args.num_heads = heads
    args.bias = bias


@register_model_architecture('mimo_transformer', 'mimo1_transformer')
def mimo1_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, heads = 1, bias = True)


@register_model_architecture('mimo_transformer', 'mimo2_transformer')
def mimo2_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, heads = 2, bias = True)


@register_model_architecture('mimo_transformer', 'mimo3_transformer')
def mimo3_transformer(args):
    base_architecture(args)
    mimo_get_attributes(args, heads = 3, bias = True)


@register_model_architecture('mimo_transformer', 'mimo1_transformer_wmt_en_de_big')
def mimo1_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo_get_attributes(args, heads = 1, bias = True)


@register_model_architecture('mimo_transformer', 'mimo2_transformer_wmt_en_de_big')
def mimo2_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo_get_attributes(args, heads = 2, bias = True)


@register_model_architecture('mimo_transformer', 'mimo3_transformer_wmt_en_de_big')
def mimo3_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    mimo_get_attributes(args, heads = 3, bias=True)

