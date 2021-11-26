
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
    SelfDirichletTransformerDecoder
)


@register_model('self_dirichlet_transformer')
class SelfDirichletTransformerModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--uniform-gauss-a', type=float, default=0.0)
        parser.add_argument('--uniform-gauss-b', type=float, default=0.0)
        parser.add_argument('--num-passes', type=int, default=0)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SelfDirichletTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=args.no_cross_attention,
            # output_projection=None
        )


def self_get_attributes(args):
    args.uniform_gauss_a = getattr(args, 'uniform_gauss_a', 0.1)
    args.uniform_gauss_b = getattr(args, 'uniform_gauss_b', 0.1)
    args.num_passes = getattr(args, 'num_passes', 5)


@register_model_architecture('self_dirichlet_transformer', 'self_dirichlet_transformer')
def self_dirichlet_transformer(args):
    base_architecture(args)
    self_get_attributes(args)


@register_model_architecture('self_dirichlet_transformer', 'self_dirichlet_transformer_wmt_en_de_big')
def self_dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
    self_get_attributes(args)