
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModelBase,
    base_architecture,
    transformer_wmt_en_de_big
)
from self_distribution_distillation_src.modules.encoder import (
    NaiveBatchFNNTransformerEncoder
)
from self_distribution_distillation_src.modules.decoder import (
    NaiveBatchFNNTransformerDecoder
)


@register_model('naive_batch_ffn_transformer')
class NaiveBatchFFNTransformerModel(TransformerModelBase):

    @classmethod
    def add_args(cls, parser):
        TransformerModelBase.add_args(parser)
        parser.add_argument('--num-heads', type=int, default=2)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return NaiveBatchFNNTransformerEncoder(
            cfg = cfg,
            dictionary = src_dict,
            embed_tokens = embed_tokens
        )

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return NaiveBatchFNNTransformerDecoder(
            cfg = cfg,
            dictionary = tgt_dict,
            embed_tokens = embed_tokens,
            no_encoder_attn = cfg.no_cross_attention,
            output_projection=None,
        )


@register_model_architecture('naive_batch_ffn_transformer', 'naive_batch1_ffn_transformer')
def batch1_transformer(args):
    base_architecture(args)


@register_model_architecture('naive_batch_ffn_transformer', 'naive_batch2_ffn_transformer')
def batch2_transformer(args):
    base_architecture(args)
    args.encoder_num_ffns = getattr(args, "encoder_num_ffns", 2)
    args.decoder_num_ffns = getattr(args, "decoder_num_ffns", 2)


@register_model_architecture('naive_batch_ffn_transformer', 'naive_batch3_ffn_transformer')
def batch3_transformer(args):
    base_architecture(args)
    args.encoder_num_ffns = getattr(args, "encoder_num_ffns", 3)
    args.decoder_num_ffns = getattr(args, "decoder_num_ffns", 3)

