
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big
)
from self_distribution_distillation_src.modules.encoder import (
    BatchFNNTransformerEncoder
)
from self_distribution_distillation_src.modules.decoder import (
    BatchFNNTransformerDecoder
)


@register_model('batch_ffn_transformer')
class BatchFFNTransformerModel(TransformerModel):

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return BatchFNNTransformerEncoder(
            args = args,
            dictionary = src_dict,
            embed_tokens = embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return BatchFNNTransformerDecoder(
            args = args,
            dictionary = tgt_dict,
            embed_tokens = embed_tokens,
            no_encoder_attn = args.no_cross_attention,
            output_projection=None,
        )


@register_model_architecture('batch_ffn_transformer', 'batch1_ffn_transformer')
def batch1_ffn_transformer(args):
    base_architecture(args)


@register_model_architecture('batch_ffn_transformer', 'batch2_ffn_transformer')
def batch2_ffn_transformer(args):
    base_architecture(args)
    args.encoder_num_ffns = getattr(args, "encoder_num_ffns", 2)
    args.decoder_num_ffns = getattr(args, "decoder_num_ffns", 2)


@register_model_architecture('batch_ffn_transformer', 'batch3_ffn_transformer')
def batch3_ffn_transformer(args):
    base_architecture(args)
    args.encoder_num_ffns = getattr(args, "encoder_num_ffns", 3)
    args.decoder_num_ffns = getattr(args, "decoder_num_ffns", 3)

