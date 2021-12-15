
from fairseq.models.transformer import TransformerConfig
from .naivebatchffnlayer import (
    NaiveBatchFFNTransformerEncoderLayer,
    NaiveBatchFFNTransformerDecoderLayer
)
from self_distribution_distillation_src.layers.ffn import (
    BatchEncoderFNNLayer, BatchDecoderFFNLayer
)


class BatchFFNTransformerEncoderLayer(NaiveBatchFFNTransformerEncoderLayer):
    def __init__(self, args):
        super(BatchFFNTransformerEncoderLayer, self).__init__(args = args)

        # Get hierarchical config
        cfg = TransformerConfig.from_namespace(args)

        # Create a naive batch fnn
        self.ffn = BatchEncoderFNNLayer(
            cfg = cfg,
            num_ffns = cfg.encoder.num_ffns,
            input_dim = self.embed_dim,
            output_dim = cfg.encoder.ffn_embed_dim,
            dropout_p = self.activation_dropout_p,
            q_noise = self.quant_noise,
            qn_block_size = self.quant_noise_block_size,
        )


class BatchFFNTransformerDecoderLayer(NaiveBatchFFNTransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn = False, add_bias_kv = False, add_zero_attn = False):
        super(BatchFFNTransformerDecoderLayer, self).__init__(
            args = args,
            no_encoder_attn = no_encoder_attn,
            add_bias_kv = add_bias_kv,
            add_zero_attn = add_zero_attn,
        )

        # Get hierarchical config
        cfg = TransformerConfig.from_namespace(args)

        # Create a naive batch fnn
        self.ffn = BatchDecoderFFNLayer(
            cfg=cfg,
            num_ffns=cfg.decoder.num_ffns,
            input_dim=self.embed_dim,
            output_dim=cfg.decoder.ffn_embed_dim,
            dropout_p=self.activation_dropout_p,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )