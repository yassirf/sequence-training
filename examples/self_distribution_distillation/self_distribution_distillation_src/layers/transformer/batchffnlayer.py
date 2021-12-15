
from .naivebatchffnlayer import (
    NaiveBatchFFNTransformerEncoderLayerBase,
    NaiveBatchFFNTransformerDecoderLayerBase
)
from self_distribution_distillation_src.layers.ffn.batchffn import (
    BatchEncoderFNNLayer, BatchDecoderFFNLayer
)


class BatchFFNTransformerEncoderLayerBase(NaiveBatchFFNTransformerEncoderLayerBase):
    def __init__(self, cfg):
        super(BatchFFNTransformerEncoderLayerBase, self).__init__(cfg = cfg)

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


class BatchFFNTransformerDecoderLayerBase(NaiveBatchFFNTransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn = False, add_bias_kv = False, add_zero_attn = False):
        super(BatchFFNTransformerDecoderLayerBase, self).__init__(
            cfg = cfg,
            no_encoder_attn = no_encoder_attn,
            add_bias_kv = add_bias_kv,
            add_zero_attn = add_zero_attn,
        )

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