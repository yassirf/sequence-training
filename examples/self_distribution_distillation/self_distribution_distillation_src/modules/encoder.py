
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import TransformerConfig, TransformerEncoder
from self_distribution_distillation_src.layers.transformer import (
    NaiveBatchFFNTransformerEncoderLayer,
    BatchFFNTransformerEncoderLayer,
)


class NaiveBatchFNNTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(NaiveBatchFNNTransformerEncoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
        )

    def build_encoder_layer(self, args):
        cfg = TransformerConfig.from_namespace(args)
        layer = NaiveBatchFFNTransformerEncoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class BatchFNNTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(BatchFNNTransformerEncoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
        )

    def build_encoder_layer(self, args):
        cfg = TransformerConfig.from_namespace(args)
        layer = BatchFFNTransformerEncoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
