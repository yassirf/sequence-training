import torch.nn as nn

from self_distribution_distillation_src.layers.ffn.fnn import (
    EncoderFNNLayer, DecoderFFNLayer
)
from self_distribution_distillation_src.layers.subbatch import (
    BatchLayer
)


class BatchEncoderFNNLayer(nn.Module):
    def __init__(self, cfg, num_ffns, input_dim, output_dim, dropout_p, q_noise, qn_block_size):
        super(BatchEncoderFNNLayer, self).__init__()

        # Create list of models
        model_list = [EncoderFNNLayer(cfg, input_dim, output_dim, dropout_p, q_noise, qn_block_size)
                      for _ in range(num_ffns)]

        # Create naive batch layer
        self.model = BatchLayer(
            cfg=cfg,
            model_list=nn.ModuleList(model_list)
        )

    def forward(self, x):
        return self.model(x)


class BatchDecoderFFNLayer(nn.Module):
    def __init__(self, cfg, num_ffns, input_dim, output_dim, dropout_p, q_noise, qn_block_size):
        super(BatchDecoderFFNLayer, self).__init__()

        # Create list of models
        model_list = [DecoderFFNLayer(cfg, input_dim, output_dim, dropout_p, q_noise, qn_block_size)
                      for _ in range(num_ffns)]

        # Create naive batch layer
        self.model = BatchLayer(
            cfg=cfg,
            model_list=nn.ModuleList(model_list)
        )

    def forward(self, x):
        return self.model(x)