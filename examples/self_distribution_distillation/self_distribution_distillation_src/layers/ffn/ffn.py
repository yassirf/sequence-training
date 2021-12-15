
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class EncoderFNNLayer(nn.Module):
    def __init__(self, cfg, input_dim, output_dim, dropout_p, q_noise, qn_block_size):
        super(EncoderFNNLayer, self).__init__()

        # Create the linear layers
        self.fc1 = quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)
        self.fc2 = quant_noise(nn.Linear(output_dim, input_dim), p=q_noise, block_size=qn_block_size)

        # Create activation function
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        # Create activation dropout
        self.activation_dropout_module = FairseqDropout(float(dropout_p), module_name=self.__class__.__name__)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class DecoderFFNLayer(nn.Module):
    def __init__(self, cfg, input_dim, output_dim, dropout_p, q_noise, qn_block_size):
        super(DecoderFFNLayer, self).__init__()

        # Create the linear layers
        self.fc1 = quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)
        self.fc2 = quant_noise(nn.Linear(output_dim, input_dim), p=q_noise, block_size=qn_block_size)

        # Create activation function
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        # Create activation dropout
        self.activation_dropout_module = FairseqDropout(float(dropout_p), module_name=self.__class__.__name__)

        # Create intermediate layernorm
        self.ffn_layernorm = LayerNorm(cfg.decoder.ffn_embed_dim) if utils.safe_getattr(cfg, 'scale_fc', False) else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        return x