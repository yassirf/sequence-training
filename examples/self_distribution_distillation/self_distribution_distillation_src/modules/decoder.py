import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from typing import Any, Dict, List, Optional

from self_distribution_distillation_src.modules.noise import MultiplicativeGaussianLayer
from fairseq import utils
from fairseq.models.transformer import TransformerDecoder
from fairseq.modules import AdaptiveSoftmax, BaseLayer


class SelfDirichletTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            bias=False
    ):
        super(SelfDirichletTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection
        )
        self.stochasticity = MultiplicativeGaussianLayer(args.uniform_gauss_a, args.uniform_gauss_b, use_gpu = True)
        self.num_passes = args.num_passes

        # Use bias in output projection
        self.bias = bool(bias)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=cfg.bias,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=cfg.bias
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        v, extra = super(SelfDirichletTransformerDecoder, self).forward(
            prev_output_tokens = prev_output_tokens,
            encoder_out = encoder_out,
            incremental_state = incremental_state,
            features_only = True,
            full_context_alignment = full_context_alignment,
            alignment_layer = alignment_layer,
            alignment_heads = alignment_heads,
            src_lengths = src_lengths,
            return_all_hiddens = return_all_hiddens
        )

        # In standard forward pass setting
        if features_only:
            return v, extra

        # Stochastic free last layer
        z = self.output_layer(v)

        # Do not perform subsequent code if in evaluation mode
        if not self.training or self.num_passes < 1:
            return z, extra

        # Stochastic last layer
        vs = v.unsqueeze(1).repeat(1, self.num_passes, 1, 1)
        zs = self.output_layer(self.stochasticity(vs))

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach()
        extra['student_predictions_dir'] = z

        # Normalise stochastic teacher predictions and train those
        z = torch.mean(zs, dim = 1)

        # Return prediction and extra
        return z, extra


class TerminationTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            bias=False,
    ):
        super(TerminationTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection
        )

        # Termination probability
        term_prob = args.termination_probability

        # Termination over the second half of the decoder
        self.half_size = (self.num_layers+1)//2 if args.half_termination else 0

        # Ensure probability mass valid
        assert term_prob * self.half_size <= 1.00

        # Create policy
        self.termination_policy = [0.0] * self.half_size
        for i in range(self.num_layers - self.half_size):
            self.termination_policy.append(term_prob/(1 - i * term_prob))

        # Use bias in output projection
        self.bias = bool(bias)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=cfg.bias,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=cfg.bias
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                    enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        termination_states: List[Optional[Tensor]] = []

        # termination network early
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # Early termination network — terminate after this layer
            early_termination = self.training and (torch.rand(1).item() < self.termination_policy[idx])

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer) or early_termination),
                need_head_weights=bool((idx == alignment_layer) or early_termination),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            # Store the termination state
            if idx >= self.half_size: termination_states.append(x)

            # Stop the model early
            if early_termination: break

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "termination_states": termination_states}

    def manual_forward_output(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x


class MimoTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            bias=False,
            num_heads=2,
    ):
        super(MimoTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection
        )

        # Use bias in output projection
        self.bias = bool(bias)

        # Number of heads in mimo model
        self.num_heads = num_heads

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
            print('y', self.adaptive_softmax)
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=cfg.bias,
            )
            self.output_projection.weight = self.embed_tokens.weight
            print('yy', self.output_projection)
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary) * cfg.num_heads, bias=cfg.bias
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
            print('yyy', self.output_projection)
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def reformat_output(self, x):
        # The input is of the form (batch * num-heads, seq, vocab * num-heads)
        bn, s, vn = x.size()

        # Get the effective batch and vocab size
        b, v = bn//self.num_heads, vn//self.num_heads

        # We need to review the input and choose the relevant ones
        x = x.view(self.num_heads, b, s, self.num_heads, v)

        # Now make a diagonal choice (ensemble, batch, num_classes) this is core to mimo
        x = torch.diagonal(x, offset=0, dim1=0, dim2=3).permute(3, 0, 1, 2)

        # Return the formatted prediction
        return x.reshape(-1, s, v)

    @staticmethod
    def ensemble(x):
        # The input is of the form (batch, seq, num, vocab)
        lps = torch.log_softmax(x, dim = -1)
        x = torch.logsumexp(lps, dim = -2) - np.log(lps.size(-2))
        return x, lps

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        v, extra = super(MimoTransformerDecoder, self).forward(
            prev_output_tokens = prev_output_tokens,
            encoder_out = encoder_out,
            incremental_state = incremental_state,
            features_only = True,
            full_context_alignment = full_context_alignment,
            alignment_layer = alignment_layer,
            alignment_heads = alignment_heads,
            src_lengths = src_lengths,
            return_all_hiddens = return_all_hiddens
        )

        # In standard forward pass setting
        if features_only:
            return v, extra

        # Get the output layer and reformat it
        z = self.output_layer(v)

        # Do not perform subsequent code if in evaluation mode
        if not self.training:

            # Review the input into separate heads
            z = z.view(z.size(0), z.size(1), self.num_heads, -1)

            # Ensemble the predictions (batch, seq, vocab)
            op, lps = self.ensemble(z)

            # Add the separate predictions to extra
            extra['teacher_predictions_lp'] = lps

            return op, extra

        # In training mode separate the different head predictions (batch, seq, vocab)
        z = self.reformat_output(z)

        return z, extra