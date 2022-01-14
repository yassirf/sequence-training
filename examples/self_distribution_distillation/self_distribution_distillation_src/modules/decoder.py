import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from typing import Any, Dict, List, Optional

from self_distribution_distillation_src.modules.noise import MultiplicativeGaussianLayer
from fairseq import utils
from fairseq.modules import AdaptiveSoftmax, BaseLayer
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import TransformerConfig, TransformerDecoder
from self_distribution_distillation_src.layers.transformer import (
    NaiveBatchFFNTransformerDecoderLayer,
    BatchFFNTransformerDecoderLayer,
)


class NaiveBatchFNNTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super(NaiveBatchFNNTransformerDecoder, self).__init__(
            args=args,
            dictionary=dictionary,
            embed_tokens=embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        cfg = TransformerConfig.from_namespace(args)
        layer = NaiveBatchFFNTransformerDecoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class BatchFNNTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super(BatchFNNTransformerDecoder, self).__init__(
            args=args,
            dictionary=dictionary,
            embed_tokens=embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        cfg = TransformerConfig.from_namespace(args)
        layer = BatchFFNTransformerDecoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class Concatenator(nn.Module):
    def __init__(self, modulelist: nn.ModuleList):
        super(Concatenator, self).__init__()
        self.modulelist = modulelist

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.modulelist], dim = -1)


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

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=args.bias,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=args.bias
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = args.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder.layers) // (num_base_layers + 1),
                BaseLayer(args),
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
        zs = v.unsqueeze(1).repeat(1, self.num_passes, 1, 1)
        zs = self.output_layer(self.stochasticity(zs))

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach()
        extra['student_predictions_dir'] = z

        # Normalise stochastic teacher predictions and train those
        z = torch.mean(zs, dim = 1)

        # Return prediction and extra
        return z, extra


class SelfGaussianTransformerDecoder(SelfDirichletTransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            bias=False
    ):
        super(SelfGaussianTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection,
            bias = bias,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        
        # Create an additional scaling factor
        self.log_scale = nn.Linear(
            self.output_embed_dim,
            len(dictionary),
            bias=args.bias
        )

        super(SelfGaussianTransformerDecoder, self).build_output_projection(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
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
        s = self.log_scale(v)
        s = torch.exp(s)

        # Student scale predictions
        extra['student_predictions_scale'] = s

        # Do not perform subsequent code if in evaluation mode
        if not self.training or self.num_passes < 1:
            return z, extra

        # Stochastic last layer
        zs = v.unsqueeze(1).repeat(1, self.num_passes, 1, 1)
        zs = self.output_layer(self.stochasticity(zs))

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach()
        extra['student_predictions_mean'] = z

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

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=args.bias,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=args.bias
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = args.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder.layers) // (num_base_layers + 1),
                BaseLayer(args),
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

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary) * args.num_heads,
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.ModuleList([
                nn.Linear(self.output_embed_dim, len(dictionary), bias=args.bias) for _ in range(args.num_heads)
            ])
            for i in range(args.num_heads):
                self.output_projection[i].weight = self.embed_tokens.embs[i].weight
        else:
            self.output_projection = nn.ModuleList([
                nn.Linear(self.output_embed_dim, len(dictionary), bias=args.bias) for _ in range(args.num_heads)
            ])
            for opp in self.output_projection:
                nn.init.normal_(opp.weight, mean=0, std=self.output_embed_dim ** -0.5)

        num_base_layers = args.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder.layers) // (num_base_layers + 1),
                BaseLayer(args),
            )

    @staticmethod
    def ensemble(x, dim = -2):
        lps = torch.log_softmax(x, dim = -1)
        x = torch.logsumexp(lps, dim = dim) - np.log(lps.size(dim))
        return x, lps

    def output_layer(self, features):
        """
        Project features to the vocabulary size.
        """
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return [opp(features) for opp in self.output_projection]
        else:
            return features

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

        # Get the output layer and reformat it list(batch, len, vocab)
        z: List[Tensor] = self.output_layer(v)

        # In inference mode the predictions have to be treated differently
        if not self.training:

            # Review the input into separate heads (batch, num, len, vocab)
            z = torch.stack(z, dim = 1)

            # Ensemble the predictions (batch, len, vocab)
            op, lps = self.ensemble(z, dim = 1)

            # Add the separate predictions to extra (batch, models, len, vocab)
            extra['teacher_predictions_lp'] = lps

            return op, extra

        # In training mode separate the different predictions are retrieved by stacking
        z = torch.stack(z, dim = 0)

        return z, extra


class SelfMimoTransformerDecoder(MimoTransformerDecoder):
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
        super(SelfMimoTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection,
            bias = bias,
            num_heads = num_heads,
        )

        # Stochasticity generation
        self.stochasticity = MultiplicativeGaussianLayer(args.uniform_gauss_a, args.uniform_gauss_b, use_gpu = True)
        self.num_passes = args.num_passes

    @staticmethod
    def ensemble(x):
        # The input is of the form (batch, seq, num, vocab)
        lps = torch.log_softmax(x, dim=-1)
        lp = torch.logsumexp(lps, dim=-2) - np.log(lps.size(-2))
        return lp, lps, x

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

        # This model only works when
        assert self.num_passes > 1

        # Get the output layer and reformat it
        z = self.output_layer(v)

        # Do not perform subsequent code if in evaluation mode
        if not self.training:
            # Review the input into separate heads
            z = z.view(z.size(0), z.size(1), self.num_heads, -1)

            # Ensemble the predictions (batch, seq, vocab)
            op, lps, la = self.ensemble(z)

            # Add the separate predictions to extra (batch, models, len, vocab)
            extra['teacher_predictions_lp'] = la.permute(0, 2, 1, 3)

            return op, extra

        # Stochastic last layer
        zs = v.unsqueeze(1).repeat(1, self.num_passes, 1, 1)
        zs = self.output_layer(self.stochasticity(zs))

        # Reformat predictions for the loss
        batch, seqlen, nvocab = z.size()

        # Get number of stochastic passes and heads
        nump, numh = self.num_passes, self.num_heads

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach().view(batch, nump, seqlen, numh, nvocab//numh)
        extra['student_predictions_dir'] = z.view(batch, seqlen, numh, nvocab//numh)

        # In training mode separate the different head predictions (batch, num, seq, vocab)
        fmtz = self.reformat_output(zs.mean(dim = 1))

        return fmtz, extra


class SelfGaussianMimoTransformerDecoder(SelfMimoTransformerDecoder):
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
        super(SelfGaussianMimoTransformerDecoder, self).__init__(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection,
            bias = bias,
            num_heads = num_heads,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        self.log_scale = nn.Linear(
            self.output_embed_dim, len(dictionary) * args.num_heads, bias=args.bias
        )
        super(SelfGaussianMimoTransformerDecoder, self).build_output_projection(
            args = args,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
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

        # This model only works when
        assert self.num_passes > 1

        # Get the output layer and reformat it
        z = self.output_layer(v)
        s = self.log_scale(v)
        s = torch.exp(s)

        # Reformat predictions for the loss
        batch, seqlen, nvocab = z.size()

        # Get number of stochastic passes and heads
        nump, numh = self.num_passes, self.num_heads

        # Student scale predictions reformatted to mimo form
        extra['student_predictions_scale'] = s.view(batch, seqlen, numh, nvocab//numh)

        # Do not perform subsequent code if in evaluation mode
        if not self.training:
            # Review the input into separate heads
            z = z.view(batch, seqlen, numh, nvocab//numh)

            # Ensemble the predictions (batch, seq, vocab)
            op, lps, la = self.ensemble(z)

            # Add the separate predictions to extra (batch, models, len, vocab)
            extra['teacher_predictions_lp'] = la.permute(0, 2, 1, 3)
            extra['student_predictions_scale'] = extra['student_predictions_scale'].permute(0, 2, 1, 3)

            return op, extra

        # Stochastic last layer
        zs = v.unsqueeze(1).repeat(1, nump, 1, 1)
        zs = self.output_layer(self.stochasticity(zs))

        # Teacher branch prediction has shape (batch, models, len, vocab)
        extra['teacher_predictions_lp'] = zs.clone().detach().view(batch, nump, seqlen, numh, nvocab//numh)
        extra['student_predictions_mean'] = z.view(batch, seqlen, numh, nvocab//numh)

        # In training mode separate the different head predictions (batch, num, seq, vocab)
        fmtz = self.reformat_output(zs.mean(dim = 1))

        return fmtz, extra
