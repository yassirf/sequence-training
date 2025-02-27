import numpy as np
import torch

from .misc import process_outputs_gaussian
from .estimators import (
    EnsembleGaussianCategoricals,
)


def compute_token_gaussian_uncertainties(args, outputs, extra):
    """
    Function which computes token-level measures of uncertainty for Gaussian Categorical model.
    :param args: specifies uncertainty estimation parameters
    :param outputs: List of Tensors of size [batch_size, seq_len, vocab] of Log Dirichlet Concentrations
    :return: Tensors of token level uncertainties of size [batch_size, seq_len]
    """
    # Get processed outputs in mimo cases
    outputs, extra = process_outputs_gaussian(outputs, extra)

    outputs = [(op, ex['student_predictions_scale']) for op, ex in zip(outputs, extra)]

    estimator = EnsembleGaussianCategoricals()
    returns = estimator(args, outputs)

    return returns['entropy_expected'].clamp_(min=0.0, max=None), \
           returns['expected_entropy'].clamp_(min=0.0, max=None), \
           returns['mutual_information'].clamp_(min=0.0, max=None)


def compute_sequence_gaussian_uncertainties(args, outputs, extra, output_ids, output_length, mask):
    """
    Function which computes sequence-level measures of uncertainty for Gaussian Categorical model.
    :param args: specifies uncertainty estimation parameters
    :param outputs: List of Tensors of size [batch_size, seq_len, vocab] of Logits
    :param output_ids: Tensor of size [batch_size, seq_len] of token ids
    :param output_length: Tensor of size [batch_size, seq_len] of masked token ids
    :param mask: Tensor of size [batch_size] of masked token ids
    :return: Tuple of tensor score, sentence log-probability and token log-probabilities
    """
    # Get processed outputs in mimo cases
    outputs, extra = process_outputs_gaussian(outputs, extra)

    # Get output tuples
    outputs = [(op, ex['student_predictions_scale']) for op, ex in zip(outputs, extra)]

    # Get samples
    outputs = EnsembleGaussianCategoricals.sample(args, outputs)

    # Compute the expectation
    expected = torch.stack(outputs, dim=2)

    # Normalise results (batch, seqlen, models, vocab)
    expected = torch.log_softmax(expected, dim=-1)

    # Expected results (batch, seqlen, vocab)
    expected = torch.logsumexp(expected, dim=2) - np.log(expected.size(2))

    # Now (batch, seqlen, 1)
    unsqueezed_ids = output_ids.unsqueeze(-1)

    # Now (batch, seqlen)
    token_log_probs = expected.gather(-1, unsqueezed_ids).squeeze(2)

    # Remove any uncertainties outside mask
    if mask.any(): token_log_probs.masked_fill_(mask, 0.0)

    # Now get sentence and averaged scores
    log_probs = token_log_probs.sum(dim=1)
    scores = -log_probs / output_length

    # Return score, sentence log-probability, token probabilities
    return scores, log_probs, token_log_probs

