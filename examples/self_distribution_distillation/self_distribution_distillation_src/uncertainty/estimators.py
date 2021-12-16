import numpy as np
import torch
from torch.distributions import normal

from typing import List, Dict, Tuple


class BaseClass(object):
    def __init__(self):
        pass

    def __call__(self, args, outputs: List[torch.Tensor], store_in: Dict) -> Dict:
        """
        Computes uncertainties and samples possible outputs
        The return should be a dictionary containing a key 'outputs'
        """
        raise NotImplementedError()


class EnsembleCategoricals(BaseClass):
    def __init__(self):
        super(EnsembleCategoricals, self).__init__()

    @staticmethod
    def compute_log_confidence(log_probs):
        return log_probs.max(axis=-1).values

    @staticmethod
    def compute_entropy(log_probs):
        entropy = - log_probs * torch.exp(log_probs)
        return entropy.sum(-1)

    def compute_expected_entropy(self, log_probs):
        entropies = self.compute_entropy(log_probs)
        return entropies.mean(-1)

    def compute_entropy_expected(self, log_probs):
        return self.compute_entropy(log_probs)

    @torch.no_grad()
    def __call__(self, args, outputs: List[torch.Tensor]) -> Dict:
        """
        Computes all default uncertainty metrics
        """

        # Assert temperature parameter exists
        temperature = getattr(args, "ood_temperature")

        # Combine all outputs into a single tensor (batch, seqlen, models, vocab)
        outputs = torch.stack(outputs, dim=2)

        # Input dimension
        batch, seqlen, n, vocab = outputs.size()

        # Create the zero matrix
        zero = torch.zeros_like(outputs[:, :, 0, 0])

        # Temperature anneal
        outputs = outputs / temperature

        # Normalise results (batch, seqlen, models, vocab)
        outputs = torch.log_softmax(outputs, dim=-1)

        # Expected results (batch, seqlen, vocab)
        expected = torch.logsumexp(outputs, dim=2) - np.log(n)

        returns = {
            'log_confidence': -self.compute_log_confidence(expected),
            'entropy_expected': self.compute_entropy_expected(expected),
            'expected_entropy': self.compute_expected_entropy(outputs) if n > 1 else zero
        }
        returns['mutual_information'] = returns['entropy_expected'] - returns['expected_entropy'] if n > 1 else zero
        return returns


class EnsembleDirichlets(EnsembleCategoricals):
    def __init__(self):
        super(EnsembleDirichlets, self).__init__()

    def compute_expected_entropy(self, log_alphas):
        alphas = torch.exp(log_alphas)
        alpha0 = torch.sum(alphas, dim=-1)

        entropy = torch.digamma(alpha0 + 1)
        entropy -= torch.sum(alphas * torch.digamma(alphas + 1), dim=-1) / alpha0

        return entropy.mean(-1)

    @torch.no_grad()
    def __call__(self, args, outputs: List[torch.Tensor]) -> Dict:
        """
        Computes all default uncertainty metrics
        """

        # Combine all outputs into a single tensor (batch, seqlen, models, vocab)
        outputs = torch.stack(outputs, dim=2)

        # Input dimension
        batch, seqlen, n, vocab = outputs.size()

        # Normalise results (batch, seqlen, models, vocab)
        expected = torch.log_softmax(outputs, dim=-1)

        # Expected results (batch, seqlen, vocab)
        expected = torch.logsumexp(expected, dim=2) - np.log(n)

        returns = {
            'log_confidence': -self.compute_log_confidence(expected),
            'entropy_expected': self.compute_entropy_expected(expected),
            'expected_entropy': self.compute_expected_entropy(outputs)
        }
        returns['mutual_information'] = returns['entropy_expected'] - returns['expected_entropy']
        return returns


class EnsembleGaussianCategoricals(EnsembleCategoricals):
    def __init__(self):
        super(EnsembleGaussianCategoricals, self).__init__()

    @staticmethod
    def sample(args, outputs: List[Tuple[torch.Tensor]]) -> List[torch.Tensor]:

        # Number of samples to draw
        num_samples = getattr(args, "ood_num_samples")

        # Get gaussian distributions
        gaussians = [normal.Normal(*op) for op in outputs]

        # Get logit samples
        samples = [g.sample() for _ in range(num_samples) for g in gaussians]

        return samples

    @torch.no_grad()
    def __call__(self, args, outputs: List[Tuple[torch.Tensor]]) -> Dict:

        # Draw logit samples from gaussian model
        samples = self.sample(args, outputs)

        return super(EnsembleGaussianCategoricals, self).__call__(
            args, samples
        )
