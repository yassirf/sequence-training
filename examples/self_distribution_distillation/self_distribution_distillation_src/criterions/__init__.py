
from .self import LabelSmoothedCrossEntropyAndSelfKLCriterion
from .selfgaussian import LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion

__all__ = [
    'LabelSmoothedCrossEntropyAndSelfKLCriterion',
    'LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion',
]