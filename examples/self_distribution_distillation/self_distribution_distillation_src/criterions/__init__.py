
from .self import LabelSmoothedCrossEntropyAndSelfKLCriterion
from .selfgaussian import LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion
from .selfcombined import LabelSmoothedCrossEntropyAndSelfCombinedCriterion

__all__ = [
    'LabelSmoothedCrossEntropyAndSelfKLCriterion',
    'LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion',
    'LabelSmoothedCrossEntropyAndSelfCombinedCriterion',
]