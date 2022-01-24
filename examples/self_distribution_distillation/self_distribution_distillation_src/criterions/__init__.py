
from .self import LabelSmoothedCrossEntropyAndSelfKLCriterion
from .selfgaussian import LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion

from .distillation import KLDivergenceCriterion
from .distillationanddir import KLDivergenceAndDirCriterion
from .distillationandgauss import KLDivergenceAndGaussCriterion
from .distillationandlaplace import KLDivergenceAndLaplaceCriterion
from .distillationandgumbel import KLDivergenceAndGumbelCriterion

from .distributiondir import DirKLDivergenceAndDirCriterion
from .distributiongauss import GaussKLDivergenceAndGaussCriterion

__all__ = [
    'LabelSmoothedCrossEntropyAndSelfKLCriterion',
    'LabelSmoothedCrossEntropyAndSelfGaussianNLLCriterion',
    #
    'KLDivergenceCriterion',
    'KLDivergenceAndDirCriterion',
    'KLDivergenceAndGaussCriterion',
    'KLDivergenceAndLaplaceCriterion',
    'KLDivergenceAndGumbelCriterion',
    #
    'DirKLDivergenceAndDirCriterion',
    'GaussKLDivergenceAndGaussCriterion',
]