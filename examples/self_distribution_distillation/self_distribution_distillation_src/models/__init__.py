
from .dirtransformer import SelfDirichletTransformerModel
from .dirtransformer import self_dirichlet_transformer, self_dirichlet_transformer_wmt_en_de_big
from .mimo import MimoTransformerModel
from .mimo import mimo1_naive_transformer, mimo2_naive_transformer, mimo3_naive_transformer
from .mimo import mimo1_naive_transformer_wmt_en_de_big, mimo2_naive_transformer_wmt_en_de_big, mimo3_naive_transformer_wmt_en_de_big
from .termination import TerminationTransformerModel
from .termination import termination_transformer, termination_transformer_wmt_en_de_big
from .termination import half_termination_transformer, half_termination_transformer_wmt_en_de_big
from .gausstransformer import SelfGaussianTransformerModel
from .gausstransformer import self_gaussian_transformer, self_gaussian_transformer_wmt_en_de_big
from .batch import *

# Combination networks
from .dirmimo import SelfMimoTransformerModel
from .dirmimo import (
    self_mimo1_naive_transformer,
    self_mimo1_naive_transformer_wmt_en_de_big,
    self_mimo2_naive_transformer,
    self_mimo2_naive_transformer_wmt_en_de_big,
    self_mimo3_naive_transformer,
    self_mimo3_naive_transformer_wmt_en_de_big,
)
from .gaussmimo import SelfGaussianMimoTransformerModel
from .gaussmimo import (
    self_gaussmimo1_naive_transformer,
    self_gaussmimo1_naive_transformer_wmt_en_de_big,
    self_gaussmimo2_naive_transformer,
    self_gaussmimo2_naive_transformer_wmt_en_de_big,
    self_gaussmimo3_naive_transformer,
    self_gaussmimo3_naive_transformer_wmt_en_de_big,
)


__all__ = [
    'SelfDirichletTransformerModel',
    'self_dirichlet_transformer',
    'self_dirichlet_transformer_wmt_en_de_big',
    'MimoTransformerModel',
    'mimo1_naive_transformer',
    'mimo2_naive_transformer',
    'mimo3_naive_transformer',
    'mimo1_naive_transformer_wmt_en_de_big',
    'mimo2_naive_transformer_wmt_en_de_big',
    'mimo3_naive_transformer_wmt_en_de_big',
    'TerminationTransformerModel',
    'termination_transformer',
    'termination_transformer_wmt_en_de_big',
    'half_termination_transformer',
    'half_termination_transformer_wmt_en_de_big',
    'SelfGaussianTransformerModel',
    'self_gaussian_transformer',
    'self_gaussian_transformer_wmt_en_de_big',
    'NaiveBatchFFNTransformerModel',
    'naive_batch1_ffn_transformer',
    'naive_batch2_ffn_transformer',
    'naive_batch3_ffn_transformer',
    'BatchFFNTransformerModel',
    'batch1_ffn_transformer',
    'batch2_ffn_transformer',
    'batch3_ffn_transformer',
]

__all__.extend([
    'SelfMimoTransformerModel',
    'self_mimo1_naive_transformer',
    'self_mimo1_naive_transformer_wmt_en_de_big',
    'self_mimo2_naive_transformer',
    'self_mimo2_naive_transformer_wmt_en_de_big',
    'self_mimo3_naive_transformer',
    'self_mimo3_naive_transformer_wmt_en_de_big',
])

__all__.extend([
    'SelfGaussianMimoTransformerModel',
    'self_gaussmimo1_naive_transformer',
    'self_gaussmimo1_naive_transformer_wmt_en_de_big',
    'self_gaussmimo2_naive_transformer',
    'self_gaussmimo2_naive_transformer_wmt_en_de_big',
    'self_gaussmimo3_naive_transformer',
    'self_gaussmimo3_naive_transformer_wmt_en_de_big',
])
