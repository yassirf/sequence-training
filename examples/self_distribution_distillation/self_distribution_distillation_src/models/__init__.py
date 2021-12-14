#import importlib
#import os
#
#for file in os.listdir(os.path.dirname(__file__)):
#    if file.endswith('.py') and not file.startswith('_'):
#        task_name = file[:file.find('.py')]
#        importlib.import_module('examples.self_distribution_distillation.models.' + task_name)

#import importlib
#import os
#
# Automatically import any Python files in the current directory
#curr_dir = os.path.dirname(__file__)
#for file in os.listdir(curr_dir):
#    path = os.path.join(curr_dir, file)
#    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
#        mod_name = file[: file.find(".py")] if file.endswith(".py") else file
#        module = importlib.import_module(__name__ + "." + mod_name)


from .dirtransformer import SelfDirichletTransformerModel
from .dirtransformer import self_dirichlet_transformer, self_dirichlet_transformer_wmt_en_de_big
from .mimo import MimoTransformerModel
from .mimo import mimo1_naive_transformer, mimo2_naive_transformer, mimo3_naive_transformer
from .termination import TerminationTransformerModel
from .termination import termination_transformer, termination_transformer_wmt_en_de_big
from .termination import half_termination_transformer, half_termination_transformer_wmt_en_de_big
from .gausstransformer import SelfGaussianTransformerModel
from .gausstransformer import self_gaussian_transformer, self_gaussian_transformer_wmt_en_de_big

__all__ = [
    'SelfDirichletTransformerModel',
    'self_dirichlet_transformer',
    'self_dirichlet_transformer_wmt_en_de_big',
    'MimoTransformerModel',
    'mimo1_naive_transformer',
    'mimo2_naive_transformer',
    'mimo3_naive_transformer',
    'TerminationTransformerModel',
    'termination_transformer',
    'termination_transformer_wmt_en_de_big',
    'half_termination_transformer',
    'half_termination_transformer_wmt_en_de_big',
    'SelfGaussianTransformerModel',
    'self_gaussian_transformer',
    'self_gaussian_transformer_wmt_en_de_big',
]