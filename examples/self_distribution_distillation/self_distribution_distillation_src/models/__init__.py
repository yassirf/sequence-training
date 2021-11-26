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


from .transformer import SelfDirichletTransformerModel
from .transformer import self_dirichlet_transformer, self_dirichlet_transformer_wmt_en_de_big

__all__ = [
    'SelfDirichletTransformerModel',
    'self_dirichlet_transformer',
    'self_dirichlet_transformer_wmt_en_de_big',
]