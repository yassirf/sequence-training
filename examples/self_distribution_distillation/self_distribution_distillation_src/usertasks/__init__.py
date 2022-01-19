#import importlib
#import os
#
#for file in os.listdir(os.path.dirname(__file__)):
#    if file.endswith('.py') and not file.startswith('_'):
#        task_name = file[:file.find('.py')]
#        importlib.import_module('examples.self_distribution_distillation.usertasks.' + task_name)


from .translation import TranslationUncertaintyTask

__all__ = [
    'TranslationUncertaintyTask',
]


#
#import importlib
#import os
#
#Automatically import any Python files in the current directory
#curr_dir = os.path.dirname(__file__)
#for file in os.listdir(curr_dir):
#    path = os.path.join(curr_dir, file)
#    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
#        mod_name = file[: file.find(".py")] if file.endswith(".py") else file
#        module = importlib.import_module(__name__ + "." + mod_name)
