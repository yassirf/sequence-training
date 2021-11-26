#import importlib
#import os
#
#for file in os.listdir(os.path.dirname(__file__)):
#    if file.endswith('.py') and not file.startswith('_'):
#        task_name = file[:file.find('.py')]
#        importlib.import_module('examples.self_distribution_distillation.utils.' + task_name)


from .device import check_device
from .dirichlet import DirichletEstimation

__all__ = [
    'check_device',
    'DirichletEstimation'
]