import torch


def check_device(use_gpu):
    """
    Return device, either cpu or gpu
    """
    available_gpu = use_gpu and torch.cuda.is_available()
    return torch.device('cuda') if available_gpu else torch.device('cpu')
