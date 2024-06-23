import torch.nn as nn

__all__ = [
    'DiscreteDiffusion',
]

class DiscreteDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
