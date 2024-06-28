import torch.nn as nn

__all__ = [
    'LayerDAG'
]

class LayerDAG(nn.Module):
    def __init__(self):
        super().__init__()
