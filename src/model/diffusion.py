import torch.nn as nn

__all__ = [
    'DiscreteDiffusion',
]

class DiscreteDiffusion(nn.Module):
    def __init__(self,
                 marginal_list,
                 T,
                 s=0.008):
        """
        Parameters
        ----------
        marginal_list : list of torch.Tensor
            marginal_list[f] is the marginal distribution of the f-th attribute
        s : float
            Constant in noise schedule
        """
        super().__init__()
