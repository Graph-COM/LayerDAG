import torch
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

        if not isinstance(marginal_list, list):
            marginal_list = [marginal_list]

        self.num_classes_list = []
        self.I_list = nn.ParameterList([])
        self.m_list = nn.ParameterList([])

        for marginal_f in marginal_list:
            num_classes_f = len(marginal_f)
            self.num_classes_list.append(num_classes_f)
            self.I_list.append(nn.Parameter(
                torch.eye(num_classes_f), requires_grad=False))
            marginal_f = marginal_f.unsqueeze(0).expand(
                num_classes_f, -1).clone()
            self.m_list.append(nn.Parameter(marginal_f, requires_grad=False))

        self.T = T
        import ipdb
        ipdb.set_trace()
