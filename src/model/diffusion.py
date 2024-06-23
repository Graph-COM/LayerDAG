import numpy as np
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
        # Cosine schedule as proposed in
        # https://arxiv.org/abs/2102.09672
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        # Schedule for \bar{alpha}_t = alpha_1 * ... * alpha_t
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        # Make the largest value 1.
        alpha_bars = alpha_bars / alpha_bars[0]
        alphas = alpha_bars[1:] / alpha_bars[:-1]

        self.betas = torch.from_numpy(1 - alphas).float()
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)

    def get_Q(self, alpha, f):
        """
        Parameters
        ----------
        f : int
            Index for the attribute
        """
        return alpha * self.I_list[f] + (1 - alpha) * self.m_list[f]

    def apply_noise(self, z, t=None):
        if t is None:
            # Sample a timestep t uniformly.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T + 1, size=(1,))

        alpha_bar_t = self.alpha_bars[t.item()]

        if z.ndim == 1:
            z = z.unsqueeze(-1)

        _, D = z.shape
        z_t_list = []
        for d in range(D):
            Q_bar_t_d = self.get_Q(alpha_bar_t, d)
            z_one_hot_d = F.one_hot(z[:, d], num_classes=self.num_classes_list[d]).float()
            prob_z_t_d = z_one_hot_d @ Q_bar_t_d
            z_t_d = prob_z_t_d.multinomial(1).squeeze(-1)
            z_t_list.append(z_t_d)

        if D == 1:
            z_t = z_t_list[0]
        else:
            z_t = torch.stack(z_t_list, dim=1)

        return t, z_t
