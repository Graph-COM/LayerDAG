import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DiscreteDiffusion',
    'EdgeDiscreteDiffusion'
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
            marginal_list[d] is the marginal distribution of the d-th attribute
        s : float
            Constant in noise schedule
        """
        super().__init__()

        if not isinstance(marginal_list, list):
            marginal_list = [marginal_list]

        self.num_classes_list = []
        self.I_list = nn.ParameterList([])
        self.m_list = nn.ParameterList([])

        for marginal_d in marginal_list:
            num_classes_d = len(marginal_d)
            self.num_classes_list.append(num_classes_d)
            self.I_list.append(nn.Parameter(
                torch.eye(num_classes_d), requires_grad=False))
            marginal_d = marginal_d.unsqueeze(0).expand(
                num_classes_d, -1).clone()
            self.m_list.append(nn.Parameter(marginal_d, requires_grad=False))

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

    def get_Q(self, alpha, d):
        """
        Parameters
        ----------
        d : int
            Index for the attribute
        """
        return alpha * self.I_list[d] + (1 - alpha) * self.m_list[d]

    def apply_noise(self, z, t=None):
        if t is None:
            # Sample a timestep t uniformly from 0 to self.T.
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

class EdgeDiscreteDiffusion(nn.Module):
    def __init__(self,
                 avg_in_deg,
                 T,
                 s=0.008):
        super().__init__()

        self.avg_in_deg = avg_in_deg

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

    def apply_noise(self, z, t=None):
        """
        Parameters
        ----------
        z : torch.Tensor of shape (A, B)
            Adjacency matrix.
            A is the number of candidate destination nodes.
            B is the number of candidate source nodes.

        Returns
        -------
        z_t : torch.Tensor of shape (A * B)
        """
        if t is None:
            # Sample a timestep t uniformly from 0 to self.T.
            # Note that the notation is slightly inconsistent with the paper.
            # t=0 corresponds to t=1 in the paper, where corruption has already taken place.
            t = torch.randint(low=0, high=self.T + 1, size=(1,))

        # TODO: Better doc
        alpha_bar_t = self.alpha_bars[t.item()]
        # Marginal probability for an edge to exist.
        mean_in_deg = min(self.avg_in_deg, z.shape[1])
        m_z_t = torch.ones(z.shape) * (mean_in_deg / z.shape[1])
        prob_z_t = alpha_bar_t * z + (1 - alpha_bar_t) * m_z_t
        z_t = torch.bernoulli(prob_z_t)

        # Make sure each node has at least one edge.
        isolated_mask = (z_t.sum(dim=1) == 0).bool()
        if isolated_mask.any():
            z_t[isolated_mask, prob_z_t[isolated_mask].argmax(dim=1)] = 1

        z_t = z_t.reshape(-1)

        return t, z_t

    def get_Qs(self,
               alpha_t,
               alpha_bar_s,
               alpha_bar_t,
               marginal):
        M = torch.zeros(2)
        M = torch.tensor([
            1 - marginal, marginal
        ])
        M = M.unsqueeze(0).expand(2, -1)
        I = torch.eye(2)

        Q_t = alpha_t * I + (1 - alpha_t) * M
        Q_bar_s = alpha_bar_s * I + (1 - alpha_bar_s) * M
        Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * M

        return Q_t, Q_bar_s, Q_bar_t
