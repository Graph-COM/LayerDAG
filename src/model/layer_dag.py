import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LayerDAG'
]

class SinusoidalPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()

        self.pe_size = pe_size
        if pe_size > 0:
            self.div_term = torch.exp(torch.arange(0, pe_size, 2) *
                                      (-math.log(10000.0) / pe_size))
            self.div_term = nn.Parameter(self.div_term, requires_grad=False)

    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0).to(position.device)

        return torch.cat([
            torch.sin(position * self.div_term),
            torch.cos(position * self.div_term)
        ], dim=-1)

class BiMPNNLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.W = nn.Linear(in_size, out_size)
        self.W_trans = nn.Linear(in_size, out_size)
        self.W_self = nn.Linear(in_size, out_size)

    def forward(self, A, A_T, h_n):
        if A.nnz == 0:
            h_n_out = self.W_self(h_n)
        else:
            h_n_out = A @ self.W(h_n) + A_T @ self.W_trans(h_n) +\
                self.W_self(h_n)
        return F.gelu(h_n_out)

class OneHotPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()

        self.pe_size = pe_size

    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0).to(position.device)

        return F.one_hot(position.clamp(max=self.pe_size - 1).long().squeeze(-1),
                         num_classes=self.pe_size)

class LayerDAG(nn.Module):
    def __init__(self,
                 device,
                 num_x_n_cat,
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 max_level=None):
        """
        Parameters
        ----------
        num_x_n_cat :
            Case1: int
            Case2: torch.LongTensor of shape (num_feats)
        """
        super().__init__()

        if isinstance(num_x_n_cat, int):
            num_x_n_cat = torch.LongTensor([num_x_n_cat])

        self.dummy_x_n = num_x_n_cat - 1
        hidden_size = len(num_x_n_cat) * node_count_encoder_config['x_n_emb_size'] +\
            node_count_encoder_config['pe_emb_size'] +\
            node_count_encoder_config['y_emb_size']
