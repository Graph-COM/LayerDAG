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

class MultiEmbedding(nn.Module):
    def __init__(self, num_x_n_cat, hidden_size):
        super().__init__()

        self.emb_list = nn.ModuleList([
            nn.Embedding(num_x_n_cat_i, hidden_size)
            for num_x_n_cat_i in num_x_n_cat.tolist()
        ])

    def forward(self, x_n_cat):
        if len(x_n_cat.shape) == 1:
            x_n_emb = self.emb_list[0](x_n_cat)
        else:
            x_n_emb = torch.cat([
                self.emb_list[i](x_n_cat[:, i]) for i in range(len(self.emb_list))
            ], dim=1)

        return x_n_emb

class BiMPNNEncoder(nn.Module):
    def __init__(self,
                 num_x_n_cat,
                 x_n_emb_size,
                 pe_emb_size,
                 hidden_size,
                 num_mpnn_layers,
                 pe=None,
                 y_emb_size=0,
                 pool=None):
        super().__init__()

        self.pe = pe
        if self.pe in ['relative_level', 'abs_level']:
            self.level_emb = SinusoidalPE(pe_emb_size)
        elif self.pe == 'relative_level_one_hot':
            self.level_emb = OneHotPE(pe_emb_size)

        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        self.y_emb = SinusoidalPE(y_emb_size)

        self.proj_input = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mpnn_layers = nn.ModuleList()
        for _ in range(num_mpnn_layers):
            self.mpnn_layers.append(BiMPNNLayer(hidden_size, hidden_size))

        self.project_output_n = nn.Sequential(
            nn.Linear((num_mpnn_layers + 1) * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.pool = pool
        if pool is not None:
            self.bn_g = nn.BatchNorm1d(hidden_size)

    def forward(self, A, x_n, abs_level, rel_level, y=None, A_n2g=None):
        A_T = A.T
        h_n = self.x_n_emb(x_n)

        if self.pe == 'abs_level':
            node_pe = self.level_emb(abs_level)

        if self.pe in ['relative_level', 'relative_level_one_hot']:
            node_pe = self.level_emb(rel_level)

        if self.pe is not None:
            h_n = torch.cat([h_n, node_pe], dim=-1)

        if y is not None:
            h_y = self.y_emb(y)
            h_n = torch.cat([h_n, h_y], dim=-1)

        h_n = self.proj_input(h_n)
        h_n_cat = [h_n]
        for layer in self.mpnn_layers:
            h_n = layer(A, A_T, h_n)
            h_n_cat.append(h_n)
        h_n = torch.cat(h_n_cat, dim=-1)
        h_n = self.project_output_n(h_n)

        if self.pool is None:
            return h_n
        elif self.pool == 'sum':
            h_g = A_n2g @ h_n
            return self.bn_g(h_g)
        elif self.pool == 'mean':
            h_g = A_n2g @ h_n
            h_g = h_g / A_n2g.sum(dim=1).unsqueeze(-1)
            return self.bn_g(h_g)

class GraphClassifier(nn.Module):
    def __init__(self,
                 graph_encoder,
                 emb_size,
                 num_classes):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, y=None):
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, y, A_n2g)
        pred_g = self.predictor(h_g)

        return pred_g

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
        node_count_encoder = BiMPNNEncoder(num_x_n_cat,
                                           hidden_size=hidden_size,
                                           **node_count_encoder_config).to(device)
