import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

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

class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 dropout):
        super().__init__()

        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_qk = nn.Linear(hidden_size, hidden_size * 2)

        self._reset_parameters()

        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.proj_new = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_qk.weight)

    def attn(self, q, k, v, num_query_cumsum):
        """
        Parameters
        ----------
        q : torch.Tensor of shape (N, F)
            Query matrix for node representations.
        k : torch.Tensor of shape (N, F)
            Key matrix for node representations.
        v : torch.Tensor of shape (N, F)
            Value matrix for node representations.
        num_query_cumsum : torch.Tensor of shape (B + 1)
            num_query_cumsum[0] is 0, num_query_cumsum[i] is the number of queries
            for the first i graphs in the batch for i > 0.

        Returns
        -------
        torch.Tensor of shape (N, F)
            Updated hidden representations of query nodes for the batch of graphs.
        """
        # Handle different numbers of query nodes in the batch with padding.
        batch_size = len(num_query_cumsum) - 1
        num_query_nodes = torch.diff(num_query_cumsum)
        max_num_nodes = num_query_nodes.max().item()

        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1])
        k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1])
        v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        pad_mask = q.new_zeros(batch_size, max_num_nodes).bool()

        for i in range(batch_size):
            q_padded[i, :num_query_nodes[i]] = q[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            k_padded[i, :num_query_nodes[i]] = k[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            v_padded[i, :num_query_nodes[i]] = v[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            pad_mask[i, num_query_nodes[i]:] = True

        # Split F into H * D, where H is the number of heads
        # D is the dimension per head.

        # (B, H, max_num_nodes, D)
        q_padded = rearrange(q_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        # (B, H, max_num_nodes, D)
        k_padded = rearrange(k_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        # (B, H, max_num_nodes, D)
        v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)

        # Q * K^T / sqrt(D)
        # (B, H, max_num_nodes, max_num_nodes)
        dot = torch.matmul(q_padded, k_padded.transpose(-1, -2)) * self.scale
        # Mask unnormalized attention logits for non-existent nodes.
        dot = dot.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        attn_scores = F.softmax(dot, dim=-1)
        # (B, H, max_num_nodes, D)
        h_n_padded = torch.matmul(attn_scores, v_padded)
        # (B * max_num_nodes, H * D) = (B * max_num_nodes, F)
        h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')

        # Unpad the aggregation results.
        # (N, F)
        pad_mask = (~pad_mask).reshape(-1)
        return h_n_padded[pad_mask]

    def forward(self, h_n, num_query_cumsum):
        # Compute value matrix
        v_n = self.to_v(h_n)

        # Compute query and key matrices
        q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1)

        h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum)
        h_n_new = self.proj_new(h_n_new)

        # Add & Norm
        h_n = self.norm1(h_n + h_n_new)
        h_n = self.norm2(h_n + self.out_proj(h_n))

        return h_n

class NodePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 num_x_n_cat,
                 x_n_emb_size,
                 t_emb_size,
                 in_hidden_size,
                 out_hidden_size,
                 num_transformer_layers,
                 num_heads,
                 dropout):
        super().__init__()

        self.graph_encoder = graph_encoder
        num_real_classes = num_x_n_cat - 1
        self.x_n_emb = MultiEmbedding(num_real_classes, x_n_emb_size)
        self.t_emb = SinusoidalPE(t_emb_size)
        in_hidden_size = in_hidden_size + t_emb_size + len(num_real_classes) * x_n_emb_size
        self.project_h_n = nn.Sequential(
            nn.Linear(in_hidden_size, out_hidden_size),
            nn.GELU()
        )

        self.trans_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.trans_layers.append(TransformerLayer(
                out_hidden_size, num_heads, dropout
            ))

        self.pred_list = nn.ModuleList([])
        num_real_classes = num_real_classes.tolist()
        for num_classes_f in num_real_classes:
            self.pred_list.append(nn.Sequential(
                nn.Linear(out_hidden_size, out_hidden_size),
                nn.GELU(),
                nn.Linear(out_hidden_size, num_classes_f)
            ))

    def forward_with_h_g(self, h_g, x_n_t,
                         t, query2g, num_query_cumsum):
        h_t = self.t_emb(t)
        h_g = torch.cat([h_g, h_t], dim=1)

        h_n_t = self.x_n_emb(x_n_t)
        h_n_t = torch.cat([h_n_t, h_g[query2g]], dim=1)
        h_n_t = self.project_h_n(h_n_t)

        for trans_layer in self.trans_layers:
            h_n_t = trans_layer(h_n_t, num_query_cumsum)

        pred = []
        for d in range(len(self.pred_list)):
            pred.append(self.pred_list[d](h_n_t))

        return pred

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t,
                t, query2g, num_query_cumsum, y=None):
        """
        Parameters
        ----------
        x_n_t : torch.LongTensor of shape (Q)
        t : torch.LongTensor of shape (B, 1)
        query2g : torch.LongTensor of shape (Q)
        num_query_cumsum : torch.LongTensor of shape (B + 1)
        """
        h_g = self.graph_encoder(A, x_n, abs_level,
                                 rel_level, y=y, A_n2g=A_n2g)
        return self.forward_with_h_g(h_g, x_n_t, t, query2g,
                                     num_query_cumsum)

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
        self.node_count_model = GraphClassifier(
            node_count_encoder,
            emb_size=hidden_size,
            num_classes=max_layer_size+1).to(device)

        self.node_diffusion = node_diffusion
        hidden_size = len(num_x_n_cat) * node_pred_graph_encoder_config['x_n_emb_size'] +\
            node_pred_graph_encoder_config['pe_emb_size'] +\
            node_pred_graph_encoder_config['y_emb_size']
        node_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, hidden_size=hidden_size,
                                                **node_pred_graph_encoder_config).to(device)
        import ipdb
        ipdb.set_trace()
