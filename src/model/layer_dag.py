import dgl.sparse as dglsp
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

class EdgePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 t_emb_size,
                 in_hidden_size,
                 out_hidden_size):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.t_emb = SinusoidalPE(t_emb_size)
        self.pred = nn.Sequential(
            nn.Linear(2 * in_hidden_size + t_emb_size, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)
        )

    def forward(self, A, x_n, abs_level, rel_level, t,
                query_src, query_dst, y=None):
        """
        t : torch.tensor of shape (num_queries, 1)
        """
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, y=y)

        h_e = torch.cat([
            self.t_emb(t),
            h_n[query_src],
            h_n[query_dst]
        ], dim=-1)

        return self.pred(h_e)

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
        self.node_pred_model = NodePredModel(node_pred_graph_encoder,
                                             num_x_n_cat,
                                             node_pred_graph_encoder_config['x_n_emb_size'],
                                             in_hidden_size=hidden_size,
                                             **node_predictor_config).to(device)

        self.edge_diffusion = edge_diffusion
        hidden_size = len(num_x_n_cat) * edge_pred_graph_encoder_config['x_n_emb_size'] +\
            edge_pred_graph_encoder_config['pe_emb_size'] +\
            edge_pred_graph_encoder_config['y_emb_size']
        edge_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, hidden_size=hidden_size,
                                                **edge_pred_graph_encoder_config).to(device)
        self.edge_pred_model = EdgePredModel(edge_pred_graph_encoder,
                                             in_hidden_size=hidden_size,
                                             **edge_predictor_config).to(device)

        self.max_level = max_level

    def posterior(self, Z_t, Q_t, Q_bar_s, Q_bar_t, Z_0):
        # (num_rows, num_classes)
        left_term = Z_t @ torch.transpose(Q_t, -1, -2)
        # (num_rows, 1, num_classes)
        left_term = left_term.unsqueeze(dim=-2)
        # (1, num_classes, num_classes)
        right_term = Q_bar_s.unsqueeze(dim=-3)
        # (num_rows, num_classes, num_classes)
        numerator = left_term * right_term

        # (num_classes, num_rows)
        prod = Q_bar_t @ torch.transpose(Z_t, -1, -2)
        # (num_rows, num_classes)
        prod = torch.transpose(prod, -1, -2)
        # (num_rows, num_classes, 1)
        denominator = prod.unsqueeze(-1)
        denominator[denominator == 0.] = 1.
        # (num_rows, num_classes, num_classes)
        out = numerator / denominator

        # (num_rows, num_classes, num_classes)
        prob = Z_0.unsqueeze(-1) * out
        # (num_rows, num_classes)
        prob = prob.sum(dim=-2)

        return prob

    def posterior_edge(self,
                       Z_t,
                       alpha_t,
                       alpha_bar_s,
                       alpha_bar_t,
                       Z_0,
                       marginal_list,
                       num_new_nodes_list,
                       num_query_list):
        batch_size = len(num_new_nodes_list)
        Z_t_list = torch.split(Z_t, num_query_list, dim=0)
        Z_0_list = torch.split(Z_0, num_query_list, dim=0)
        device = Z_t.device
        e_mask_list = []

        for i in range(batch_size):
            Z_t_i = Z_t_list[i]
            Z_0_i = Z_0_list[i]

            Q_t_i, Q_bar_s_i, Q_bar_t_i = self.edge_diffusion.get_Qs(
                alpha_t, alpha_bar_s, alpha_bar_t, marginal_list[i])
            Q_t_i = Q_t_i.to(device)
            Q_bar_s_i = Q_bar_s_i.to(device)
            Q_bar_t_i = Q_bar_t_i.to(device)

            # (num_rows, num_classes)
            left_term_i = Z_t_i @ torch.transpose(Q_t_i, -1, -2)
            # (num_rows, 1, num_classes)
            left_term_i = left_term_i.unsqueeze(dim=-2)
            # (1, num_classes, num_classes)
            right_term_i = Q_bar_s_i.unsqueeze(dim=-3)
            # (num_rows, num_classes, num_classes)
            numerator_i = left_term_i * right_term_i

            # (num_classes, num_rows)
            prod_i = Q_bar_t_i @ torch.transpose(Z_t_i, -1, -2)
            # (num_rows, num_classes)
            prod_i = torch.transpose(prod_i, -1, -2)
            # (num_rows, num_classes, 1)
            denominator_i = prod_i.unsqueeze(-1)
            denominator_i[denominator_i == 0.] = 1.
            # (num_rows, num_classes, num_classes)
            out_i = numerator_i / denominator_i

            # (num_rows, num_classes, num_classes)
            prob_i = Z_0_i.unsqueeze(-1) * out_i
            # (num_rows, num_classes)
            prob_i = prob_i.sum(dim=-2)
            prob_i = prob_i / (prob_i.sum(dim=-1, keepdim=True) + 1e-6)

            # Get the probabilities for edge existence.
            prob_i = prob_i[:, 1]
            prob_i = prob_i.reshape(num_new_nodes_list[i], -1)
            e_mask_i = torch.bernoulli(prob_i)

            isolated_mask_i = (e_mask_i.sum(dim=1) == 0).bool()
            if isolated_mask_i.any():
                e_mask_i[isolated_mask_i, prob_i[isolated_mask_i].argmax(dim=1)] = 1
            e_mask_list.append(e_mask_i.reshape(-1))

        return torch.cat(e_mask_list).bool()

    @torch.no_grad()
    def sample_node_layer(self,
                          A,
                          x_n,
                          abs_level,
                          rel_level,
                          A_n2g,
                          curr_level=None,
                          y=None,
                          min_num_steps_n=None,
                          max_num_steps_n=None):
        device = A.device

        node_count_logits = self.node_count_model(A, x_n, abs_level,
                                                  rel_level, A_n2g=A_n2g, y=y)

        # For the first layer, the layer size must be nonzero.
        if curr_level == 0:
            node_count_logits[:, 0] = float('-inf')

        node_count_probs = node_count_logits.softmax(dim=-1)
        num_new_nodes = node_count_probs.multinomial(1)

        num_new_nodes_total = num_new_nodes.sum().item()
        batch_size = num_new_nodes.shape[0]
        if num_new_nodes_total == 0:
            return [torch.LongTensor([]).to(device)
                    for _ in range(batch_size)]

        num_classes_list = self.node_diffusion.num_classes_list
        marginal_list = self.node_diffusion.m_list
        D = len(num_classes_list)

        x_n_t = []
        for d in range(D):
            marginal_d = marginal_list[d]
            prior_d = marginal_d[0][None, :].expand(num_new_nodes_total, -1)
            # (num_new_nodes_total)
            x_n_t_d = prior_d.multinomial(1).squeeze(-1)
            x_n_t.append(x_n_t_d)
        x_n_t = torch.stack(x_n_t, dim=1).to(device)

        # Iteratively sample p(D^s | D^t) for t = 1, ..., T, with s = t - 1.
        h_g = self.node_pred_model.graph_encoder(A, x_n, abs_level, rel_level,
                                                 y=y, A_n2g=A_n2g)

        num_query_cumsum = torch.cumsum(torch.tensor(
            [0] + num_new_nodes.squeeze(-1).tolist()), dim=0).long().to(device)
        query2g = []
        for i in range(batch_size):
            query2g.append(torch.ones(num_query_cumsum[i+1] - num_query_cumsum[i]).fill_(i).long())
        query2g = torch.cat(query2g).to(device)

        T_x_n = self.node_diffusion.T
        if max_num_steps_n is not None:
            T_x_n = min(T_x_n, max_num_steps_n)

        time_x_n_list = list(reversed(range(0, T_x_n)))
        if min_num_steps_n is not None:
            num_steps_n = min_num_steps_n + int(
                (T_x_n - min_num_steps_n) * (curr_level / self.max_level)
            )
            time_x_n_list = time_x_n_list[-num_steps_n:]

        for s_x_n in time_x_n_list:
            t_x_n = s_x_n + 1

            # Note that computing Q_bar_t from alpha_bar_t is the same
            # as computing Q_t from alpha_t.
            alpha_t = self.node_diffusion.alphas[t_x_n]
            alpha_bar_s = self.node_diffusion.alpha_bars[s_x_n]
            alpha_bar_t = self.node_diffusion.alpha_bars[t_x_n]

            t_x_n_tensor = torch.LongTensor([[t_x_n]]).expand(batch_size, -1).to(device)
            x_n_0_logits = self.node_pred_model.forward_with_h_g(
                h_g, x_n_t, t_x_n_tensor, query2g,
                num_query_cumsum)

            x_n_s = []
            for d in range(D):
                Q_t_d = self.node_diffusion.get_Q(alpha_t, d).to(device)
                Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s, d).to(device)
                Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t, d).to(device)

                x_n_0_probs_d = x_n_0_logits[d].softmax(dim=-1)
                # (num_new_nodes, num_classes)
                x_n_t_one_hot_d = F.one_hot(x_n_t[:, d], num_classes=num_classes_list[d]).float()

                x_n_s_probs_d = self.posterior(x_n_t_one_hot_d, Q_t_d, Q_bar_s_d,
                                               Q_bar_t_d, x_n_0_probs_d)
                x_n_s_d = x_n_s_probs_d.multinomial(1).squeeze(-1)
                x_n_s.append(x_n_s_d)

            x_n_t = torch.stack(x_n_s, dim=1)

        return torch.split(x_n_t, num_new_nodes.squeeze(-1).tolist())

    @torch.no_grad()
    def sample_edge_layer(self, num_nodes_cumsum, edge_index_list,
                          batch_x_n, batch_abs_level, batch_rel_level,
                          num_new_nodes_list, batch_query_src, batch_query_dst,
                          query_src_list, query_dst_list,
                          batch_y=None,
                          curr_level=None,
                          min_num_steps_e=None,
                          max_num_steps_e=None):
        device = batch_x_n.device

        e_t_mask_list = []
        batch_size = len(num_new_nodes_list)
        marginal_list = []
        num_query_list = []
        for i in range(batch_size):
            num_query_i = len(query_src_list[i])
            num_query_list.append(num_query_i)

            num_new_nodes_i = num_new_nodes_list[i]
            prior_i = torch.ones(num_query_i).reshape(num_new_nodes_i, -1)
            mean_in_deg_i = min(self.edge_diffusion.avg_in_deg, prior_i.shape[1])
            marginal_i = mean_in_deg_i / prior_i.shape[1]
            marginal_list.append(marginal_i)
            prior_i = prior_i * marginal_i
            e_t_mask_i = torch.bernoulli(prior_i)
            isolated_mask = (e_t_mask_i.sum(dim=1) == 0).bool()
            if isolated_mask.any():
                e_t_mask_i[isolated_mask, torch.zeros(int(isolated_mask.sum().item())).long()] = 1
            e_t_mask_list.append(e_t_mask_i.reshape(-1))

        e_t_mask = torch.cat(e_t_mask_list).bool().to(device)

        num_nodes = len(batch_x_n)
        num_queries = len(batch_query_src)

        batch_edge_index = self.get_batch_A(
            num_nodes_cumsum, edge_index_list, device,
            return_edge_index=True)

        # Iteratively sample p(D^s | D^t) for t = 1, ..., T, with s = t - 1.
        T_x_e = self.edge_diffusion.T
        if max_num_steps_e is not None:
            T_x_e = min(T_x_e, max_num_steps_e)

        time_x_e_list = list(reversed(range(0, T_x_e)))
        if min_num_steps_e is not None:
            num_steps_e = min_num_steps_e + int(
                (T_x_e - min_num_steps_e) * (curr_level / self.max_level)
            )
            time_x_e_list = time_x_e_list[-num_steps_e:]

        for s_x_e in time_x_e_list:
            t_x_e = s_x_e + 1

            # Note that computing Q_bar_t from alpha_bar_t is the same
            # as computing Q_t from alpha_t.
            alpha_t = self.edge_diffusion.alphas[t_x_e]
            alpha_bar_s = self.edge_diffusion.alpha_bars[s_x_e]
            alpha_bar_t = self.edge_diffusion.alpha_bars[t_x_e]

            edge_index_t = torch.stack([
                batch_query_dst[e_t_mask],
                batch_query_src[e_t_mask]
            ]).to(device)

            A = dglsp.spmatrix(
                torch.cat([batch_edge_index, edge_index_t], dim=1),
                shape=(num_nodes, num_nodes)).to(device)
            t_x_e_tensor = torch.LongTensor([t_x_e])[None, :].expand(
                num_queries, -1).to(device)
            e_0_logits = self.edge_pred_model(
                A, batch_x_n, batch_abs_level, batch_rel_level, t_x_e_tensor,
                batch_query_src, batch_query_dst, batch_y)
            e_0_probs = e_0_logits.softmax(dim=-1)
            # (num_queries, num_classes)
            e_t_one_hot = F.one_hot(e_t_mask.long(), num_classes=2).float()

            e_t_mask = self.posterior_edge(e_t_one_hot,
                                           alpha_t,
                                           alpha_bar_s,
                                           alpha_bar_t,
                                           e_0_probs,
                                           marginal_list,
                                           num_new_nodes_list,
                                           num_query_list)

        num_query_split = [len(query_src_i) for query_src_i in query_src_list]
        e_t_mask_split = torch.split(e_t_mask, num_query_split)

        edge_index_list_ = []
        for i in range(len(edge_index_list)):
            edge_index_i = edge_index_list[i]
            e_t_mask_i = e_t_mask_split[i]
            edge_index_l_i = torch.stack([
                query_dst_list[i][e_t_mask_i],
                query_src_list[i][e_t_mask_i]
            ])
            edge_index_i = torch.cat([edge_index_i, edge_index_l_i], dim=1)
            edge_index_list_.append(edge_index_i)
        edge_index_list = edge_index_list_

        return edge_index_list

    def get_batch_A(self, num_nodes_cumsum, edge_index_list, device, return_edge_index=False):
        batch_size = len(edge_index_list)
        edge_index_list_ = []
        for i in range(batch_size):
            edge_index_list_.append(edge_index_list[i] + num_nodes_cumsum[i])

        batch_edge_index = torch.cat(edge_index_list_, dim=1)

        if return_edge_index:
            return batch_edge_index

        N = num_nodes_cumsum[-1].item()
        batch_A = dglsp.spmatrix(batch_edge_index, shape=(N, N)).to(device)

        return batch_A

    def get_batch_A_n2g(self, num_nodes_cumsum, device):
        batch_size = len(num_nodes_cumsum) - 1
        nids = []
        gids = []
        for i in range(batch_size):
            nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i+1]).long())
            gids.append(torch.ones(num_nodes_cumsum[i+1] - num_nodes_cumsum[i]).fill_(i).long())

        nids = torch.cat(nids, dim=0)
        gids = torch.cat(gids, dim=0)
        n2g_index = torch.stack([gids, nids])

        N = num_nodes_cumsum[-1].item()
        batch_A_n2g = dglsp.spmatrix(n2g_index, shape=(batch_size, N)).to(device)

        return batch_A_n2g

    def get_batch_y(self, y_list, x_n_list, device):
        if y_list is None:
            return None

        y_list_ = []
        for i in range(len(x_n_list)):
            y_list_.append(torch.zeros(len(x_n_list[i]), 1).fill_(y_list[i]))
        batch_y = torch.cat(y_list_).to(device)

        return batch_y

    @torch.no_grad()
    def sample(self,
               device,
               batch_size=1,
               y=None,
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):
        if y is not None:
            assert batch_size == len(y)
        y_list = y

        edge_index_list = [
            torch.LongTensor([[], []]).to(device)
            for _ in range(batch_size)
        ]

        if isinstance(self.dummy_x_n, int):
            init_x_n = torch.LongTensor([[self.dummy_x_n]]).to(device)
        elif isinstance(self.dummy_x_n, torch.Tensor):
            init_x_n = self.dummy_x_n.to(device).unsqueeze(0)
        else:
            raise NotImplementedError
        x_n_list = [init_x_n for _ in range(batch_size)]
        batch_x_n = torch.cat(x_n_list)
        batch_y = self.get_batch_y(y_list, x_n_list, device)

        level = 0.
        abs_level_list = [
            torch.tensor([[level]]).to(device)
            for _ in range(batch_size)
        ]
        batch_abs_level = torch.cat(abs_level_list)
        batch_rel_level = batch_abs_level.max() - batch_abs_level

        edge_index_finished = []
        x_n_finished = []
        if y is not None:
            y_finished = []

        num_nodes_cumsum = torch.cumsum(torch.tensor(
            [0] + [len(x_n_i) for x_n_i in x_n_list]), dim=0)
        while True:
            batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, device)
            batch_A_n2g = self.get_batch_A_n2g(num_nodes_cumsum, device)
            x_n_l_list = self.sample_node_layer(
                batch_A, batch_x_n, batch_abs_level, batch_rel_level,
                batch_A_n2g, curr_level=level,
                y=batch_y,
                min_num_steps_n=min_num_steps_n,
                max_num_steps_n=max_num_steps_n)

            edge_index_list_ = []
            x_n_list_ = []
            abs_level_list_ = []
            query_src_list = []
            query_dst_list = []
            num_new_nodes_list = []
            batch_query_src = []
            batch_query_dst = []

            if y is not None:
                y_list_ = []
            else:
                y_list_ = None

            level += 1
            node_count = 0
            for i, x_n_l_i in enumerate(x_n_l_list):
                if len(x_n_l_i) == 0:
                    edge_index_finished.append(edge_index_list[i] - 1)
                    x_n_finished.append(x_n_list[i][1:])
                    if y is not None:
                        y_finished.append(y_list[i])
                else:
                    edge_index_list_.append(edge_index_list[i])
                    x_n_list_.append(torch.cat([x_n_list[i], x_n_l_i]))
                    if y is not None:
                        y_list_.append(y_list[i])
                    abs_level_list_.append(
                        torch.cat([
                            abs_level_list[i],
                            torch.zeros(len(x_n_l_i), 1).fill_(level).to(device)
                        ])
                    )

                    N_old_i = len(x_n_list[i])
                    N_new_i = len(x_n_l_i)

                    query_src_i = []
                    query_dst_i = []

                    src_candidates_i = list(range(1, N_old_i))
                    for dst_i in range(N_old_i, N_old_i + N_new_i):
                        query_src_i.extend(src_candidates_i)
                        query_dst_i.extend([dst_i] * len(src_candidates_i))
                    query_src_i = torch.LongTensor(query_src_i).to(device)
                    query_dst_i = torch.LongTensor(query_dst_i).to(device)

                    query_src_list.append(query_src_i)
                    query_dst_list.append(query_dst_i)
                    batch_query_src.append(query_src_i + node_count)
                    batch_query_dst.append(query_dst_i + node_count)
                    num_new_nodes_list.append(N_new_i)

                    node_count = node_count + N_old_i + N_new_i

            edge_index_list = edge_index_list_
            x_n_list = x_n_list_
            y_list = y_list_
            abs_level_list = abs_level_list_

            if len(edge_index_list) == 0:
                break

            num_nodes_cumsum = torch.cumsum(torch.tensor(
                [0] + [len(x_n_i) for x_n_i in x_n_list]), dim=0)
            batch_x_n = torch.cat(x_n_list)
            batch_abs_level = torch.cat(abs_level_list)
            batch_rel_level = batch_abs_level.max() - batch_abs_level
            batch_y = self.get_batch_y(y_list, x_n_list, device)

            if level == 1:
                continue

            batch_query_src = torch.cat(batch_query_src)
            batch_query_dst = torch.cat(batch_query_dst)

            edge_index_list = self.sample_edge_layer(
                num_nodes_cumsum, edge_index_list, batch_x_n, batch_abs_level,
                batch_rel_level, num_new_nodes_list, batch_query_src,
                batch_query_dst, query_src_list, query_dst_list, batch_y,
                curr_level=level,
                min_num_steps_e=min_num_steps_e,
                max_num_steps_e=max_num_steps_e)

            if self.max_level is not None and level == self.max_level:
                break

        for i in range(len(edge_index_list)):
            edge_index_finished.append(edge_index_list[i] - 1)
            x_n_finished.append(x_n_list[i][1:])

        if y is None:
            return edge_index_finished, x_n_finished
        else:
            y_finished.extend(y_list)
            return edge_index_finished, x_n_finished, y_finished
