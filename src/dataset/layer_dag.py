import torch

from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset',
           'LayerDAGNodePredDataset',
           'LayerDAGEdgePredDataset',
           'collate_node_count',
           'collate_node_pred',
           'collate_edge_pred']

class LayerDAGBaseDataset(Dataset):
    def __init__(self, conditional=False):
        self.input_src = []
        self.input_dst = []
        self.input_x_n = []
        self.input_level = []

        self.input_e_start = []
        self.input_e_end = []
        self.input_n_start = []
        self.input_n_end = []

        self.conditional = conditional
        if conditional:
            self.input_y = []
            self.input_g = []

    def get_in_deg(self, dst, num_nodes):
        return torch.bincount(dst, minlength=num_nodes).tolist()

    def get_out_adj_list(self, src, dst):
        out_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            out_adj_list[src[i]].append(dst[i])
        return out_adj_list

    def get_in_adj_list(self, src, dst):
        in_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            in_adj_list[dst[i]].append(src[i])
        return in_adj_list

    def base_postprocess(self):
        self.input_src = torch.LongTensor(self.input_src)
        self.input_dst = torch.LongTensor(self.input_dst)

        # Case1: self.input_x_n[0] is an int.
        # Case2: self.input_x_n[0] is a tensor of shape (F).
        self.input_x_n = torch.LongTensor(self.input_x_n)
        self.input_level = torch.LongTensor(self.input_level)

        self.input_e_start = torch.LongTensor(self.input_e_start)
        self.input_e_end = torch.LongTensor(self.input_e_end)
        self.input_n_start = torch.LongTensor(self.input_n_start)
        self.input_n_end = torch.LongTensor(self.input_n_end)

        if self.conditional:
            self.input_y = torch.tensor(self.input_y)
            self.input_g = torch.LongTensor(self.input_g)

class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)

        # Size of the next layer to predict.
        self.label = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                # Index of y in self.input_y
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            # For recording indices of the node attributes in self.input_x_n
            input_n_start = len(self.input_x_n)
            input_n_end = len(self.input_x_n)

            # For recording indices of the edges in self.input_src/self.input_dst
            input_e_start = len(self.input_src)
            input_e_end = len(self.input_src)

            # Use a dummy node for representing the initial empty DAG.
            self.input_x_n.append(dag_dataset.dummy_category)
            input_n_end += 1
            src = src + 1
            dst = dst + 1

            # Layer ID
            level = 0
            self.input_level.append(level)

            num_nodes = len(x_n) + 1
            in_deg = self.get_in_deg(dst, num_nodes)

            src = src.tolist()
            dst = dst.tolist()
            x_n = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src, dst)
            in_adj_list = self.get_in_adj_list(src, dst)

            frontiers = [
                u for u in range(1, num_nodes) if in_deg[u] == 0
            ]
            frontier_size = len(frontiers)
            while frontier_size > 0:
                # There is another layer.
                level += 1

                # Record indices for retrieving edges in the previous layers
                # for model input.
                self.input_e_start.append(input_e_start)
                self.input_e_end.append(input_e_end)

                # Record indices for retrieving node attributes in the previous
                # layers for model input.
                self.input_n_start.append(input_n_start)
                self.input_n_end.append(input_n_end)

                if conditional:
                    # Record the index for retrieving graph-level conditional
                    # information for model input.
                    self.input_g.append(input_g)
                self.label.append(frontier_size)

                # (1) Add the node attributes/edges for the current layer.
                # (2) Get the next layer.
                next_frontiers = []
                for u in frontiers:
                    # -1 for the initial dummy node
                    self.input_x_n.append(x_n[u - 1])
                    self.input_level.append(level)

                    for t in in_adj_list[u]:
                        self.input_src.append(t)
                        self.input_dst.append(u)
                        input_e_end += 1

                    for v in out_adj_list[u]:
                        in_deg[v] -= 1
                        if in_deg[v] == 0:
                            next_frontiers.append(v)
                input_n_end += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

            # Handle termination, namely predicting the layer size to be 0.
            self.input_e_start.append(input_e_start)
            self.input_e_end.append(input_e_end)
            self.input_n_start.append(input_n_start)
            self.input_n_end.append(input_n_end)
            if conditional:
                self.input_g.append(input_g)
            self.label.append(frontier_size)

        self.base_postprocess()
        self.label = torch.LongTensor(self.label)
        # Maximum number of nodes in a layer.
        self.max_layer_size = self.label.max().item()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        input_e_start = self.input_e_start[index]
        input_e_end = self.input_e_end[index]
        input_n_start = self.input_n_start[index]
        input_n_end = self.input_n_end[index]

        # Absolute and relative (with respect to the new layer) layer idx
        # for potential extra encodings.
        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item()

            return self.input_src[input_e_start:input_e_end],\
                self.input_dst[input_e_start:input_e_end],\
                self.input_x_n[input_n_start:input_n_end],\
                input_abs_level, input_rel_level, input_y, self.label[index]
        else:
            return self.input_src[input_e_start:input_e_end],\
                self.input_dst[input_e_start:input_e_end],\
                self.input_x_n[input_n_start:input_n_end],\
                input_abs_level, input_rel_level, self.label[index]

class LayerDAGNodePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)

        # Indices for retrieving the labels
        # (node attributes for the next layer)
        self.label_start = []
        self.label_end = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                # Index of y in self.input_y
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            # For recording indices of the node attributes in self.input_x_n,
            # which will be model input.
            input_n_start = len(self.input_x_n)
            input_n_end = len(self.input_x_n)

            # For recording indices of the edges in self.input_src/self.input_dst,
            # which will be model input.
            input_e_start = len(self.input_src)
            input_e_end = len(self.input_src)

            # Use a dummy node for representing the initial empty DAG, which
            # will be model input.
            self.input_x_n.append(dag_dataset.dummy_category)
            input_n_end += 1
            src = src + 1
            dst = dst + 1
            # For recording indices of the node attributes in self.input_x_n,
            # which will be ground truth labels for model predictions.
            label_start = len(self.input_x_n)

            # Layer ID
            level = 0
            self.input_level.append(level)

            num_nodes = len(x_n) + 1
            in_deg = self.get_in_deg(dst, num_nodes)

            src = src.tolist()
            dst = dst.tolist()
            x_n = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src, dst)
            in_adj_list = self.get_in_adj_list(src, dst)

            frontiers = [
                u for u in range(1, num_nodes) if in_deg[u] == 0
            ]
            frontier_size = len(frontiers)
            while frontier_size > 0:
                # There is another layer.
                level += 1

                # Record indices for retrieving edges in the previous layers
                # for model input.
                self.input_e_start.append(input_e_start)
                self.input_e_end.append(input_e_end)

                # Record indices for retrieving node attributes in the previous
                # layers for model input.
                self.input_n_start.append(input_n_start)
                self.input_n_end.append(input_n_end)

                if conditional:
                    # Record the index for retrieving graph-level conditional
                    # information for model input.
                    self.input_g.append(input_g)

                # Record indices for retrieving node attributes of the new
                # layer for model predictions.
                self.label_start.append(label_start)
                label_end = label_start + frontier_size
                self.label_end.append(label_end)
                label_start = label_end

                # (1) Add the node attributes/edges for the current layer.
                # (2) Get the next layer.
                next_frontiers = []
                for u in frontiers:
                    # -1 for the initial dummy node
                    self.input_x_n.append(x_n[u - 1])
                    self.input_level.append(level)

                    for t in in_adj_list[u]:
                        self.input_src.append(t)
                        self.input_dst.append(u)
                        input_e_end += 1

                    for v in out_adj_list[u]:
                        in_deg[v] -= 1
                        if in_deg[v] == 0:
                            next_frontiers.append(v)
                input_n_end += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)

        if get_marginal:
            # Case 1 (a single node attribute): self.input_x_n is of shape (N).
            # Case 2 (multiple node attributes): self.input_x_n is of shape (N, F).
            input_x_n = self.input_x_n
            if input_x_n.ndim == 1:
                input_x_n = input_x_n.unsqueeze(-1)

            num_feats = input_x_n.shape[-1]
            x_n_marginal = []
            for f in range(num_feats):
                input_x_n_f = input_x_n[:, f]
                unique_x_n_f, x_n_count_f = input_x_n_f.unique(return_counts=True)
                assert unique_x_n_f.max().item() == len(x_n_count_f) - 1,\
                    'Need to re-label node types to be consecutive integers starting from 0'

                # The last category is the dummy category.
                num_x_n_types_f = len(x_n_count_f) - 1
                x_n_marginal_f = torch.zeros(num_x_n_types_f)

                for c in range(len(x_n_count_f)):
                    x_n_type_f_c = unique_x_n_f[c].item()
                    # No need to include the dummy category for marginal computation.
                    if x_n_type_f_c != num_x_n_types_f:
                        x_n_marginal_f[x_n_type_f_c] = x_n_count_f[c].item()

                x_n_marginal_f /= (x_n_marginal_f.sum() + 1e-8)
                x_n_marginal.append(x_n_marginal_f)

            self.x_n_marginal = x_n_marginal

    def __len__(self):
        return len(self.label_start)

    def __getitem__(self, index):
        input_e_start = self.input_e_start[index]
        input_e_end = self.input_e_end[index]
        input_n_start = self.input_n_start[index]
        input_n_end = self.input_n_end[index]
        label_start = self.label_start[index]
        label_end = self.label_end[index]

        # Absolute and relative (with respect to the new layer) layer idx
        # for potential extra encodings.
        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level

        z = self.input_x_n[label_start:label_end]
        t, z_t = self.node_diffusion.apply_noise(z)

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item()

            return self.input_src[input_e_start:input_e_end],\
                self.input_dst[input_e_start:input_e_end],\
                self.input_x_n[input_n_start:input_n_end],\
                input_abs_level, input_rel_level, z_t, t, input_y, z
        else:
            return self.input_src[input_e_start:input_e_end],\
                self.input_dst[input_e_start:input_e_end],\
                self.input_x_n[input_n_start:input_n_end],\
                input_abs_level, input_rel_level, z_t, t, z

class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)

        self.query_src = []
        self.query_dst = []
        # Indices for retrieving the query node pairs
        self.query_start = []
        self.query_end = []
        self.label = []

        num_edges = 0
        num_nonsrc_nodes = 0
        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                # Index of y in self.input_y
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            # For recording indices of the node attributes in self.input_x_n,
            # which will be model input.
            input_n_start = len(self.input_x_n)
            input_n_end = len(self.input_x_n)

            # For recording indices of the edges in self.input_src/self.input_dst,
            # which will be model input.
            input_e_start = len(self.input_src)
            input_e_end = len(self.input_src)

            # For recording indices of the query node pairs in
            # self.query_src/self.query_dst for model predictions.
            query_start = len(self.query_src)
            query_end = len(self.query_src)

            # Use a dummy node for representing the initial empty DAG, which
            # will be model input.
            self.input_x_n.append(dag_dataset.dummy_category)
            input_n_end += 1
            src = src + 1
            dst = dst + 1

            # Layer ID
            level = 0
            self.input_level.append(level)

            num_nodes = len(x_n) + 1
            in_deg = self.get_in_deg(dst, num_nodes)

            src = src.tolist()
            dst = dst.tolist()
            x_n = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src, dst)
            in_adj_list = self.get_in_adj_list(src, dst)

            prev_frontiers = [
                u for u in range(1, num_nodes) if in_deg[u] == 0
            ]
            current_frontiers = []
            level += 1

            num_edges += len(src)
            num_nonsrc_nodes += len(x_n) - len(prev_frontiers)

            for u in prev_frontiers:
                self.input_x_n.append(x_n[u - 1])
                self.input_level.append(level)

                for v in out_adj_list[u]:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        current_frontiers.append(v)
            input_n_end += len(prev_frontiers)

            src_candidates = prev_frontiers

            while len(current_frontiers) > 0:
                level += 1

                next_frontiers = []
                temp_edge_count = 0
                for u in current_frontiers:
                    self.input_x_n.append(x_n[u - 1])
                    self.input_level.append(level)

                    self.query_src.extend(src_candidates)
                    self.query_dst.extend([u] * len(src_candidates))
                    query_end += len(src_candidates)
                    for t in src_candidates:
                        if t in in_adj_list[u]:
                            self.input_src.append(t)
                            self.input_dst.append(u)
                            temp_edge_count += 1
                            self.label.append(1)
                        else:
                            self.label.append(0)

                    for v in out_adj_list[u]:
                        in_deg[v] -= 1
                        if in_deg[v] == 0:
                            next_frontiers.append(v)

                input_n_end += len(current_frontiers)

                # Record indices for retrieving edges in the previous layers
                # for model input.
                self.input_e_start.append(input_e_start)
                self.input_e_end.append(input_e_end)

                # Record indices for retrieving node attributes in the previous
                # layers for model input.
                self.input_n_start.append(input_n_start)
                self.input_n_end.append(input_n_end)

                if conditional:
                    # Record the index for retrieving graph-level conditional
                    # information for model input.
                    self.input_g.append(input_g)

                # Record indices for retrieving query node pairs
                # for model predictions.
                self.query_start.append(query_start)
                self.query_end.append(query_end)

                src_candidates.extend(current_frontiers)
                prev_frontiers = current_frontiers
                current_frontiers = next_frontiers
                input_e_end += temp_edge_count
                query_start = query_end

        self.base_postprocess()
        self.query_src = torch.LongTensor(self.query_src)
        self.query_dst = torch.LongTensor(self.query_dst)
        self.query_start = torch.LongTensor(self.query_start)
        self.query_end = torch.LongTensor(self.query_end)
        self.label = torch.LongTensor(self.label)

        self.avg_in_deg = num_edges / num_nonsrc_nodes

    def __len__(self):
        return len(self.query_start)

    def __getitem__(self, index):
        input_e_start = self.input_e_start[index]
        input_e_end = self.input_e_end[index]
        input_src = self.input_src[input_e_start:input_e_end]
        input_dst = self.input_dst[input_e_start:input_e_end]

        input_n_start = self.input_n_start[index]
        input_n_end = self.input_n_end[index]
        input_x_n = self.input_x_n[input_n_start:input_n_end]

        # Absolute and relative (with respect to the new layer) layer idx
        # for potential extra encodings.
        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level

        query_start = self.query_start[index]
        query_end = self.query_end[index]
        query_src = self.query_src[query_start:query_end]
        query_dst = self.query_dst[query_start:query_end]
        label = self.label[query_start:query_end]

        unique_src = torch.unique(query_src, sorted=False)
        unique_dst = torch.unique(query_dst, sorted=False)
        label_adj = label.reshape(len(unique_dst), len(unique_src))

        t, label_t = self.edge_diffusion.apply_noise(label_adj)

        mask = (label_t == 1)
        noisy_src = query_src[mask]
        noisy_dst = query_dst[mask]

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item()

            return input_src, input_dst, noisy_src, noisy_dst, input_x_n,\
                input_abs_level, input_rel_level, t, input_y, query_src, query_dst, label
        else:
            return input_src, input_dst, noisy_src, noisy_dst, input_x_n,\
                input_abs_level, input_rel_level, t, query_src, query_dst, label

def collate_common(src, dst, x_n, abs_level, rel_level):
    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in x_n]), dim=0)

    batch_size = len(x_n)
    src_ = []
    dst_ = []
    for i in range(batch_size):
        src_.append(src[i] + num_nodes_cumsum[i])
        dst_.append(dst[i] + num_nodes_cumsum[i])
    src = torch.cat(src_, dim=0)
    dst = torch.cat(dst_, dim=0)
    edge_index = torch.stack([dst, src])

    x_n = torch.cat(x_n, dim=0).long()
    abs_level = torch.cat(abs_level, dim=0).float().unsqueeze(-1)
    rel_level = torch.cat(rel_level, dim=0).float().unsqueeze(-1)

    # Prepare edge index for node to graph mapping
    nids = []
    gids = []
    for i in range(batch_size):
        nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i+1]).long())
        gids.append(torch.ones(num_nodes_cumsum[i+1] - num_nodes_cumsum[i]).fill_(i).long())
    nids = torch.cat(nids, dim=0)
    gids = torch.cat(gids, dim=0)
    n2g_index = torch.stack([gids, nids])

    return batch_size, edge_index, x_n, abs_level, rel_level, n2g_index

def collate_node_count(data):
    if len(data[0]) == 7:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y, batch_label = map(list, zip(*data))

        y_ = []
        for i in range(len(batch_x_n)):
            y_.extend([batch_y[i]] * len(batch_x_n[i]))
        batch_y = torch.tensor(y_).unsqueeze(-1)
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(
            list, zip(*data))

    batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level,\
        batch_n2g_index = collate_common(
            batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    batch_label = torch.stack(batch_label)

    if len(data[0]) == 7:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, batch_y, batch_n2g_index, batch_label
    else:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, batch_n2g_index, batch_label

def collate_node_pred(data):
    if len(data[0]) == 8:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level,\
            batch_z_t, batch_t, batch_z = map(list, zip(*data))
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level,\
            batch_z_t, batch_t, batch_y, batch_z = map(list, zip(*data))
        # Broadcast graph-level conditional information to nodes.
        y_ = []
        for i in range(len(batch_x_n)):
            y_.extend([batch_y[i]] * len(batch_x_n[i]))
        batch_y = torch.tensor(y_).unsqueeze(-1)

    batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level,\
        batch_n2g_index = collate_common(
            batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    num_query_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(z_t_i) for z_t_i in batch_z_t]), dim=0)
    query2g = []
    for i in range(batch_size):
        query2g.append(torch.ones(num_query_cumsum[i+1] - num_query_cumsum[i]).fill_(i).long())
    query2g = torch.cat(query2g)

    batch_z_t = torch.cat(batch_z_t)
    batch_t = torch.cat(batch_t).unsqueeze(-1)
    batch_z = torch.cat(batch_z)

    if batch_z.ndim == 1:
        batch_z = batch_z.unsqueeze(-1)

    if len(data[0]) == 8:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, batch_n2g_index, batch_z_t, batch_t, query2g,\
            num_query_cumsum, batch_z
    else:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y,\
            query2g, num_query_cumsum, batch_z

def collate_edge_pred(data):
    if len(data[0]) == 11:
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n,\
            batch_abs_level, batch_rel_level, batch_t, batch_query_src,\
            batch_query_dst, batch_label = map(list, zip(*data))
    else:
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n,\
            batch_abs_level, batch_rel_level, batch_t, batch_y,\
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        # Broadcast graph-level conditional information to nodes.
        y_ = []
        for i in range(len(batch_x_n)):
            y_.extend([batch_y[i]] * len(batch_x_n[i]))
        batch_y = torch.tensor(y_).unsqueeze(-1)

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in batch_x_n]), dim=0)

    batch_size = len(batch_x_n)
    src_ = []
    dst_ = []
    noisy_src_ = []
    noisy_dst_ = []
    query_src_ = []
    query_dst_ = []
    t_ = []
    for i in range(batch_size):
        src_.append(batch_src[i] + num_nodes_cumsum[i])
        dst_.append(batch_dst[i] + num_nodes_cumsum[i])
        noisy_src_.append(batch_noisy_src[i] + num_nodes_cumsum[i])
        noisy_dst_.append(batch_noisy_dst[i] + num_nodes_cumsum[i])
        query_src_.append(batch_query_src[i] + num_nodes_cumsum[i])
        query_dst_.append(batch_query_dst[i] + num_nodes_cumsum[i])
        t_.append(batch_t[i].expand(len(batch_query_src[i]), -1))

    src = torch.cat(src_, dim=0)
    dst = torch.cat(dst_, dim=0)
    edge_index = torch.stack([dst, src])
    noisy_src = torch.cat(noisy_src_, dim=0)
    noisy_dst = torch.cat(noisy_dst_, dim=0)
    noisy_edge_index = torch.stack([noisy_dst, noisy_src])
    query_src = torch.cat(query_src_)
    query_dst = torch.cat(query_dst_)
    t = torch.cat(t_)

    batch_x_n = torch.cat(batch_x_n, dim=0).long()
    batch_abs_level = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1)
    batch_rel_level = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)

    batch_label = torch.cat(batch_label)

    if len(data[0]) == 11:
        return edge_index, noisy_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, t, query_src, query_dst, batch_label
    else:
        return edge_index, noisy_edge_index, batch_x_n, batch_abs_level,\
            batch_rel_level, t, batch_y, query_src, query_dst, batch_label
