import torch

from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset',
           'LayerDAGNodePredDataset']

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
        self.max_num_nodes = self.label.max().item()

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
