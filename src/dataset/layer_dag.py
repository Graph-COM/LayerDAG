import torch

from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset']

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
