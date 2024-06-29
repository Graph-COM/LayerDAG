import torch

from torch.utils.data import Dataset

class DAGDataset(Dataset):
    """
    Parameters
    ----------
    label : bool
        Whether each DAG has a label like runtime/latency.
    """
    def __init__(self, num_categories, label=False):
        self.src = []
        self.dst = []
        self.x_n = []

        self.label = label
        if self.label:
            self.y = []

        self.dummy_category = num_categories
        if isinstance(self.dummy_category, torch.Tensor):
            self.dummy_category = self.dummy_category.tolist()

        self.num_categories = num_categories + 1

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        if self.label:
            return self.src[index], self.dst[index], self.x_n[index], self.y[index]
        else:
            return self.src[index], self.dst[index], self.x_n[index]

    def add_data(self, src, dst, x_n, y=None):
        self.src.append(src)
        self.dst.append(dst)
        self.x_n.append(x_n)
        if (y is not None) and (self.label):
            self.y.append(y)
