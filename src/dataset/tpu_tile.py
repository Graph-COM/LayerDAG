import os
import torch

from .general import DAGDataset

def get_tpu_tile():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(root_path, '../../data_files/tpu_tile_processed')

    train_path = os.path.join(root_path, 'train.pth')
    val_path = os.path.join(root_path, 'val.pth')
    test_path = os.path.join(root_path, 'test.pth')

    print('Loading TPU Tile dataset...')
    # Load the pre-processed TPU Tile dataset, where for each kernel graph, we
    # average the normalized runtime over multiple compiler configurations.
    train_set = torch.load(train_path)
    val_set = torch.load(val_path)
    test_set = torch.load(test_path)
