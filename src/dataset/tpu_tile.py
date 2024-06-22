import os
import torch

from .general import DAGDataset

def get_tpu_tile():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(root_path, '../../data_files/tpu_tile_processed')
