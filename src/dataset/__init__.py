from .layer_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile

def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        return get_tpu_tile()
    else:
        return NotImplementedError
