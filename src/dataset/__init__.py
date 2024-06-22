from .general import DAGDataset

def load_dataset(dataset_name):
    if dataset_name == 'tpu_tile':
        pass
    else:
        return NotImplementedError
