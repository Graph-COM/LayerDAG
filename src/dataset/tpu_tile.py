import os
import torch

from .general import DAGDataset

def to_dag_dataset(data_dict, num_categories):
    dataset = DAGDataset(num_categories=num_categories, label=True)

    src_list = data_dict['src_list']
    dst_list = data_dict['dst_list']
    x_n_list = data_dict['x_n_list']
    y_list = data_dict['y_list']

    num_g = len(src_list)
    for i in range(num_g):
        dataset.add_data(src_list[i],
                         dst_list[i],
                         x_n_list[i],
                         y_list[i])

    return dataset

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

    num_categories = torch.cat(train_set['x_n_list']).max().item() + 1
    train_set = to_dag_dataset(train_set, num_categories)
    val_set = to_dag_dataset(val_set, num_categories)
    test_set = to_dag_dataset(test_set, num_categories)

    return train_set, val_set, test_set
