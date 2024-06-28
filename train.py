import dgl.sparse as dglsp
import pandas as pd
import time
import torch
import torch.nn as nn
import wandb

from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from setup_utils import set_seed, load_yaml
from src.dataset import load_dataset, LayerDAGNodeCountDataset,\
    LayerDAGNodePredDataset, LayerDAGEdgePredDataset, collate_node_count
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

def main_node_count(device, train_set, val_set, model, config, patience):
    pass

def main(args):
    torch.set_num_threads(args.num_threads)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    set_seed(args.seed)

    config = load_yaml(args.config_file)
    dataset = config['general']['dataset']
    config_df = pd.json_normalize(config, sep='/')

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())

    wandb.init(
        project=f'LayerDAG_{dataset}',
        name=f'{ts}',
        config=config_df.to_dict(orient='records')[0]
    )

    # For training the generative model, no need to use the test set.
    train_set, val_set, _ = load_dataset(dataset)

    train_node_count_dataset = LayerDAGNodeCountDataset(train_set, config['general']['conditional'])
    val_node_count_dataset = LayerDAGNodeCountDataset(val_set, config['general']['conditional'])

    train_node_pred_dataset = LayerDAGNodePredDataset(train_set, config['general']['conditional'])
    val_node_pred_dataset = LayerDAGNodePredDataset(
        val_set, config['general']['conditional'], get_marginal=False)

    node_diffusion_config = {
        'marginal_list': train_node_pred_dataset.x_n_marginal,
        'T': config['node_pred']['T']
    }
    node_diffusion = DiscreteDiffusion(**node_diffusion_config)
    train_node_pred_dataset.node_diffusion = node_diffusion
    val_node_pred_dataset.node_diffusion = node_diffusion

    train_edge_pred_dataset = LayerDAGEdgePredDataset(train_set, config['general']['conditional'])
    val_edge_pred_dataset = LayerDAGEdgePredDataset(val_set, config['general']['conditional'])

    edge_diffusion_config = {
        'avg_in_deg': train_edge_pred_dataset.avg_in_deg,
        'T': config['edge_pred']['T']
    }
    edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)
    train_edge_pred_dataset.edge_diffusion = edge_diffusion
    val_edge_pred_dataset.edge_diffusion = edge_diffusion

    model_config = {
        'num_x_n_cat': train_set.num_categories,
        'node_count_encoder_config': config['node_count']['model'],
        'max_layer_size': train_node_count_dataset.max_layer_size,
        'node_pred_graph_encoder_config': config['node_pred']['graph_encoder'],
        'node_predictor_config': config['node_pred']['predictor'],
        'edge_pred_graph_encoder_config': config['edge_pred']['graph_encoder'],
        'edge_predictor_config': config['edge_pred']['predictor'],
        'max_level': max(train_node_pred_dataset.input_level.max().item(),
                         val_node_pred_dataset.input_level.max().item())
    }
    model = LayerDAG(device=device,
                     node_diffusion=node_diffusion,
                     edge_diffusion=edge_diffusion,
                     **model_config)

    node_count_state_dict = main_node_count(
        device, train_node_count_dataset, val_node_count_dataset,
        model.node_count_model, config['node_count'], config['general']['patience'])
    model.node_count_model.load_state_dict(node_count_state_dict)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
