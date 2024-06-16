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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
