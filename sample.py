import os
import torch

from pprint import pprint
from tqdm import tqdm

from setup_utils import set_seed
from src.dataset import load_dataset, DAGDataset
from src.eval import TPUTileEvaluator
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

def sample_tpu_subset(args, device, dummy_category, model, subset):
    syn_set = DAGDataset(dummy_category, label=True)

    raw_y_batch = []
    for i, y in enumerate(tqdm(subset.y)):
        raw_y_batch.append(y)
        if (len(raw_y_batch) == args.batch_size) or (i == len(subset.y) - 1):
            batch_edge_index, batch_x_n, batch_y = model.sample(
                device, len(raw_y_batch), raw_y_batch,
                min_num_steps_n=args.min_num_steps_n,
                max_num_steps_n=args.max_num_steps_n,
                min_num_steps_e=args.min_num_steps_e,
                max_num_steps_e=args.max_num_steps_e)

            for j in range(len(batch_edge_index)):
                edge_index_j = batch_edge_index[j]
                dst_j, src_j = edge_index_j.cpu()
                syn_set.add_data(src_j, dst_j, batch_x_n[j].cpu(),
                                 batch_y[j])

            raw_y_batch = []

    return syn_set

def dump_to_file(syn_set, file_name, sample_dir):
    file_path = os.path.join(sample_dir, file_name)
    data_dict = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
        'y_list': []
    }
    for i in range(len(syn_set)):
        src_i, dst_i, x_n_i, y_i = syn_set[i]

        data_dict['src_list'].append(src_i)
        data_dict['dst_list'].append(dst_i)
        data_dict['x_n_list'].append(x_n_i)
        data_dict['y_list'].append(y_i)

    torch.save(data_dict, file_path)

def eval_tpu_tile(args, device, model):
    sample_dir = 'tpu_tile_samples'
    os.makedirs(sample_dir, exist_ok=True)

    evaluator = TPUTileEvaluator()
    train_set, val_set, _ = load_dataset('tpu_tile')

    train_syn_set = sample_tpu_subset(args, device, train_set.dummy_category, model, train_set)
    val_syn_set = sample_tpu_subset(args, device, train_set.dummy_category, model, val_set)

    evaluator.eval(train_syn_set, val_syn_set)

    dump_to_file(train_syn_set, 'train.pth', sample_dir)
    dump_to_file(val_syn_set, 'val.pth', sample_dir)

def main(args):
    torch.set_num_threads(args.num_threads)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    ckpt = torch.load(args.model_path)

    dataset = ckpt['dataset']
    assert dataset == 'tpu_tile'

    node_diffusion = DiscreteDiffusion(**ckpt['node_diffusion_config'])
    edge_diffusion = EdgeDiscreteDiffusion(**ckpt['edge_diffusion_config'])

    model = LayerDAG(device=device,
                     node_diffusion=node_diffusion,
                     edge_diffusion=edge_diffusion,
                     **ckpt['model_config'])
    pprint(ckpt['model_config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    set_seed(args.seed)

    eval_tpu_tile(args, device, model)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--min_num_steps_n", type=int, default=None)
    parser.add_argument("--min_num_steps_e", type=int, default=None)
    parser.add_argument("--max_num_steps_n", type=int, default=None)
    parser.add_argument("--max_num_steps_e", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
