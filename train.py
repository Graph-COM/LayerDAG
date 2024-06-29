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
    LayerDAGNodePredDataset, LayerDAGEdgePredDataset, collate_node_count,\
    collate_node_pred, collate_edge_pred
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

@torch.no_grad()
def eval_node_count(device, val_loader, model):
    model.eval()
    total_nll = 0
    total_count = 0
    true_count = 0
    for batch_data in tqdm(val_loader):
        if len(batch_data) == 8:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
            batch_y = batch_y.to(device)
        else:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_n2g_index, batch_label = batch_data
            batch_y = None

        num_nodes = len(batch_x_n)
        batch_A = dglsp.spmatrix(
            batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_A_n2g, batch_y)

        batch_nll = -batch_logits.log_softmax(dim=-1)
        # In case the max layer size in the validation set is larger than
        # that in the training set.
        batch_label = batch_label.clamp(max=batch_nll.shape[-1] - 1)
        batch_nll = batch_nll[torch.arange(batch_size).to(device), batch_label]
        total_nll += batch_nll.sum().item()

        batch_probs = batch_logits.softmax(dim=-1)
        batch_preds = batch_probs.multinomial(1).squeeze(-1)
        true_count += (batch_preds == batch_label).sum().item()

        total_count += batch_size

    return total_nll / total_count, true_count / total_count

def main_node_count(device, train_set, val_set, model, config, patience):
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              collate_fn=collate_node_count,
                              **config['loader'],
                              drop_last=True)
    val_loader = DataLoader(val_set,
                            shuffle=False,
                            collate_fn=collate_node_count,
                            **config['loader'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_val_acc = 0
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0
    for epoch in range(config['num_epochs']):
        model.train()
        for batch_data in tqdm(train_loader):
            if len(batch_data) == 8:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
                batch_y = batch_y.to(device)
            else:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_n2g_index, batch_label = batch_data
                batch_y = None

            num_nodes = len(batch_x_n)
            batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_label = batch_label.to(device)

            batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                               batch_rel_level, batch_A_n2g, batch_y)

            loss = criterion(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'node_count/loss': loss.item()})

        val_nll, val_acc = eval_node_count(device, val_loader, model)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1
        wandb.log({'node_count/epoch': epoch,
                   'node_count/val_nll': val_nll,
                   'node_count/best_val_nll': best_val_nll,
                   'node_count/val_acc': val_acc,
                   'node_count/best_val_acc': best_val_acc,
                   'node_count/num_patient_epochs': num_patient_epochs})

        if (patience is not None) and (num_patient_epochs == patience):
            break

    return best_state_dict

@torch.no_grad()
def eval_node_pred(device, val_loader, model):
    model.eval()
    total_nll = 0
    total_count = 0
    for batch_data in tqdm(val_loader):
        if len(batch_data) == 11:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_n2g_index, batch_z_t, batch_t, query2g,\
                num_query_cumsum, batch_z = batch_data
            batch_y = None
        else:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y,\
                query2g, num_query_cumsum, batch_z = batch_data
            batch_y = batch_y.to(device)

        num_nodes = len(batch_x_n)
        batch_A = dglsp.spmatrix(
            batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_z_t = batch_z_t.to(device)
        batch_t = batch_t.to(device)
        query2g = query2g.to(device)
        num_query_cumsum = num_query_cumsum.to(device)
        batch_z = batch_z.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_A_n2g, batch_z_t, batch_t,
                             query2g, num_query_cumsum, batch_y)

        D = len(batch_logits)
        batch_num_queries = batch_logits[0].shape[0]
        for d in range(D):
            batch_logits_d = batch_logits[d]
            batch_nll_d = -batch_logits_d.log_softmax(dim=-1)
            batch_nll_d = batch_nll_d[torch.arange(batch_num_queries).to(device), batch_z[:, d]]
            total_nll += batch_nll_d.sum().item()
        total_count += batch_num_queries * D

    return total_nll / total_count

def main_node_pred(device, train_set, val_set, model, config, patience):
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              collate_fn=collate_node_pred,
                              **config['loader'])
    val_loader = DataLoader(val_set,
                            collate_fn=collate_node_pred,
                            **config['loader'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0
    for epoch in range(config['num_epochs']):
        val_nll = eval_node_pred(device, val_loader, model)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        wandb.log({'node_pred/epoch': epoch,
                   'node_pred/val_nll': val_nll,
                   'node_pred/best_val_nll': best_val_nll,
                   'node_pred/num_patient_epochs': num_patient_epochs})

        if (patience is not None) and (num_patient_epochs == patience):
            break

        model.train()
        for batch_data in tqdm(train_loader):
            if len(batch_data) == 11:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_n2g_index, batch_z_t, batch_t,\
                    query2g, num_query_cumsum, batch_z = batch_data
                batch_y = None
            else:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_n2g_index, batch_z_t, batch_t,\
                    batch_y, query2g, num_query_cumsum, batch_z = batch_data
                batch_y = batch_y.to(device)

            num_nodes = len(batch_x_n)
            batch_A = dglsp.spmatrix(
                batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_A_n2g = dglsp.spmatrix(
                batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_z_t = batch_z_t.to(device)
            batch_t = batch_t.to(device)
            query2g = query2g.to(device)
            num_query_cumsum = num_query_cumsum.to(device)
            batch_z = batch_z.to(device)

            batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                               batch_rel_level, batch_A_n2g, batch_z_t,
                               batch_t, query2g, num_query_cumsum, batch_y)

            loss = 0
            D = len(batch_pred)
            for d in range(D):
                loss = loss + criterion(batch_pred[d], batch_z[:, d])
            loss /= D

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'node_pred/loss': loss.item()})

    return best_state_dict

@torch.no_grad()
def eval_edge_pred(device, val_loader, model):
    model.eval()
    total_nll = 0
    total_count = 0
    for batch_data in tqdm(val_loader):
        if len(batch_data) == 9:
            batch_edge_index, batch_noisy_edge_index, batch_x_n,\
                batch_abs_level, batch_rel_level, batch_t, batch_query_src,\
                batch_query_dst, batch_label = batch_data
            batch_y = None
        else:
            batch_edge_index, batch_noisy_edge_index, batch_x_n,\
                batch_abs_level, batch_rel_level, batch_t, batch_y,\
                batch_query_src, batch_query_dst, batch_label = batch_data
            batch_y = batch_y.to(device)

        num_nodes = len(batch_x_n)
        batch_A = dglsp.spmatrix(
            torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1),
            shape=(num_nodes, num_nodes)).to(device)
        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_t = batch_t.to(device)
        batch_query_src = batch_query_src.to(device)
        batch_query_dst = batch_query_dst.to(device)
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_t, batch_query_src,
                             batch_query_dst, batch_y)
        batch_nll = -batch_logits.log_softmax(dim=-1)
        batch_num_queries = batch_logits.shape[0]
        batch_nll = batch_nll[
            torch.arange(batch_num_queries).to(device), batch_label]
        total_nll += batch_nll.sum().item()
        total_count += batch_num_queries

    return total_nll / total_count

def main_edge_pred(device, train_set, val_set, model, config, patience):
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              collate_fn=collate_edge_pred,
                              **config['loader'])
    val_loader = DataLoader(val_set,
                            collate_fn=collate_edge_pred,
                            **config['loader'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0
    for epoch in range(config['num_epochs']):
        val_nll = eval_edge_pred(device, val_loader, model)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1
        wandb.log({'edge_pred/epoch': epoch,
                   'edge_pred/val_nll': val_nll,
                   'edge_pred/best_val_nll': best_val_nll,
                   'edge_pred/num_patient_epochs': num_patient_epochs})

        if (patience is not None) and (num_patient_epochs == patience):
            break

        model.train()
        for batch_data in tqdm(train_loader):
            if len(batch_data) == 9:
                batch_edge_index, batch_noisy_edge_index, batch_x_n,\
                    batch_abs_level, batch_rel_level, batch_t,\
                    batch_query_src, batch_query_dst, batch_label = batch_data
                batch_y = None
            else:
                batch_edge_index, batch_noisy_edge_index, batch_x_n,\
                    batch_abs_level, batch_rel_level, batch_t,\
                    batch_y, batch_query_src, batch_query_dst, batch_label = batch_data
                batch_y = batch_y.to(device)

            num_nodes = len(batch_x_n)
            batch_A = dglsp.spmatrix(
                torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1),
                shape=(num_nodes, num_nodes)).to(device)
            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_t = batch_t.to(device)
            batch_query_src = batch_query_src.to(device)
            batch_query_dst = batch_query_dst.to(device)
            batch_label = batch_label.to(device)

            batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                               batch_rel_level, batch_t, batch_query_src,
                               batch_query_dst, batch_y)
            loss = criterion(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'edge_pred/loss': loss.item()})

    return best_state_dict

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

    node_pred_state_dict = main_node_pred(
        device, train_node_pred_dataset, val_node_pred_dataset,
        model.node_pred_model, config['node_pred'], config['general']['patience'])
    model.node_pred_model.load_state_dict(node_pred_state_dict)

    edge_pred_state_dict = main_edge_pred(
        device, train_edge_pred_dataset, val_edge_pred_dataset,
        model.edge_pred_model, config['edge_pred'], config['general']['patience'])
    model.edge_pred_model.load_state_dict(edge_pred_state_dict)

    save_path = f'model_{dataset}_{ts}.pth'
    torch.save({
        'dataset': dataset,
        'node_diffusion_config': node_diffusion_config,
        'edge_diffusion_config': edge_diffusion_config,
        'model_config': model_config,
        'model_state_dict': model.state_dict()
    }, save_path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
