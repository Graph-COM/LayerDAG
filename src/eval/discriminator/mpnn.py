import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from scipy import stats
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTrainer
from .data_utils import collate_fn

__all__ = ['MPNNTrainer']

class MultiEmbedding(nn.Module):
    def __init__(self, num_x_n_cat, hidden_size):
        super().__init__()

        self.emb_list = nn.ModuleList([
            nn.Embedding(num_x_n_cat_i, hidden_size)
            for num_x_n_cat_i in num_x_n_cat
        ])

    def forward(self, x_n_cat):
        if len(x_n_cat.shape) == 1:
            x_n_emb = self.emb_list[0](x_n_cat)
        else:
            x_n_emb = torch.cat([
                self.emb_list[i](x_n_cat[:, i]) for i in range(len(self.emb_list))
            ], dim=1)

        return x_n_emb

class MPNNLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.W = nn.Linear(hidden_size, hidden_size)
        self.W_self = nn.Linear(hidden_size, hidden_size)
        self.W_trans = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h_n):
        if A.nnz == 0:
            h_n_out = self.W_self(h_n)
        else:
            h_n_out = A @ self.W(h_n) + self.W_self(h_n) + A.T @ self.W_trans(h_n)
            # h_n_out = A @ self.W(h_n) + self.W_self(h_n)
        return F.relu(h_n_out)

class MPNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_mpnn_layers,
                 num_x_n_cat=None):
        super().__init__()

        # Backward compatbility with earlier TPU Tile checkpoints
        if num_x_n_cat is None:
            num_x_n_cat = [47]

        self.x_n_emb = MultiEmbedding(num_x_n_cat, hidden_size)
        hidden_size_total = len(num_x_n_cat) * hidden_size
        self.mpnn_layers = nn.ModuleList()
        for _ in range(num_mpnn_layers):
            self.mpnn_layers.append(MPNNLayer(hidden_size_total))

        self.out_proj = nn.Sequential(
            nn.Linear((num_mpnn_layers + 1) * hidden_size_total, hidden_size_total),
            nn.ReLU(),
        )

        self.pred = nn.Sequential(
            nn.Linear(hidden_size_total, hidden_size_total),
            nn.ReLU(),
            nn.Linear(hidden_size_total, 1)
        )

    def forward(self, A, x_n, A_n_g):
        # A = A + A.T
        h_n = self.x_n_emb(x_n)

        h_n_cat = [h_n]
        for layer in self.mpnn_layers:
            h_n = layer(A, h_n)
            h_n_cat.append(h_n)
        h_n = torch.cat(h_n_cat, dim=-1)
        h_n = self.out_proj(h_n)
        h_g = A_n_g @ h_n

        return self.pred(h_g)

class MPNNTrainer(BaseTrainer):
    def __init__(self,
                 hyper_space='tpu_tile',
                 search_priority_increasing=None):
        if hyper_space == 'tpu_tile':
            hyper_space = {
                "lr": [1e-3],
                "num_mpnn_layers": [4],
                "hidden_size": [128],
                "num_x_n_cat": [[47]],
                "num_epochs": [500]
            }
        elif hyper_space == 'hls_dsp':
            hyper_space = {
                "lr": [1e-3],
                "num_mpnn_layers": [1],
                "hidden_size": [32],
                "num_x_n_cat": [[3, 107, 7, 45, 2, 2, 21]],
                "num_epochs": [500]
            }
        elif hyper_space == 'hls_lut':
            hyper_space = {
                "lr": [1e-3],
                "num_mpnn_layers": [3],
                "hidden_size": [64],
                "num_x_n_cat": [[3, 107, 7, 45, 2, 2, 21]],
                "num_epochs": [500]
            }
        elif hyper_space == 'nas_cpu':
            hyper_space = {
                "lr": [1e-3],
                "num_mpnn_layers": [1],
                "hidden_size": [1],
                "num_x_n_cat": [[9, 2, 5, 4, 5, 4, 4, 5, 4, 3, 3, 5, 4, 5]],
                "num_epochs": [500]
            }

        if search_priority_increasing is None:
            search_priority_increasing = ["lr", "num_mpnn_layers", "hidden_size", "num_x_n_cat", "num_epochs"]

        super().__init__(hyper_space=hyper_space,
                         search_priority_increasing=search_priority_increasing)

    def preprocess(self, edge_index, x_n, num_nodes_cumsum, label):
        N = int(num_nodes_cumsum[-1])
        A = dglsp.spmatrix(edge_index, shape=(N, N)).to(self.device)
        x_n = x_n.to(self.device)
        label = label.to(self.device)

        batch_size = len(num_nodes_cumsum) - 1
        src = []
        dst = []
        for i in range(batch_size):
            num_nodes_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
            dst.extend([i] * num_nodes_i)
            src.extend(list(range(num_nodes_cumsum[i], num_nodes_cumsum[i + 1])))

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        n_g_edge_index = torch.stack([dst, src])
        A_n_g = dglsp.spmatrix(n_g_edge_index, shape=(len(label), len(x_n))).to(self.device)

        return A, x_n, A_n_g, label

    def train_epoch(self, train_loader, model, optimizer):
        model.train()
        for batch_data in train_loader:
            batch_edge_index, batch_x_n, num_nodes_cumsum, batch_label = batch_data
            A, x_n, A_n_g, label = self.preprocess(
                batch_edge_index, batch_x_n, num_nodes_cumsum, batch_label)
            pred = model(A, x_n, A_n_g)
            loss = F.smooth_l1_loss(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def eval_epoch(self, data_loader, model, spearman=False):
        model.eval()

        full_label = []
        full_pred = []
        for batch_data in data_loader:
            batch_edge_index, batch_x_n, num_nodes_cumsum, batch_label = batch_data
            A, x_n, A_n_g, label = self.preprocess(
                batch_edge_index, batch_x_n, num_nodes_cumsum, batch_label)
            pred = model(A, x_n, A_n_g)
            full_label.append(label.cpu())
            full_pred.append(pred.cpu())

        full_label = torch.cat(full_label).squeeze(-1)
        full_pred = torch.cat(full_pred).squeeze(-1)
        coef = stats.pearsonr(full_label.numpy(), full_pred.numpy())[0]

        mae = F.l1_loss(full_label, full_pred)

        if spearman:
            spearman_coef = stats.spearmanr(full_label.numpy(), full_pred.numpy())[0]

            return coef, spearman_coef, mae
        else:
            return coef, mae

    def fit_trial(self,
                  train_set,
                  val_set,
                  hidden_size,
                  num_mpnn_layers,
                  num_x_n_cat,
                  lr,
                  num_epochs,
                  batch_size=256,
                  num_workers=0,
                  patience_limit=100):
        torch.set_num_threads(20)
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  shuffle=True)

        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=collate_fn)

        model = MPNN(num_x_n_cat=num_x_n_cat,
                     hidden_size=hidden_size,
                     num_mpnn_layers=num_mpnn_layers).to(self.device)
        optimizer = Adam(model.parameters(), lr=lr)

        best_val_coef = float('-inf')
        patience = 0
        best_model_state_dict = deepcopy(model.state_dict())
        for epoch in tqdm(range(num_epochs)):
            self.train_epoch(train_loader, model, optimizer)
            val_coef, val_mae = self.eval_epoch(val_loader, model)
            if val_coef > best_val_coef:
                patience = 0
                best_val_coef = val_coef
                best_model_state_dict = deepcopy(model.state_dict())
            else:
                patience += 1

            print(f'Epoch {epoch} | Best Val coef: {best_val_coef:.4f} | Val coef: {val_coef:.4f} | Val mae: {val_mae:.4f}')

            if patience == patience_limit:
                break

        model.load_state_dict(best_model_state_dict)
        return best_val_coef, model

    def fit(self,
            train_set,
            val_set):
        config_list = self.get_config_list()

        best_coef = float('-inf')
        with tqdm(config_list) as tconfig:
            tconfig.set_description("Training MPNN discriminator")

            for config in tconfig:
                trial_coef, trial_model = self.fit_trial(
                    train_set, val_set, **config)
                if trial_coef > best_coef:
                    best_coef = trial_coef
                    best_model = trial_model
                    best_model_config = {
                        "hidden_size": config["hidden_size"],
                        "num_mpnn_layers": config["num_mpnn_layers"],
                        "num_x_n_cat": config["num_x_n_cat"]
                    }
                tconfig.set_postfix(pearson=best_coef)

                if trial_coef == 1.0:
                    break

        self.model = best_model
        self.best_model_config = best_model_config

    def predict(self,
                test_set,
                batch_size=256,
                num_workers=0):
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
        return self.eval_epoch(test_loader, self.model, spearman=True)

    def load_model(self, model_path):
        cpt = torch.load(model_path)
        model = MPNN(**cpt["model_config"]).to(self.device)
        model.load_state_dict(cpt["model_state_dict"])
        self.model = model
