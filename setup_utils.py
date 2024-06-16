import dgl
import numpy as np
import pydantic
import random
import torch
import yaml

from typing import Optional

def set_seed(seed=0):
    if seed is None:
        return

    dgl.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataLoaderYaml(pydantic.BaseModel):
    batch_size: int
    num_workers: int

class BiMPNNYaml(pydantic.BaseModel):
    x_n_emb_size: int
    pe_emb_size: Optional[int] = 0
    y_emb_size: Optional[int] = 0
    num_mpnn_layers: int
    pool: Optional[str] = None
    pe: Optional[str] = None

class OptimizerYaml(pydantic.BaseModel):
    lr: float
    amsgrad: bool

class NodeCountYaml(pydantic.BaseModel):
    loader: DataLoaderYaml
    model: BiMPNNYaml
    num_epochs: int
    optimizer: OptimizerYaml

class NodePredictorYaml(pydantic.BaseModel):
    t_emb_size: int
    out_hidden_size: int
    num_transformer_layers: int
    num_heads: int
    dropout: float

class NodePredYaml(pydantic.BaseModel):
    T: int
    loader: DataLoaderYaml
    num_epochs: int
    graph_encoder: BiMPNNYaml
    predictor: NodePredictorYaml
    optimizer: OptimizerYaml

class EdgePredictorYaml(pydantic.BaseModel):
    t_emb_size: int
    out_hidden_size: int

class EdgePredYaml(pydantic.BaseModel):
    T: int
    loader: DataLoaderYaml
    num_epochs: int
    graph_encoder: BiMPNNYaml
    predictor: EdgePredictorYaml
    optimizer: OptimizerYaml

class GeneralYaml(pydantic.BaseModel):
    dataset: str
    conditional: bool
    patience: Optional[int] = None

class LayerDAGYaml(pydantic.BaseModel):
    general: GeneralYaml
    node_count: NodeCountYaml
    node_pred: NodePredYaml
    edge_pred: EdgePredYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    return LayerDAGYaml(**yaml_data).model_dump()
