import torch
import torch.nn as nn

__all__ = [
    'LayerDAG'
]

class LayerDAG(nn.Module):
    def __init__(self,
                 device,
                 num_x_n_cat,
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 max_level=None):
        """
        Parameters
        ----------
        num_x_n_cat :
            Case1: int
            Case2: torch.LongTensor of shape (num_feats)
        """
        super().__init__()

        if isinstance(num_x_n_cat, int):
            num_x_n_cat = torch.LongTensor([num_x_n_cat])
