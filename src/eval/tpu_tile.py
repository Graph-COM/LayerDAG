import os

from .discriminator import BaseEvaluator, MPNNTrainer
from ..dataset import load_dataset

__all__ = ['TPUTileEvaluator']

class TPUTileEvaluator:
    def __init__(self):
        train_set, val_set, test_set = load_dataset('tpu_tile')

        cpt_path = "tpu_tile_cpts"
        os.makedirs(cpt_path, exist_ok=True)
        self.mpnn_evaluator = BaseEvaluator(MPNNTrainer,
                                            os.path.join("tpu_tile_cpts", "mpnn.pth"),
                                            train_set,
                                            val_set,
                                            test_set)

    def eval(self, train_syn_set, val_syn_set):
        self.mpnn_evaluator.eval(train_syn_set, val_syn_set)
        self.summary()

    def summary(self):
        print('\n')
        print('MPNN Discriminator')
        print('------------------')
        print('\n')

        print('Real')
        print('------------------')
        print(f'Pearson Coeff: {self.mpnn_evaluator.real_pearson_coeff}')
        print('\n')

        print('Synthetic')
        print('------------------')
        print(f'Pearson Coeff: {self.mpnn_evaluator.syn_pearson_coeff}')
