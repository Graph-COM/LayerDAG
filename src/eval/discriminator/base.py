import itertools
import os
import torch

__all__ = ['BaseEvaluator']

class BaseTrainer:
    def __init__(self,
                 hyper_space,
                 search_priority_increasing):
        """Base class for training a discriminative model.

        Parameters
        ----------
        search_priority_increasing : list of str
            The priority of hyperparameters to search, from lowest to highest.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.hyper_space = hyper_space
        self.search_priority_increasing = search_priority_increasing

    def get_config_list(self):
        vals = [self.hyper_space[k] for k in self.search_priority_increasing]

        config_list = []
        for items in itertools.product(*vals):
            items_dict = dict(zip(self.search_priority_increasing, items))
            config_list.append(items_dict)

        return config_list

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.best_model_config
        }, model_path)

class BaseEvaluator:
    def __init__(self,
                 Trainer,
                 model_path,
                 real_train_set,
                 real_val_set,
                 real_test_set):
        self.Trainer = Trainer
        self.real_test_set = real_test_set

        self.model_real = Trainer()
        if (model_path is not None) and (os.path.exists(model_path)):
            self.model_real.load_model(model_path)
        else:
            self.model_real.fit(real_train_set,
                                real_val_set)
            if model_path is not None:
                self.model_real.save_model(model_path)

        self.real_pearson_coeff, self.real_spearman_coeff, self.real_mae = self.model_real.predict(real_test_set)

    def eval(self, train_syn_set, val_syn_set):
        model_syn = self.Trainer()
        model_syn.fit(train_syn_set, val_syn_set)
        self.syn_pearson_coeff, self.syn_spearman_coeff, self.syn_mae = model_syn.predict(self.real_test_set)
