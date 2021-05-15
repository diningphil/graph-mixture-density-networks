import json
from copy import deepcopy
from pathlib import Path
from experiment.experiment import s2c
import evaluation.util

class RandomSearch:
    """ This class performs a random search for hyper-parameters optimisation over the search spaces defined in the configuration file """


    def __init__(self, data_root, dataset_class, dataset_name, **configs_dict):
        """
        Initializes the RandomSearch object by looking for specific keys in the dictionary, namely 'experiment', 'device',
        'model', 'dataset-getter' and 'higher_results_are_better'. The configuration dictionary should have
        a field named 'random' in which all possible hyper-parameters are listed (see examples)
        :param data_root: the root directory in which the dataset is stored
        :param dataset_class: one of the classes in datasets.datasets that specifies how to process the data
        :param dataset_name: the name of the dataset
        :param configs_dict: the configuration dictionary
        """
        self.configs_dict = configs_dict
        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.experiment = self.configs_dict['experiment']
        self.higher_results_are_better = self.configs_dict['higher_results_are_better']
        self.log_every = self.configs_dict['log_every']
        self.device = self.configs_dict['device']
        self.num_dataloader_workers = self.configs_dict['num_dataloader_workers']
        self.pin_memory = self.configs_dict['pin_memory']
        self.model = self.configs_dict['model']
        self.dataset_getter = self.configs_dict['dataset-getter']
        self.num_samples = self.configs_dict['num_samples']

        # For continual learning tasks
        # - Reharsal
        self.n_tasks = self.configs_dict.get('n_tasks', None)
        self.n_rehearsal_patterns_per_task = self.configs_dict.get('n_rehearsal_patterns_per_task', None)

        # This MUST be called at the END of the init method!
        self.hparams = self._gen_configs()

    def _gen_configs(self):
        '''
        Takes a dictionary of key:list pairs and computes possible hyper-parameter configurations.
        :return: A list of possible configurations
        '''
        configs = [cfg for cfg in self._gen_helper(self.configs_dict['random'])]
        for cfg in configs:
            cfg.update({"dataset": self.dataset_name,
                        "dataset_getter": self.dataset_getter,
                        "dataset_class": self.dataset_class,
                        "data_root": self.data_root,
                        "model": self.model,
                        "device": self.device,
                        "num_dataloader_workers": self.num_dataloader_workers,
                        "pin_memory": self.pin_memory,
                        "experiment": self.experiment,
                        "higher_results_are_better": self.higher_results_are_better,
                        "log_every": self.log_every,
                        "n_tasks": self.n_tasks,
                        "n_rehearsal_patterns_per_task": self.n_rehearsal_patterns_per_task})

        return configs

    def _gen_helper(self, cfgs_dict):
        keys = cfgs_dict.keys()
        param = list(keys)[0]

        for _ in range(self.num_samples):
            result = {}
            for key, values in cfgs_dict.items():
                # BASE CASE: key is associated to an atomic value
                if type(values) in [str, int, float, bool, None]:
                    result[key] = values
                # DICT CASE: call _dict_helper on this dict
                elif type(values) == dict:
                    result[key] = self._dict_helper(deepcopy(values))

            yield deepcopy(result)


    def _dict_helper(self, configs):
        if 'sample_method' in configs:
            return self._sampler_helper(configs)

        for key, values in configs.items():
            if type(values) == dict:
                configs[key] = self._dict_helper(configs[key])

        return configs

    def _sampler_helper(self, configs):
        method, args = configs['sample_method'], configs['args']
        sampler = s2c(f'evaluation.util.{method}')

        sample = sampler(*args)

        if type(sample) == dict:
            return self._dict_helper(sample)

        return sample

    def __iter__(self):
        return iter(self.hparams)

    def __len__(self):
        return len(self.hparams)

    def __getitem__(self, index):
        return self.hparams[index]

    @property
    def exp_name(self):
        return f"{self.model.split('.')[-1]}_{self.dataset_name}"

    @property
    def num_configs(self):
        return len(self)
