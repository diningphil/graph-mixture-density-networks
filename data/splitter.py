import torch
import random
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected, to_dense_adj, add_self_loops
from experiment.experiment import s2c


class Fold:
    """
    Simple class that stores training and validation/test indices
    """
    def __init__(self, train_idxs, val_idxs=None, test_idxs=None):
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

class InnerFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and validation indices (model selection)
    """
    def todict(self):
        return {"train": self.train_idxs, "val": self.val_idxs}


class OuterFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and test indices (risk assessment)
    """
    def todict(self):
        return {"train": self.train_idxs, "val": self.val_idxs, "test": self.test_idxs}


class NoShuffleTrainTestSplit:

    def __init__(self, test_ratio):
        self.test_ratio = test_ratio

    def split(self, idxs, y=None):
        n_samples = len(idxs)
        n_test = int(n_samples*self.test_ratio)
        n_train = n_samples - n_test
        train_idxs = np.arange(n_train)
        test_idxs = np.arange(n_train, n_train + n_test)
        return [(train_idxs, test_idxs)]


class Splitter:
    """
    Class that generates the splits at dataset creation time.
    """

    @classmethod
    def load(cls, path):
        """
        Loads the splits from disk
        :param path: the path of the yaml file with the splits
        :return: a Splitter object
        """
        splits = torch.load(path)

        splitter_classname = splits.get("splitter_class", "Splitter")
        # v0.4.0, backward compatibility with 0.3.2
        if 'dataset.' in splitter_classname:
            splitter_classname.replace('datasets.', 'data.')
        splitter_class = s2c('data.splitter.' + splitter_classname)

        splitter_args = splits.get("splitter_args")
        splitter = splitter_class(**splitter_args)

        assert splitter.n_outer_folds == len(splits["outer_folds"])
        assert splitter.n_inner_folds == len(splits["inner_folds"][0])

        for fold_data in splits["outer_folds"]:
            # v0.4.0, backward compatibility with 0.3.2
            if not hasattr(fold_data, "val"):
                fold_data["val"] = None
            splitter.outer_folds.append(OuterFold(fold_data["train"], val_idxs=fold_data["val"], test_idxs=fold_data["test"]))

        for inner_split in splits["inner_folds"]:
            inner_split_data = []
            for fold_data in inner_split:
                # v0.4.0, backward compatibility with 0.3.2
                if not hasattr(fold_data, "val") and "val" not in fold_data:
                    fold_data["val"] = None
                inner_split_data.append(InnerFold(fold_data["train"], val_idxs=fold_data["val"]))
            splitter.inner_folds.append(inner_split_data)

        return splitter

    def __init__(self, n_outer_folds, n_inner_folds, seed, stratify, shuffle, val_ratio=0.1, test_ratio=0.1):
        """
        Initializes the splitter
        :param n_outer_folds: number of outer folds (risk assessment). 1 means hold-out, >1 means k-fold
        :param n_inner_folds: number of inner folds (model selection). 1 means hold-out, >1 means k-fold
        :param seed: random seed for reproducibility (on the same machine)
        :param stratify: whether to apply stratification or not (should be true for classification tasks)
        :param shuffle: whether to apply shuffle or not
        :param val_ratio: percentage of validation set for hold_out model selection
        :param test_ratio: percentage of test set for hold_out model assessment
        """
        self.outer_folds = []
        self.inner_folds = []
        self.processed = False
        self.stratify = stratify
        self.shuffle= shuffle

        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        '''
        self.kwargs = kwargs
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        '''
    def _get_splitter(self, n_splits, stratified, test_ratio):
        if n_splits == 1:
            if not self.shuffle:
                assert stratified == False, "Stratified not implemented when shuffle is False"
                splitter = NoShuffleTrainTestSplit(test_ratio=test_ratio)
            else:
                if stratified:
                    splitter = StratifiedShuffleSplit(n_splits, test_size=test_ratio, random_state=self.seed)
                else:
                    splitter = ShuffleSplit(n_splits, test_size=test_ratio, random_state=self.seed)
        elif n_splits > 1:
            if stratified:
                splitter = StratifiedKFold(n_splits, shuffle=self.shuffle, random_state=self.seed)
            else:
                splitter = KFold(n_splits, shuffle=self.shuffle, random_state=self.seed)
        else:
            raise ValueError(f"'n_splits' must be >=1, got {n_splits}")

        return splitter

    def split(self, dataset, targets=None):
        """
        Computes the splits. The outer split does not include validation (can be extracted from the training set if needed)
        :param dataset: the Dataset object
        :param targets: targets used for stratification
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        idxs = range(len(dataset))

        if not self.processed:

            stratified = self.stratify
            outer_idxs = np.array(idxs)

            outer_splitter = self._get_splitter(
                n_splits=self.n_outer_folds,
                stratified=stratified,
                test_ratio=self.test_ratio)  # This is the true test (outer test)

            for train_idxs, test_idxs in outer_splitter.split(outer_idxs, y=targets):

                assert set(train_idxs) == set(outer_idxs[train_idxs])
                assert set(test_idxs) == set(outer_idxs[test_idxs])

                inner_fold_splits = []
                inner_idxs = outer_idxs[train_idxs]  # equals train_idxs because outer_idxs was ordered
                inner_targets = targets[train_idxs] if targets is not None else None

                inner_splitter = self._get_splitter(
                    n_splits=self.n_inner_folds,
                    stratified=stratified,
                    test_ratio=self.val_ratio)  # The inner "test" is, instead, the validation set

                for inner_train_idxs, inner_val_idxs in inner_splitter.split(inner_idxs, y=inner_targets):
                    inner_fold = InnerFold(train_idxs=inner_idxs[inner_train_idxs].tolist(), val_idxs=inner_idxs[inner_val_idxs].tolist())
                    inner_fold_splits.append(inner_fold)
                self.inner_folds.append(inner_fold_splits)

                # Obtain outer val from outer train in an holdout fashion
                outer_val_splitter = self._get_splitter(n_splits=1, stratified=stratified, test_ratio=self.val_ratio)  # Use val ratio to compute outer val
                outer_train_idxs, outer_val_idxs = list(outer_val_splitter.split(inner_idxs, y=inner_targets))[0]

                # False if empty
                assert not bool(set(inner_train_idxs) & set(inner_val_idxs) & set(test_idxs))
                assert not bool(set(inner_idxs[inner_train_idxs]) & set(inner_idxs[inner_val_idxs]) & set(test_idxs))
                assert not bool(set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs))
                assert not bool(set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs))
                assert not bool(set(inner_idxs[outer_train_idxs]) & set(inner_idxs[outer_val_idxs]) & set(test_idxs))

                np.random.shuffle(outer_train_idxs)
                np.random.shuffle(outer_val_idxs)
                np.random.shuffle(test_idxs)
                outer_fold = OuterFold(train_idxs=inner_idxs[outer_train_idxs].tolist(), val_idxs=inner_idxs[outer_val_idxs].tolist(), test_idxs=outer_idxs[test_idxs].tolist())
                self.outer_folds.append(outer_fold)

            self.processed = True

    def _splitter_args(self):
        return {
                "n_outer_folds": self.n_outer_folds,
                "n_inner_folds": self.n_inner_folds,
                "seed": self.seed,
                "stratify": self.stratify,
                "shuffle": self.shuffle,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                }

    def save(self, path):
        """
        Saves the split into a yaml file
        :param path: filepath where to save the object
        """
        print("Saving splits on disk...")
        savedict = {"seed": self.seed,
                    "splitter_class": self.__class__.__name__,
                    "splitter_args": self._splitter_args()
                    }

        savedict["outer_folds"] = [o.todict() for o in self.outer_folds]
        savedict["inner_folds"] = []
        for inner_split in self.inner_folds:
            savedict["inner_folds"].append([i.todict() for i in inner_split])
        torch.save(savedict, path)
        print("Done.")


class SIRSplitter(Splitter):

    def __init__(self, n_outer_folds=1, n_inner_folds=1, seed=42, **kwargs):
        """
        Initializes the splitter
        :param n_outer_folds: number of outer folds (risk assessment). Must be 1 (holdout)
        :param n_inner_folds: number of inner folds (model selection). Must be 1 (holdout)
        :param seed: random seed for reproducibility
        """

        self.stratify = False
        self.shuffle = False
        assert n_outer_folds == 1, "Only one outer fold is allowed"
        assert n_inner_folds == 1, "Only one inner fold is allowed"

        super().__init__(n_outer_folds, n_inner_folds, seed, self.stratify, self.shuffle)

    def split(self, dataset, targets=None, test_ratio=0.1):
        """
        Computes the splits
        :param dataset: the Dataset object
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        idxs = range(len(dataset))

        if not self.processed:

            outer_idxs = np.array(idxs)

            train_idxs, test_idxs = train_test_split(outer_idxs, test_ratio=test_ratio, shuffle=False)

            inner_fold_splits = []

            inner_train_idxs, inner_val_idxs = train_test_split(train_idxs, test_ratio=test_ratio, shuffle=False)
            np.random.shuffle(inner_train_idxs)

            inner_fold = InnerFold(inner_train_idxs.tolist(), inner_val_idxs.tolist())
            inner_fold_splits.append(inner_fold)
            self.inner_folds.append(inner_fold_splits)

            np.random.shuffle(train_idxs)
            outer_fold = OuterFold(train_idxs.tolist(), test_idxs.tolist())
            self.outer_folds.append(outer_fold)

            self.processed = True
