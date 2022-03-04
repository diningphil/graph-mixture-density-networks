from pydgn.data.dataset import TUDatasetInterface


class AlchemyZINCDatasetInterface(TUDatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(root, name, transform, pre_transform, pre_filter, **kwargs)

        if 'alchemy_full' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)

            # Normalize all target variables (just for stability purposes)
            mean = self.data.y.mean(0).unsqueeze(0)
            std = self.data.y.std(0).unsqueeze(0)
            self.data.y = (self.data.y - mean) / std

        if 'ZINC_full' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        if 'alchemy_full' in self.name:
            return self.data.y.shape[1]
        if 'ZINC_full' in self.name:
            return self.data.y.shape[1]
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()