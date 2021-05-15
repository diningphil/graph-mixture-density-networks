import os
import sys
import json
import shutil
import torch
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import scipy.io
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
stderr_tmp = sys.stderr
null = open(os.devnull, 'w')
sys.stderr = null
from dgl.data.utils import load_graphs
sys.stderr = stderr_tmp


class DatasetInterface:

    name = None

    @property
    def dim_node_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, use_node_attr=False, use_edge_attr=False, cleaned=False):
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)

        if 'alchemy_full' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)

            # Normalize all target variables (for training stability purposes)
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
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class BarabasiAlbertDataset(InMemoryDataset, DatasetInterface):

    def __init__(self, root, name, size, connectivity, transform=None, pre_transform=None):
        self.no_graph_samples = 100
        self.number_of_simulations = 1000
        self.M = [2, 5, 10, 20]
        self.SIZES = [10, 50, 100, 200, 500, 1000]

        self.size = size
        self.name = f'{name}_{size}_{connectivity}'
        self.m = [int(c) for c in connectivity.split("@")]

        super(BarabasiAlbertDataset, self).__init__(root, transform, pre_transform)
        # print("Loading dataset..")
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print("Dataset Loaded.")

    @property
    def raw_file_names(self):
        files_list = []
        for no_edge in self.M:
            for graph_size in self.SIZES:
                if no_edge < graph_size:
                    graphs_folder = Path(self.root) / Path('raw', f'graphs_size{graph_size}_noedge{no_edge}')

                    for graph_sample in range(self.no_graph_samples):
                        json_filepath = graphs_folder/ Path(f'data_{graph_sample}.json')
                        files_list.append(json_filepath)
                        graph_filename = graphs_folder / Path(f'sample{graph_sample}.bin')
                        files_list.append(graph_filename)
        return files_list

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def download(self):
        pass
        #print(f'Skipping download to {self.raw_dir}')

    def process(self):

        # Read data into huge `Data` list.
        data_list = []

        # The order of this for loop is closely related to the SIRSplitter data splitter
        for graph_sample in range(self.no_graph_samples):
            print(f'Processing graph {graph_sample+1} of {self.no_graph_samples}')

            for m in self.m:

                graphs_folder = Path(self.root) / Path('raw', f'graphs_size{self.size}_noedge{m}')
                # print(f'Processing graphs of size {self.size} and parameter M {m}')

                json_filepath = str(graphs_folder / Path(f'data_{graph_sample}.json'))
                graph_filename = str(graphs_folder / Path(f'sample{graph_sample}.bin'))

                # Load using DGL library
                graph = load_graphs(str(graph_filename))
                graph = graph[0][0].to_networkx()

                # print('Loading', json_filepath)
                with open(json_filepath, 'r') as f:
                    simul_data = json.load(f)

                simulations = simul_data['graph_samples'][0]['simulations']  # only one element in this list
                df = pd.DataFrame(simulations)

                for initial_probability_of_infection in [0.01, 0.05, 0.1]:   # available as well. stick to this for now.
                    filtered = df[df['init_infection_prob']==initial_probability_of_infection]
                    beta = filtered['beta'].to_numpy()
                    gamma = filtered['gamma'].to_numpy()
                    infected = filtered['total_infected']
                    first_infected = torch.tensor(filtered['first_infected'].tolist()).unsqueeze(2).float()

                    assert len(infected) == self.number_of_simulations
                    for i, y in enumerate(infected):
                        if i == 100:  # take a maximum of 100 simulations per graph (we have 1k just in case)
                            break
                        # Create PyG Data object from networkx DiGraph
                        graph_data = from_networkx(graph)
                        # Add 1 as node features
                        graph_data.x = torch.cat((torch.ones(self.size).unsqueeze(1),
                                              torch.tensor([float(beta[i])/float(gamma[i]), float(beta[i]),
                                              float(gamma[i])]).unsqueeze(0).repeat(self.size, 1),
                                              first_infected[i, :]), dim=1)
                        graph_data.y = torch.tensor([y]).unsqueeze(1)

                        data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def dim_node_features(self):
        return self.data.x.shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1


class ErdosRenyiDataset(InMemoryDataset, DatasetInterface):

    def __init__(self, root, name, size, connectivity, transform=None, pre_transform=None):
        self.no_graph_samples = 100
        self.number_of_simulations = 1000
        self.M = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        self.SIZES = [10, 50, 100, 200, 500, 1000]

        self.size = size
        self.name = f'{name}_{size}_{connectivity}'
        self.m = [float(c) for c in connectivity.split("@")]

        super(ErdosRenyiDataset, self).__init__(root, transform, pre_transform)
        # print("Loading dataset..")
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print("Dataset Loaded.")

    @property
    def raw_file_names(self):
        files_list = []
        for no_edge in self.M:
            for graph_size in self.SIZES:
                if no_edge < graph_size:
                    graphs_folder = Path(self.root) / Path('raw', f'graphs_size{graph_size}_p{no_edge}')

                    for graph_sample in range(self.no_graph_samples):
                        json_filepath = graphs_folder/ Path(f'data_{graph_sample}.json')
                        files_list.append(json_filepath)
                        graph_filename = graphs_folder / Path(f'sample{graph_sample}.bin')
                        files_list.append(graph_filename)
        return files_list

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def download(self):
        pass
        #print(f'Skipping download to {self.raw_dir}')

    def process(self):

        # Read data into huge `Data` list.
        data_list = []

        # The order of this for loop is closely related to the SIRSplitter data splitter
        for graph_sample in range(self.no_graph_samples):
            print(f'Processing graph {graph_sample+1} of {self.no_graph_samples}')

            for m in self.m:

                graphs_folder = Path(self.root) / Path('raw', f'graphs_size{self.size}_p{m}')
                # print(f'Processing graphs of size {self.size} and parameter M {m}')

                json_filepath = str(graphs_folder / Path(f'data_{graph_sample}.json'))
                graph_filename = str(graphs_folder / Path(f'sample{graph_sample}.bin'))

                # Load using DGL library
                graph = load_graphs(str(graph_filename))
                graph = graph[0][0].to_networkx()

                # print('Loading', json_filepath)
                with open(json_filepath, 'r') as f:
                    simul_data = json.load(f)

                simulations = simul_data['graph_samples'][0]['simulations']  # only one element in this list
                df = pd.DataFrame(simulations)

                for initial_probability_of_infection in [0.01, 0.05, 0.1]:   # available as well. stick to this for now.
                    filtered = df[df['init_infection_prob']==initial_probability_of_infection]
                    beta = filtered['beta'].to_numpy()
                    gamma = filtered['gamma'].to_numpy()
                    infected = filtered['total_infected']
                    first_infected = torch.tensor(filtered['first_infected'].tolist()).unsqueeze(2).float()

                    assert len(infected) == self.number_of_simulations
                    for i, y in enumerate(infected):
                        if i == 100:  # take a maximum of 100 simulations per graph (we have 1k just in case)
                            break
                        # Create PyG Data object from networkx DiGraph
                        graph_data = from_networkx(graph)
                        # Add 1 as node features
                        graph_data.x = torch.cat((torch.ones(self.size).unsqueeze(1),
                                              torch.tensor([float(beta[i])/float(gamma[i]), float(beta[i]),
                                              float(gamma[i])]).unsqueeze(0).repeat(self.size, 1),
                                              first_infected[i, :]), dim=1)
                        graph_data.y = torch.tensor([y]).unsqueeze(1)

                        data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def dim_node_features(self):
        return self.data.x.shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1
