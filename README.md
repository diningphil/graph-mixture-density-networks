# Graph Mixture Density Networks
![](https://github.com/diningphil/graph-mixture-density-networks/raw/main/images/gmdn.png)

## Summary
The Graph Mixture Density Network is a supervised learning framework to model multimodal output distributions that are conditioned on arbitrary graphs.

If you happen to use or modify this code, please remember to cite us:

[*Federico Errica , Davide Bacciu, Alessio Micheli: Graph Mixture Density Networks. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:3025-3035, 2021.*](https://arxiv.org/abs/2012.03085)

### General Usage

This repo builds upon [PyDGN](https://github.com/diningphil/PyDGN), a framework to easily develop and test new DGNs.
See how to construct your dataset and then train your model there.

**This repo assumes PyDGN 1.0.5 is used.** Compatibility with future versions is likely but not guaranteed, e.g., custom metrics need to be slightly modified starting from PyDGN 1.2.0.

#### Example on `alchemy_full`

    pydgn-dataset --config-file DATA_CONFIGS/config_alchemy_full.yml
    pydgn-train  --config-file MODEL_CONFIGS/config_alchemy_full.yml 

## Reproducibility of ICML 2021 paper

See [this GMDN release](https://github.com/diningphil/graph-mixture-density-networks/releases/tag/v1-ICML).

It relies on
- [PyDGN](https://github.com/diningphil/PyDGN)  (we used PyDGN 0.5.1, see "general usage" for more recent versions.)
- [DGL](https://www.dgl.ai) 0.4.0

#### Data and Splits
To keep the memory footprint low, we do not provide the raw results of the simulations. We instead release the processed dataset, which contains all the information used to run the simulations and the final results (target labels). However, through the notebooks in `GMDN_NOTEBOOKS`, it is possible to perform random simulations of the stochastic SIR model to create datasets of varying dimensions. This repo also provides Pytorch Geometric classes for all datasets
so that a user can easily load the data in memory and then convert it in some other form of interest.

The splits used in our experiments are available in the `GMDN_SPLITS` folder. These are simple dictionaries that can be loaded using Pytorch.

Data for the Barabasi-Albert and Erdos-Renyi experiments can be downloaded at [this link](https://www.dropbox.com/sh/cv6blu0w3pqevxq/AAAJFC2wpLuDVfe75qAoD7hga?dl=0). However, we are working to provide a more reliable link in the future.
The file contains the datasets in highly compressed form, for a total of 3.7GB. Note: the fully uncompressed folder (all datasets) will take 155GB of space.


#### Preprocess your dataset

    python build_dataset.py --config-file [your data config file]

#### Launch an experiment

    python launch_experiment.py --config-file [your exp. config file] --splits-folder [the splits MAIN folder] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --max-cpus [max cpu parallelism] --max-gpus [max gpu parallelism] --gpus-per-task [how many gpus to allocate for each job] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]

(Note: it is very important that you use the `TUDatasetInterface` implementation provided in `gmdn_dataset`, as it adds some preprocessing to the chemical datasets)

To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.
