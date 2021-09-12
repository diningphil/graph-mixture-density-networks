# Graph Mixture Density Networks (GMDN)
![](https://github.com/diningphil/graph-mixture-density-networks/raw/main/images/gmdn.png)

## Summary
The Graph Mixture Density Network is a supervised learning framework to model multimodal output distributions that are conditioned on arbitrary graphs.

The library includes data and scripts to reproduce the experiments reported in the paper. If you happen to use or modify this code, please remember to cite us:

[*Federico Errica , Davide Bacciu, Alessio Micheli: Graph Mixture Density Networks. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:3025-3035, 2021.*](https://arxiv.org/abs/2012.03085)

## Data and Splits
To keep the memory footprint low, we do not provide the raw results of the simulations. We instead release the processed dataset, which contains all the information used to run the simulations and the final results (target labels). However, through the notebooks in `GMDN_NOTEBOOKS`, it is possible to perform random simulations of the stochastic SIR model to create datasets of varying dimensions. This repo also provides Pytorch Geometric classes for all datasets
so that a user can easily load the data in memory and then convert it in some other form of interest.

The splits used in our experiments are available in the `GMDN_SPLITS` folder. These are simple dictionaries that can be loaded using Pytorch.

Data for the Barabasi-Albert and Erdos-Renyi experiments can be downloaded at [this link](https://www.dropbox.com/sh/cv6blu0w3pqevxq/AAAJFC2wpLuDVfe75qAoD7hga?dl=0). However, we are working to provide a more reliable link in the future.
The file contains the datasets in highly compressed form, for a total of 3.7GB. Note: the fully uncompressed folder (all datasets) will take 155GB of space.

## Requirements

- [PyDGN](https://github.com/diningphil/PyDGN)  (we used PyDGN 0.5.1)
- [DGL](https://www.dgl.ai) 0.4.0

### Preprocess your dataset
Use

    python build_dataset.py --config-file [your data config file]

For example

    python build_dataset.py --config-file GMDN_DATA_CONFIG/config_alchemy_full.yml

### Launch an experiment in debug mode
Use

    python launch_experiment.py --config-file [your exp. config file] --splits-folder [the splits MAIN folder] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --max-cpus [max cpu parallelism] --max-gpus [max gpu parallelism] --gpus-per-task [how many gpus to allocate for each job] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]

For example (uses GPU - to use CPU only, modify the config file accordingly and set `--max-gpus 0`)

    python launch_experiment.py --config-file GMDN_MODEL_CONFIG/PROTEINS/GMDN/config_GMDN_alchemy_full.yml --splits-folder GMDN_SPLITS/ --data-splits GMDN_SPLITS/alchemy_full/alchemy_full_outer1_inner1.splits --data-root DATA --dataset-name alchemy_full --dataset-class gmdn_dataset.TUDatasetInterface --max-cpus 1 --max-gpus 1 --final-training-runs 10 --result-folder RESULTS

(Note: it is very important that you use the `TUDatasetInterface` implementation provided in `gmdn_dataset`, as it adds some preprocessing to the chemical datasets)

To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.

## Contributing
**This research software is provided as-is**.
If you find a bug or have generic/technical questions, please email us.

## License:
This repo is GPL 3.0 licensed, as written in the LICENSE file.
