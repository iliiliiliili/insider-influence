[![DOI](https://zenodo.org/badge/513898679.svg)](https://zenodo.org/badge/latestdoi/513898679)

# Implementation of [Variational Graph Convolutional Neural Networks](https://arxiv.org/abs/2507.01699) for financial data analysis.

## Installation

Use `bash -i install.sh` to install with Conda and module. Edit to select different python and cuda versions or to install without module/Conda tools.

## Usage

### Train one

```
python train.py --name=usual --dataset_folder=data --networks=gcn --path=. --results_folder=results --models_folder=models
```

### Train modeling

```
python modeling.py base_experiment_all --devices=8 --processes_per_device=3",
```

```
python modeling.py train_for_relaibility_diagrams --results_folder=results-reliability --plots_folder=plots-reliability-with-uncertainty-extra",
```

### Test a specific model

```
python train.py test --name=activation_end/gstd-mode_multiply/gstd_0.1 --networks=vgat --train_samples=8 --path=vnn_gat/iv_baselines_f0x3 --init_vnn_from=models/baselines/gat --init_vnn_from_original=False --INIT_WEIGHTS=fill:stds:0.001:0.001 --global_std_mode=multiply --activation_mode=end --batch_norm_mode=end --ignore_existing=True --GLOBAL_STD=0.1 --test_with_uncertainty=False --draw_uncertainty_graphs=False --test_samples=[20] --seeds=[1] --runs_per_variational_model=1
```

## Methods
Baseline spatial methods (GAT, GCN) and their varaitional versions (VGAT, VGCN) are implemented in `./networks`. Variational models start with a `vnn` prefix. 

## Insider influence data set

An illustration of the code that is used to construct the final data sets from raw data are in files public_data_generation_illustation.py and public_dataset_anonymization.py.

The anonymized data is available online in figshare: 10.6084/m9.figshare.20310240

The data set is composed of eight different parts. Each part differs depending on the prediction horizon (Simultaneous or Lead-Lag), prediction window (Daily or Weekly), and type of trading (Buy or Sell). Each observation in the data sets speaks about the trading behavior in the local neighborhood of the ego-investor (an investor whose trading behavior is aimed to be predicted). Each observation contains the following information:

* **Adjacency of matrix** (Public_Graphs.csv) of the local neighborhood of the ego investor. The adjacency matrix is composed by selecting 49 neighbors in the local neighborhood of the ego investor. The neighbor selection is made leveraging random walk with a restart.
* **Influence features** (Public_Influence_Features.csv) of the local neighborhood of the ego investor. Here the binary flags indicate whether neighboring investors of the ego investor have traded in a specific direction.
* **Normalized embedding** (Public_Normalized_Embedding.csv) of the nodes in the local neighborhood of the ego investor. Here the embedding encodes the position of neighboring nodes in the overall social network.
* **Labels** (Public_Labels.csv) that indicate if the ego investor has traded the same security as at least one other investor in her local neighborhood in the same direction, in the same (Simultaneous) or subsequent (Lead-lag) period.

Additionally, there is information about:
* **Distances** (Public_Distances.csv) between the ego investor and the insiders of the security for which the observation is recorded.
    * 0 - the investor is an insider in the company whose security is the subject of a given observation.
    * 1 - the investor is one social connection away from an insider of the security, which is the subject of a given observation,
    * ...
* **Family flags** (Public_Family_flag.csv) indicate whether the ego investor is not an insider but a family member or a related company to some insider.
    * 0 - ego investor is not a family member nor a related company to an insider. It means that the ego investor is herself an insider in some company.
    * 1 - ego investor is a family member or a related company to some insider.
* **Own company flags** (Public_Own_Company_flag.csv) indicate if the ego investor is an insider (or a family member or related company of the insider) concerning the security for which the observation is recorded.
    * 0 - concerning the security for which the observation is recorded, the **ego investor is neither an insider nor a family member or a related company to an insider** of the company that issued the security.
    * 1 - concerning the security for which the observation is recorded, the ego investor is either an insider, a family member, or a related company to an insider of the company that issued the security.
