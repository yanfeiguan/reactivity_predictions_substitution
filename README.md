# Reactivity predictions for substitution reactions

This repository contains code and model for predicting regio-selectivity for substitution reactions as described in 
paper: xxxxxxx.

## Requirements

1. python 3.7
2. [Chemprop-atom-bond](https://github.com/yanfeiguan/chemprop-atom-bond) (extra dependcy required)
3. tensorflow 2.3.0

## Available models

We provide three types of models to predict the selectivity.

## Data

In order to train the model, you must provide training data containing reactions (as reaction SMILES with mapped atom number) and 
potential products (as molecular SMILES strings with mapped atom number). 

The data file must be be a **CSV file with a fixed header row**. An example input file is provided as in [data_example.csv](./data_example.csv)
```
,reaction_id,rxn_smiles,products_run
0,244c06c0151311ea81f9b7db9d39a498,[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10]>>[Br:2][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1,[Br:1][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1.[Br:1][c:6]1[cH:5][c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7]1.[Br:1][c:7]1[cH:6][cH:5][c:4]([OH:3])[c:9]([F:10])[cH:8]1.[Br:1][c:8]1[cH:7][cH:6][cH:5][c:4]([OH:3])[c:9]1[F:10]
1,2182e760151311ea81f9b7db9d39a498,[F:1][F:2].[NH2:3][c:4]1[cH:5][cH:6][nH:7][c:8](=[O:9])[n:10]1>>[F:2][c:5]1[c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7][cH:6]1,[F:1][c:5]1[c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7][cH:6]1.[F:1][c:6]1[cH:5][c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7]1
```

in which, rxn_smiles are the reaction SMILES. And products_run are the potential products (major.minor1.minor2.....).

##Training
This repo provides two model architectures as described in the paper.

###GNN
A conventional graph neural network that relies only on the machine learned reaction representation of a given reaction. 
To train the model, run:
```
python reactivitiy.py -m GNN --data_path data_example.csv --model_path <path> 
```

where `<path>` is the path to a checkpoint file, in which you want to store the parameters of the trained network.

###ml-QM-GNN

This is the novel fusion model introduced in the paper, which combines machine learned reaction representation and on-the-fly
calculated QM descriptors. To use this architecture, the [Chemprop-atom-bond](https://github.com/yanfeiguan/chemprop-atom-bond) 
must be installed. To trainthe model, run:

```
python reactivitiy.py -m ml_QM_GNN --data_path data_example.csv --model_path <path> 
``` 

## Predicting
Users can also use the trained model to predict selectivities for their own reactions. To predict, run:

```
python reactivitiy -m ml_QM_GNN --data_path data_example.csv --model_path <path> -p 
```

where `<path>` is path to the checkpoint file storing the parameters of the model

We have pretrained model stored in the [trained_model](./trained_model), corresponding to the three classes of 
substitution reactions discussed in the paper. 

The predicted result will be saved as a '.csv' file in the user specified output directory by the flag `--output_dir`, 
which is `output` by default.