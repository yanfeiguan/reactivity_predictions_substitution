# Reactivity predictions for substitution reactions

This repository contains code and model for predicting regio-selectivity for substitution reactions as described in 
[Regio-Selectivity Prediction with a Machine-Learned Reaction Representation and On-the-Fly Quantum Mechanical Descriptors](https://chemrxiv.org/articles/preprint/Regio-Selectivity_Prediction_with_a_Machine-Learned_Reaction_Representation_and_On-the-Fly_Quantum_Mechanical_Descriptors/12907316)

## Requirements

1. python 3.7
2. tensorflow 2.0.0
3. rdkit
3. [qmdesc](https://github.com/yanfeiguan/chemprop-atom-bond) (https://github.com/yanfeiguan/qmdesc) (python package for predicting QM descriptors on the fly)

### Conda environment
To set up a conda environment:
```
conda env create --name test --file environment.yml
```

## Data

In order to train the model, you must provide training data containing reactions (as reaction SMILES with mapped atom number) and 
potential products (as molecular SMILES strings with mapped atom number). 

The data file must be a **CSV file that must include reaction_id, rxn_smiles, and products_run in the header row**. An example input file is provided as in [data_example.csv](./data_example.csv)
```
,reaction_id,rxn_smiles,products_run
0,244c06c0151311ea81f9b7db9d39a498,[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10]>>[Br:2][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1,[Br:1][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1.[Br:1][c:6]1[cH:5][c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7]1.[Br:1][c:7]1[cH:6][cH:5][c:4]([OH:3])[c:9]([F:10])[cH:8]1.[Br:1][c:8]1[cH:7][cH:6][cH:5][c:4]([OH:3])[c:9]1[F:10]
1,2182e760151311ea81f9b7db9d39a498,[F:1][F:2].[NH2:3][c:4]1[cH:5][cH:6][nH:7][c:8](=[O:9])[n:10]1>>[F:2][c:5]1[c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7][cH:6]1,[F:1][c:5]1[c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7][cH:6]1.[F:1][c:6]1[cH:5][c:4]([NH2:3])[n:10][c:8](=[O:9])[nH:7]1
```

in which, rxn_smiles are the reaction SMILES. And products_run are the potential products (major.minor1.minor2.....).

### USPTO demo data
We provide three classes of reactions (aromatic CH functionalization, aromatic CX substitution, and other substitution reactions)
to demonstrate our eventual model for the regio-selectivity predictions (Figure 7 in our paper). The `.csv` file for three classes of reactions 
are provided in the [uspto_demo_data](./uspto_demo_data) directory.  

## Training
This repo provides two model architectures as described in the paper.

### GNN
A conventional graph neural network that relies only on the machine learned reaction representation of a given reaction. 
To train the model, run:
```
python reactivitiy.py -m GNN --data_path <path to the .csv file> --model_dir <directory to save the trained model> 
```

For example, to train the model on CH functionalization reactions to predict the regio-selectivity:
```angular2
python reactivitiy.py -m GNN --data_path uspto_demo_data/uspto_CH.csv --model_dir trained_model/GNN_uspto_CH
```

A checkpoint file, `best_model.hdf5`, will be saved in the `trained_model/GNN_uspto_CH` directory.

### ml-QM-GNN

This is the novel fusion model introduced in the paper, which combines machine learned reaction representation and on-the-fly
calculated QM descriptors. To use this architecture, the [Chemprop-atom-bond](https://github.com/yanfeiguan/chemprop-atom-bond) 
must be installed. To train the model, run:

```
python reactivitiy.py --data_path <path to the .csv file> --model_dir <directory to save the trained model> 
``` 

The `reactivity.py` use `ml-QM-GNN` mode by default. The workflow first predict QM atomic/bond descriptors for all reactants found in reactions.
The predicted descriptors will then be scaled between `[0, 1]` through min-max scaler. A dictionary containing scikit-learn scaler object will be saved 
as `scalers.pickle` in the `model_dir` for later predicting task. A checkpoint file, `best_model.hdf5` will also be saved in the `model_dir`

For example:
```angular2
python reactivitiy.py --data_path uspto_demo_data/uspto_CH.csv --model_dir trained_model/ml_QM_GNN_uspto_CH
```

## Predicting
To use the trained model, run:

```
python reactivitiy -m <mode> --data_path <path to the predicting .csv file> --model_dir <directory containing the trained model> -p 
```

where `data_path` is path to the predicting `.csv` file, whose format is the same as the one discussed. `model_dir` is the directory holding the trained model. 
The model must be named as `best_model.hdb5` and stores parameters only. The `model_dir` must also include a `scalers.pickle` under `ml_QM_GNN` mode as discussed in the
[training](#Training) session.

We provide models trained on Pistachio regio-selective reactions, which are stored in the [trained_model](./trained_model). For example:
```angular2
python reactivitiy.py -m ml_QM_GNN --data_path uspto_demo_data/uspto_CH.csv --model_dir trained_model/ml_QM_GNN_CH -p 
``` 

The predicted result will be saved as a '.csv' file in the user specified output directory by the flag `--output_dir`, 
which is `output` by default. An example of the predicted selectivity:
```angular2
,rxn_id,predicted
0,86,"[0.9857026934623718, 0.004333616700023413, 0.00996393896639347]"
1,126,"[0.9388353824615479, 0.06116437166929245]"
2,7220,"[0.9834543466567993, 0.016545617952942848]"
```
where predicted column is the softmaxed selectivity score for each potential products. 