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

The data file must be be a **CSV file with a fixed header row**. An example input file is provided as [data_example.csv](./data_example.csv)


###GNN
A conventional graph 