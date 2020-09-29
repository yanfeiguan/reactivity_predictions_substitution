from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

tqdm.pandas()

GLOBAL_SCALE = ['partial_charge', 'fukui_neu', 'fukui_elec']
ATOM_SCALE = ['NMR']


def check_chemprop_out(df):
    invalid = []
    for _,r in df.iterrows():
        for c in ['partial_charge', 'fukui_neu', 'fukui_elec', 'NMR', 'bond_order', 'bond_length']:
            if np.any(pd.isna(r[c])):
                invalid.append(r['smiles'])
                break
    return invalid


def min_max_normalize(df, scalers=None):

    if scalers is None:
        scalers = get_scaler(df)

    for column in GLOBAL_SCALE:
        scaler = scalers[column]
        df[column] = df[column].apply(lambda x: scaler.transform(x.reshape(-1, 1)).reshape(-1))

    def min_max_by_atom(atoms, data, scaler):
        data = [scaler[a].transform(np.array([[d]]))[0][0] for a, d in zip(atoms, data)]
        return np.array(data)

    if ATOM_SCALE:
        print('postprocessing atom-wise scaling')
        df['atoms'] = df.smiles.apply(lambda x: get_atoms(x))
        for column in ATOM_SCALE:
            df[column] = df.progress_apply(lambda x: min_max_by_atom(x['atoms'], x[column], scalers[column]), axis=1)

    df['bond_order_matrix'] = df.apply(lambda x: bond_to_matrix(x['smiles'], x['bond_order']), axis=1)
    df['distance_matrix'] = df.apply(lambda x: bond_to_matrix(x['smiles'], x['bond_length']), axis=1)

    df = df[['smiles', 'partial_charge', 'fukui_neu', 'fukui_elec', 'NMR', 'bond_order_matrix', 'distance_matrix']]
    df = df.set_index('smiles')

    return df, scalers


def get_scaler(df):
    scalers = {}
    for column in GLOBAL_SCALE:
        scaler = MinMaxScaler()
        data = np.concatenate(df[column].tolist()).reshape(-1, 1)

        scaler.fit(data)
        scalers[column] = scaler

    if ATOM_SCALE:
        atoms = df.smiles.apply(lambda x: get_atoms(x))
        atoms = np.concatenate(atoms)
        for column in ATOM_SCALE:
            data = np.concatenate(df[column].tolist())

            data = pd.DataFrame({'atoms': atoms, 'data': data})
            data = data.groupby('atoms').agg({'data': lambda x: list(x)})['data'].apply(lambda x: np.array(x)).to_dict()

            scalers[column] = {}
            for k, d in data.items():
                scaler = MinMaxScaler()
                scalers[column][k] = scaler.fit(d.reshape(-1, 1))

    return scalers


def bond_to_matrix(smiles, bond_vector):
    m = Chem.MolFromSmiles(smiles)

    m = Chem.AddHs(m)

    bond_matrix = np.zeros([len(m.GetAtoms()), len(m.GetAtoms())])
    for i, bp in enumerate(bond_vector):
        b = m.GetBondWithIdx(i)
        bond_matrix[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = bond_matrix[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = bp

    return bond_matrix


def get_atoms(smiles):
    m = Chem.MolFromSmiles(smiles)

    m = Chem.AddHs(m)

    atoms = [x.GetSymbol() for x in m.GetAtoms()]

    return atoms


def minmax_by_element(r, minmax, target):
    target = r[target]
    elements = r['atoms']
    for i, a in enumerate(elements):
        target[i] = (target[i] - minmax[a][0]) / (minmax[a][1] - minmax[a][0] + np.finfo(float).eps)

    return target
