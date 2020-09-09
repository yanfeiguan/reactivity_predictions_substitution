from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

def check_chemprop_out(df):
    invalid = []
    for _,r in df.iterrows():
        for c in ['partial_charge', 'fukui_neu', 'fukui_elec', 'NMR', 'bond_order', 'bond_length']:
            if np.any(pd.isna(r[c])):
                invalid.append(r['smiles'])
                break
    return invalid



def min_max_normalize(df, ref_df=None):
    if ref_df is None:
        ref_df = df
    ref_df['atoms'] = ref_df.smiles.apply(lambda x: get_atoms(x))
    df['atoms'] = df.smiles.apply(lambda x: get_atoms(x))
    # max-min trough atom types
    # NMR
    nmrs = np.concatenate(ref_df.NMR.tolist())
    atoms = np.concatenate(ref_df.atoms.tolist())
    minmax = {}
    for a, n in zip(atoms, nmrs):
        if a not in minmax:
            minmax[a] = [n]
        else:
            minmax[a].append(n)

    for k in minmax.keys():
        minmax[k] = [min(minmax[k]), max(minmax[k])]

    df['NMR'] = df.progress_apply(lambda x: minmax_by_element(x, minmax, 'NMR'), axis=1)

    df['bond_order_matrix'] = df.apply(lambda x: bond_to_matrix(x['smiles'], x['bond_order']), axis=1)
    df['distance_matrix'] = df.apply(lambda x: bond_to_matrix(x['smiles'], x['bond_length']), axis=1)

    #partial charge
    charges = np.concatenate(ref_df.partial_charge.tolist())
    min_charge = charges.min()
    max_charge = charges.max()

    df['partial_charge'] = df.partial_charge.apply(
        lambda x: (x - min_charge) / (max_charge - min_charge + np.finfo(float).eps))

    #fukui neu indices
    charges = np.concatenate(ref_df.fukui_neu.tolist())
    min_charge = charges.min()
    max_charge = charges.max()

    df['fukui_neu'] = df.fukui_neu.apply(
        lambda x: (x - min_charge) / (max_charge - min_charge + np.finfo(float).eps))

    # fukui elec indices
    charges = np.concatenate(ref_df.fukui_elec.tolist())
    min_charge = charges.min()
    max_charge = charges.max()

    df['fukui_elec'] = df.fukui_elec.apply(
        lambda x: (x - min_charge) / (max_charge - min_charge + np.finfo(float).eps))

    df = df[['smiles', 'partial_charge', 'fukui_neu', 'fukui_elec', 'NMR', 'bond_order_matrix', 'distance_matrix']]
    df = df.set_index('smiles')

    return df


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