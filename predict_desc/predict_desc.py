import os

import numpy as np
import pandas as pd
from rdkit import Chem

from chemprop.parsing import add_predict_args, modify_predict_args
from chemprop.train import make_predictions
from .post_process import check_chemprop_out, min_max_normalize

def predict_desc(args):
    import chemprop
    chemprop_root = os.path.dirname(os.path.dirname(chemprop.__file__))

    #trick chemprop
    args.test_path = 'foo'
    args.checkpoint_path = os.path.join(chemprop_root, 'trained_model', 'QM_137k.pt')
    modify_predict_args(args)

    def num_atoms_bonds(smiles):
        m = Chem.MolFromSmiles(smiles)

        m = Chem.AddHs(m)

        return len(m.GetAtoms()), len(m.GetBonds())


    # predict descriptors for reactants in the reactions
    reactivity_data = pd.read_csv(args.data_path, index_col=0)
    reactants = set()
    for _, row in reactivity_data.iterrows():
        rs, _, _ = row['rxn_smiles'].split('>')
        rs = rs.split('.')
        for r in rs:
            reactants.add(r)
    reactants = list(reactants)

    print('Predicting descriptors for reactants...')
    test_preds, test_smiles = make_predictions(args, smiles=reactants)

    partial_charge = test_preds[0]
    partial_neu = test_preds[1]
    partial_elec = test_preds[2]
    NMR = test_preds[3]

    bond_order = test_preds[4]
    bond_distance = test_preds[5]

    n_atoms, n_bonds = zip(*[num_atoms_bonds(x) for x in reactants])

    partial_charge = np.split(partial_charge.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    partial_neu = np.split(partial_neu.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    partial_elec = np.split(partial_elec.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    NMR = np.split(NMR.flatten(), np.cumsum(np.array(n_atoms)))[:-1]

    bond_order = np.split(bond_order.flatten(), np.cumsum(np.array(n_bonds)))[:-1]
    bond_distance = np.split(bond_distance.flatten(), np.cumsum(np.array(n_bonds)))[:-1]

    df = pd.DataFrame(
        {'smiles': reactants, 'partial_charge': partial_charge, 'fukui_neu': partial_neu, 'fukui_elec': partial_elec,
         'NMR': NMR, 'bond_order': bond_order, 'bond_length': bond_distance})

    invalid = check_chemprop_out(df)
    # FIXME remove invalid molecules from reaction dataset
    print(invalid)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    df.to_pickle(os.path.join(args.output_path, 'reactants_descriptors.pickle'))
    df = min_max_normalize(df, args.ref_data_path)
    df.to_pickle(os.path.join(args.output_path, 'reactants_descriptors_norm.pickle'))

    return df