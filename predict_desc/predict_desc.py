import os
import pickle

import pandas as pd
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler
from tqdm import tqdm

from .post_process import check_chemprop_out, min_max_normalize

def reaction_to_reactants(reactions):
    reactants = set()
    for r in reactions:
        rs = r.split('>')[0].split('.')
        reactants.update(set(rs))
    return list(reactants)


def predict_desc(args, normalize=True):

    def num_atoms_bonds(smiles):
        m = Chem.MolFromSmiles(smiles)

        m = Chem.AddHs(m)

        return len(m.GetAtoms()), len(m.GetBonds())


    # predict descriptors for reactants in the reactions
    reactivity_data = pd.read_csv(args.data_path, index_col=0)
    reactants = reaction_to_reactants(reactivity_data['rxn_smiles'].tolist())

    print('Predicting descriptors for reactants...')

    handler = ReactivityDescriptorHandler()
    descs = []
    for smiles in tqdm(reactants):
        descs.append(handler.predict(smiles))

    df = pd.DataFrame(descs)

    invalid = check_chemprop_out(df)
    # FIXME remove invalid molecules from reaction dataset
    print(invalid)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df.to_pickle(os.path.join(args.output_dir, 'reactants_descriptors.pickle'))
    save_dir = args.model_dir

    if not normalize:
        return df

    if not args.predict:
        df, scalers = min_max_normalize(df)
        pickle.dump(scalers, open(os.path.join(save_dir, 'scalers.pickle'), 'wb'))
    else:
        scalers = pickle.load(open(os.path.join(save_dir, 'scalers.pickle'), 'rb'))
        df, _ = min_max_normalize(df, scalers=scalers)

    df.to_pickle(os.path.join(args.output_dir, 'reactants_descriptors_norm.pickle'))

    return df
