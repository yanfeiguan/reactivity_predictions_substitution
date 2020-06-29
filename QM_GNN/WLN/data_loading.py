# import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from random import shuffle
from graph_utils.mol_graph import get_bond_edits, smiles2graph_pr, pack1D, pack2D, pack2D_withidx, get_mask
from graph_utils.ioutils_direct import binary_features_batch
from rdkit import Chem

# get QM descriptor

def graph_batch_generator(smiles, labels, batch_size):
    steps = int(np.ceil(len(smiles) / batch_size))
    zipped = list(zip(smiles, labels))
    shuffle(zipped)
    smiles, labels = zip(*zipped)
    while True:
        for n in range(steps):
            start = n * batch_size
            stop = start + batch_size
            graph_inputs = list(smiles2graph_list_bin(smiles[start:stop]))
            max_natoms = graph_inputs[0].shape[1]
            # binary_features = np.array([get_bin_feature(smi, max_natoms) for smi in smiles[start:stop]])
            bond_labels = []
            sp_labels = []
            for smi, edit in zip(smiles[start:stop], labels[start:stop]):
                l = get_bond_label(smi, edit, max_natoms)
                bond_labels.append(l[0])
                sp_labels.append(l[1])
            # graph_inputs.append(binary_features)
            labels_batch = labels[start:stop]
            # print('bond label', l[0].shape)
            yield (graph_inputs, np.array(bond_labels))


class Graph_DataLoader(Sequence):
    def __init__(self, smiles, products, rxn_id, batch_size, shuffle=True):
        self.smiles = smiles
        self.products = products
        self.rxn_id = rxn_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.atom_classes = {}

    def __len__(self):
        return int(np.ceil(len(self.smiles) / self.batch_size))

    def __getitem__(self, index):
        smiles_tmp = self.smiles[index * self.batch_size:(index + 1) * self.batch_size]
        products_tmp = self.products[index * self.batch_size:(index + 1) * self.batch_size]
        rxn_id_tmp = self.rxn_id[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(smiles_tmp, products_tmp, rxn_id_tmp)
        return x, y

    def on_epoch_end(self):
        if self.shuffle == True:
            zipped = list(zip(self.smiles, self.products, self.rxn_id))
            shuffle(zipped)
            self.smiles, self.products, self.rxn_id = zip(*zipped)

    def __data_generation(self, smiles_tmp, products_tmp, rxn_id_tmp):
        # idxfunc = lambda x:x.GetIntProp('molAtomMapNumber') - 1

        # res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_tmp))
        prs_extend = []
        labels_extend = []
        #r_pretrained_fatoms = []
        rxn_id_extend = []
        for r, ps, rxn_id in zip(smiles_tmp, products_tmp, rxn_id_tmp):
            size = len(ps.split('.'))
            #r_pretrained_fatoms.extend([pretrained_fatoms.loc[rxn_id]['atom_features']] * size)
            rxn_id_extend.extend([rxn_id]*size)
            prs_extend.extend([smiles2graph_pr(p, r, core_buffer=0) for p in ps.split('.')])
            labels_extend.extend([1] + [0] * (size - 1))

        rs_extends, smiles_extend = zip(*prs_extend)

        # generate molecular representations for reactants
        '''
        rs_graphs = []
        for r_pretrained_fatom, rsmiles, rxn_id, rs_extend in zip(r_pretrained_fatoms,
                                                                  smiles_extend, rxn_id_extend,
                                                                  rs_extends):

            fatom_index = \
                {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rsmiles).GetAtoms()}

            rsmiles = rsmiles.split('.')
            dicts = []
            compounds_ids = []

            for smile in rsmiles:
                rxn = rxn_comp_df.loc[smile]
                dicts.append(rxn['atomNum_dict'])
                compounds_ids.append(rxn['compounds_ID'])

            pretrained_fatom = np.zeros([len(fatom_index), 38])
            for map_idx, idx in fatom_index.items():
                pretrained_fatom[idx, :] = r_pretrained_fatom[map_idx, :]

            rs_core_qm = np.zeros([len(fatom_index), 12])
            for dict, compound_id, smiles in zip(dicts, compounds_ids, rsmiles):
                qm_series = pd.read_pickle(os.path.join(qm_descriptors_folder, '{}.pickle'.format(compound_id)))

                partial_charge = qm_series['NPA'][:, :1]
                ao = qm_series['AO']
                lp = qm_series['long_pair']
                fukui_elec = qm_series['fukui_elec'][:, :1]
                fukui_neu = qm_series['fukui_neu'][:, :1]

                qm_descriptor = np.concatenate([partial_charge, ao, lp, fukui_elec, fukui_neu], axis=-1)
                for map_idx, idx in dict.items():
                    rs_core_qm[fatom_index[map_idx - 1], :] = qm_descriptor[idx, :]

            rs_fatom = rs_core_qm
            rs_extend = [rs_fatom] + list(rs_extend[1:])
            rs_graphs.append(rs_extend)
        '''

        fatom_list, fatom_qm_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = \
            zip(*rs_extends)
        res_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                            pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                            binary_features_batch(smiles_extend), pack1D(core_mask), pack2D(fatom_qm_list))

        return res_graph_inputs, np.array(labels_extend).astype('int32')
