# import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from random import shuffle
from ..graph_utils.mol_graph import get_bond_edits, smiles2graph_pr, pack1D, pack2D, pack2D_withidx, get_mask
from ..graph_utils.ioutils_direct import binary_features_batch


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
        prs_extend = []
        labels_extend = []

        rxn_id_extend = []
        for r, ps, rxn_id in zip(smiles_tmp, products_tmp, rxn_id_tmp):
            size = len(ps.split('.'))

            rxn_id_extend.extend([rxn_id]*size)
            prs_extend.extend([smiles2graph_pr(p, r, core_buffer=0) for p in ps.split('.')])
            labels_extend.extend([1] + [0] * (size - 1))

        rs_extends, smiles_extend = zip(*prs_extend)

        fatom_list, fatom_qm_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = \
            zip(*rs_extends)
        res_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                            pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                            binary_features_batch(smiles_extend), pack1D(core_mask), pack2D(fatom_qm_list))

        return res_graph_inputs, np.array(labels_extend).astype('int32')
