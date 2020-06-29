import os, sys
#sys.path.append(os.path.join(os.getcwd(), '../'))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import pandas as pd

from ml_QM_GNN.WLN.models import WLNPairwiseAtomClassifier
from random import shuffle
import argparse
from ml_QM_GNN.WLN.data_loading import Graph_DataLoader
from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
from scipy.special import softmax
from rdkit import rdBase
from tqdm import tqdm

from data_process import check_chemprop_out, min_max_normalize

from chemprop.parsing import add_predict_args, modify_predict_args
from chemprop.train import make_predictions

from rdkit import Chem

#find chemprop root path
import chemprop
chemprop_root = os.path.dirname(os.path.dirname(chemprop.__file__))

rdBase.DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
add_predict_args(parser)
parser.add_argument('-r', '--restart', action='store_true')
parser.add_argument('-p', '--predict', action='store_true')
parser.add_argument('-m', '--model_path', default='trained_model')
parser.add_argument('-o', '--output_path', default='output')
parser.add_argument('-f', '--feature', default=50, type=int)
parser.add_argument('-d', '--depth', default=3, type=int)
parser.add_argument('-dp', '--data_path', default='data/regio_nonstereo_12k_QM', type=str)
parser.add_argument('-rdp', '--ref_data_path', default=None, type=str)
parser.add_argument('--ini_lr', default=0.001, type=float)
parser.add_argument('--lr_ratio', default=0.95, type=float)
args = parser.parse_args()

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
#FIXME remove invalid molecules from reaction dataset
print(invalid)

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

df.to_pickle(os.path.join(args.output_path, 'reactants_descriptors.pickle'))
df = min_max_normalize(df, args.ref_data_path)
df.to_pickle(os.path.join(args.output_path, 'reactants_descriptors_norm.pickle'))
initialize_qm_descriptors(df=df)

batch_size = 10
top = 100
if not args.predict:
    test = reactivity_data.sample(frac=0.1)
    valid = reactivity_data[~reactivity_data.reaction_id.isin(test.reaction_id)].sample(frac=1/9, random_state=1)
    train = reactivity_data[~(reactivity_data.reaction_id.isin(test.reaction_id) |
                              reactivity_data.reaction_id.isin(valid.reaction_id))]

    train_rxn_id = train['reaction_id'].values
    train_smiles = train.rxn_smiles.str.split('>', expand=True)[0].values
    train_products = train.products_run.values

    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid.rxn_smiles.str.split('>', expand=True)[0].values
    valid_products = valid.products_run.values

    train_gen = Graph_DataLoader(train_smiles, train_products, train_rxn_id, batch_size)
    train_steps = np.ceil(len(train_smiles) / batch_size).astype(int)

    valid_gen = Graph_DataLoader(valid_smiles, valid_products, valid_rxn_id, batch_size)
    valid_steps = np.ceil(len(valid_smiles) / batch_size).astype(int)
    for x, _ in Graph_DataLoader([train_smiles[0]], [train_products[0]], [train_rxn_id[0]], 1):
        x_build = x
else:
    test = reactivity_data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_products = test.products_run.values

    test_gen = Graph_DataLoader(test_smiles, test_products, test_rxn_id, batch_size, shuffle=False)
    test_steps = np.ceil(len(test_smiles) / batch_size).astype(int)

# need an input to initialize the graph network
    for x, _ in Graph_DataLoader([test_smiles[0]], [test_products[0]], [test_rxn_id[0]], 1):
        x_build = x


def wln_loss(y_true, y_pred):

    #softmax cross entropy
    flat_label = K.cast(K.reshape(y_true, [-1]), 'float32')
    flat_score = K.reshape(y_pred, [-1])

    reaction_seg = K.cast(tf.math.cumsum(flat_label), 'int32') - tf.constant([1], dtype='int32')

    max_seg = tf.gather(tf.math.segment_max(flat_score, reaction_seg), reaction_seg)
    exp_score = tf.exp(flat_score-max_seg)

    softmax_denominator = tf.gather(tf.math.segment_sum(exp_score, reaction_seg), reaction_seg)
    softmax_score = exp_score/softmax_denominator

    softmax_score = tf.clip_by_value(softmax_score, K.epsilon(), 1-K.epsilon())
    try:
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))/flat_score.shape[0]
    except:
        #during initialization
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))


def regio_acc(y_true_g, y_pred):
    y_true_g = K.reshape(y_true_g, [-1])
    y_pred = K.reshape(y_pred, [-1])

    reaction_seg = K.cast(tf.math.cumsum(y_true_g), 'int32') - tf.constant([1], dtype='int32')

    top_score = tf.math.segment_max(y_pred, reaction_seg)
    major_score = tf.gather(y_pred, tf.math.top_k(y_true_g, tf.size(top_score))[1])

    match = tf.equal(top_score, major_score)
    return tf.reduce_sum(K.cast(match, 'float32'))/K.cast(tf.size(top_score), 'float32')


save_name = args.model_path
save_dir = os.path.dirname(save_name)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model = WLNPairwiseAtomClassifier(args.feature, args.depth, output_dim=5)
opt = tf.keras.optimizers.Adam(lr=0.0007, clipnorm=5)
model.compile(
    optimizer=opt,
    loss=wln_loss,
    metrics=[
        regio_acc,
    ],
)
model.predict_on_batch(x_build)
model.summary()

if args.restart or args.predict:
    model.load_weights(save_name)

checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)


def lr_multiply_ratio(initial_lr, lr_ratio):
    def lr_multiplier(idx):
        return initial_lr*lr_ratio**idx
    return lr_multiplier


reduce_lr = keras.callbacks.LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

callbacks = [checkpoint, reduce_lr]

if not args.predict:
    model = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=50,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=10
    )
else:
    predicted = []
    for x, y in tqdm(test_gen, total=int(len(test_smiles)/batch_size)):
        out = model.predict_on_batch(x)
        out = np.reshape(out, [-1])
        predicted_rxn = []
        for y_predicted, y_true in zip(out, y):
            if y_true == 1 and predicted_rxn:
                predicted_rxn = softmax(predicted_rxn)
                predicted.append(list(predicted_rxn))

            if y_true == 1:
                predicted_rxn = []

            predicted_rxn.append(y_predicted)

        predicted_rxn = softmax(predicted_rxn)
        predicted.append(list(predicted_rxn))

    test_predicted = pd.DataFrame({'rxn_id': test_rxn_id, 'predicted': predicted})
    test_predicted.to_csv(os.path.join(args.output_path, 'predicted.csv'))


