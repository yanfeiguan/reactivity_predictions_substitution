import os, sys
#sys.path.append(os.path.join(os.getcwd(), '../'))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import pandas as pd
from graph_utils.mol_graph import smiles2graph_list, get_bond_edits
from graph_utils.ioutils_direct import get_bin_feature, get_bond_label, smiles2graph_list_bin
from WLN.models import WLNPairwiseAtomClassifier
from random import shuffle
import argparse
from WLN.data_loading import Graph_DataLoader
from scipy.special import softmax
from rdkit import rdBase
from tqdm import tqdm
import pickle

rdBase.DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--restart', action='store_true')
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-m', '--modelpath')
parser.add_argument('--k_fold', default=10, type=int)
parser.add_argument('-f', '--feature', default=50, type=int)
parser.add_argument('-d', '--depth', default=3, type=int)
parser.add_argument('-dp', '--data_path', default='data/regio_nonstereo_12k_QM', type=str)
parser.add_argument('--ini_lr', default=0.001, type=float)
parser.add_argument('--lr_ratio', default=0.95, type=float)
parser.add_argument('-w', '--workers', default=5, type=int)
parser.add_argument('-s', '--sample', type=int)
args = parser.parse_args()

batch_size = 10
top = 100

df = pd.read_csv(args.data_path, index_col=0)
df = df.sample(frac=1, random_state=1)


def wln_loss(y_true, y_pred):

    #softmax cross entropy
    flat_label = K.cast(K.reshape(y_true, [-1]), 'float32')
    flat_score = K.reshape(y_pred, [-1])
    #return tf.nn.softmax_cross_entropy_with_logits(labels=flat_label, logits=flat_score)

    reaction_seg = K.cast(tf.math.cumsum(flat_label), 'int32') - tf.constant([1], dtype='int32')

    max_seg = tf.gather(tf.math.segment_max(flat_score, reaction_seg), reaction_seg)
    exp_score = tf.exp(flat_score-max_seg)

    softmax_denominator = tf.gather(tf.math.segment_sum(exp_score, reaction_seg), reaction_seg)
    softmax_score = exp_score/softmax_denominator
    #softmax_score = tf.expand_dims(exp_score/softmax_denominator, 0)
    #flat_label = tf.expand_dims(flat_label, 0)

    #score_op = tf.print(tf.losses.categorical_crossentropy(flat_label, softmax_score), summarize=-1, output_stream=sys.stdout)
    #softmax_score_op = tf.print(softmax_score, summarize=-1, output_stream=sys.stdout)
    softmax_score = tf.clip_by_value(softmax_score, K.epsilon(), 1-K.epsilon())
    try:
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))/flat_score.shape[0]
    except:
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))

    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_label, logits=flat_score)
    #return K.sum(loss, axis=-1, keepdims=False)/K.cast(tf.size(flat_label), 'float32')


def regio_acc(y_true_g, y_pred):
    y_true_g = K.reshape(y_true_g, [-1])
    y_pred = K.reshape(y_pred, [-1])

    #y_true_g_op = tf.print(y_true_g, summarize=-1, output_stream=sys.stdout)
    #y_pred_op = tf.print(tf.nn.softmax(y_pred), summarize=-1, output_stream=sys.stdout)

    reaction_seg = K.cast(tf.math.cumsum(y_true_g), 'int32') - tf.constant([1], dtype='int32')

    top_score = tf.math.segment_max(y_pred, reaction_seg)
    major_score = tf.gather(y_pred, tf.math.top_k(y_true_g, tf.size(top_score))[1])

    match = tf.equal(top_score, major_score)
    return tf.reduce_sum(K.cast(match, 'float32'))/K.cast(tf.size(top_score), 'float32')


def top_10_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=10)


def top_20_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=20)


def top_100_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=100)


if not os.path.exists(args.modelpath):
    os.mkdir(args.modelpath)



def lr_multiply_ratio(initial_lr, lr_ratio):
    def lr_multiplier(idx):
        return initial_lr*lr_ratio**idx
    return lr_multiplier


# split df into k_fold groups
k_fold_arange = np.linspace(0, len(df), args.k_fold+1).astype(int)

score = []
for i in range(args.k_fold):
    test = df[k_fold_arange[i]:k_fold_arange[i+1]]
    valid = df[~df.reaction_id.isin(test.reaction_id)].sample(frac=1/(args.k_fold-1), random_state=1)
    train = df[~(df.reaction_id.isin(test.reaction_id) | df.reaction_id.isin(valid.reaction_id))]
    if args.sample:
        try:
            train = train.sample(n=args.sample, random_state=1)
        except:
            pass

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

    model = WLNPairwiseAtomClassifier(args.feature, args.depth, output_dim=5)
    opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5)
    model.compile(
        optimizer=opt,
        loss=wln_loss,
        metrics=[
            regio_acc,
        ],
        #clipnorm=5.
    )
    #model.fit(x_build, y_build)

    save_name = os.path.join(args.modelpath, 'best_model_{}.hdf5'.format(i))
    checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

    callbacks = [checkpoint, reduce_lr]

    print('training the {}th iteration'.format(i))
    hist = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=30,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers,
        verbose=2,
    )

    with open(os.path.join(args.modelpath, 'history_{}.pickle'.format(i)), 'wb') as hist_pickle:
        pickle.dump(hist.history, hist_pickle)

    model.load_weights(save_name)

    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_products = test.products_run.values

    test_gen = Graph_DataLoader(test_smiles, test_products, test_rxn_id, batch_size, shuffle=False)
    test_steps = np.ceil(len(test_smiles) / batch_size).astype(int)

    predicted = []
    for x, y in tqdm(test_gen, total=int(len(test_smiles) / batch_size)):
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
    test_predicted.to_csv(os.path.join(args.modelpath, 'test_predicted_{}.csv'.format(i)))

    success_rate = len(test_predicted[test_predicted.predicted.apply(lambda x: x[0] > max(x[1:]))])/len(test_predicted)
    score.append(success_rate)
    print('success rate for iter {}: {}'.format(i, success_rate))
    print(score)

print('success rate for {}-fold cross-validation: {}'.format(args.k_fold, np.mean(np.array(score))))




