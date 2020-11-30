import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
from scipy.special import softmax

from tqdm import tqdm
from utils import wln_loss, regio_acc, lr_multiply_ratio, parse_args

args, dataloader, classifier = parse_args()
reactivity_data = pd.read_csv(args.data_path, index_col=0)

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

    train_gen = dataloader(train_smiles, train_products, train_rxn_id, args.batch_size)
    train_steps = np.ceil(len(train_smiles) / args.batch_size).astype(int)

    valid_gen = dataloader(valid_smiles, valid_products, valid_rxn_id, args.batch_size)
    valid_steps = np.ceil(len(valid_smiles) / args.batch_size).astype(int)
    for x, _ in dataloader([train_smiles[0]], [train_products[0]], [train_rxn_id[0]], 1):
        x_build = x
else:
    test = reactivity_data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_products = test.products_run.values

    test_gen = dataloader(test_smiles, test_products, test_rxn_id, args.batch_size, predict=True)
    test_steps = np.ceil(len(test_smiles) / args.batch_size).astype(int)

# need an input to initialize the graph network
    for x in dataloader([test_smiles[0]], [test_products[0]], [test_rxn_id[0]], 1, predict=True):
        x_build = x

save_name = os.path.join(args.model_dir, 'best_model.hdf5')
if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)

model = classifier(args.feature, args.depth, output_dim=5)
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

reduce_lr = LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

callbacks = [checkpoint, reduce_lr]

if not args.predict:
    hist = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=50,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=10
    )
else:
    predicted = []
    for x in tqdm(test_gen, total=int(len(test_smiles) / args.batch_size)):
        out = model.predict_on_batch(x)
        predicted.append(out)

    predicted = np.concatenate(predicted, axis=0)
    num_outcomes = [len(x.split('.')) for x in test_products]
    rxn_split = np.cumsum(num_outcomes)
    predicted = np.split(predicted, rxn_split)[:-1]
    predicted = [softmax(x.reshape(-1, 1)).reshape(-1).tolist() for x in predicted]

    test_predicted = pd.DataFrame({'rxn_id': test_rxn_id, 'predicted': predicted})
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test_predicted.to_csv(os.path.join(args.output_dir, 'predicted.csv'))


