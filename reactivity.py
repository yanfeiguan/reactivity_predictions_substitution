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

if args.model == 'ml_QM_GNN':
    from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
    from predict_desc.predict_desc import predict_desc, reaction_to_reactants
    from predict_desc.post_process import min_max_normalize
else:
    if args.model == 'QM_GNN':
        from QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
        initialize_qm_descriptors(path=args.desc_path)

if not args.predict:
    splits = args.splits
    test_ratio = splits[0]/sum(splits)
    valid_ratio = splits[1]/sum(splits[1:])
    test = reactivity_data.sample(frac=test_ratio)
    valid = reactivity_data[~reactivity_data.reaction_id.isin(test.reaction_id)].sample(frac=valid_ratio, random_state=1)
    train = reactivity_data[~(reactivity_data.reaction_id.isin(test.reaction_id) |
                              reactivity_data.reaction_id.isin(valid.reaction_id))]

    if args.model == 'ml_QM_GNN':
        df = predict_desc(args, normalize=False)
        train_reactants = reaction_to_reactants(train['rxn_smiles'].tolist())
        df, _ = min_max_normalize(df, train_smiles=train_reactants)
        initialize_qm_descriptors(df=df)

    train_rxn_id = train['reaction_id'].values
    train_smiles = train.rxn_smiles.str.split('>', expand=True)[0].values
    train_products = train.products_run.values

    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid.rxn_smiles.str.split('>', expand=True)[0].values
    valid_products = valid.products_run.values

    train_gen = dataloader(train_smiles, train_products, train_rxn_id, args.selec_batch_size)
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)

    valid_gen = dataloader(valid_smiles, valid_products, valid_rxn_id, args.selec_batch_size)
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)
    for x, _ in dataloader([train_smiles[0]], [train_products[0]], [train_rxn_id[0]], 1):
        x_build = x
else:
    test = reactivity_data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_products = test.products_run.values

    if args.model == 'ml_QM_GNN':
        df = predict_desc(args)
        initialize_qm_descriptors(df=df)

    test_gen = dataloader(test_smiles, test_products, test_rxn_id, args.selec_batch_size, predict=True)
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

# need an input to initialize the graph network
    for x in dataloader([test_smiles[0]], [test_products[0]], [test_rxn_id[0]], 1, predict=True):
        x_build = x

save_name = os.path.join(args.model_dir, 'best_model.hdf5')

model = classifier(args.feature, args.depth)
opt = tf.keras.optimizers.Adam(lr=args.ini_lr, clipnorm=5)
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
        train_gen, steps_per_epoch=train_steps, epochs=args.selec_epochs,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers
    )
else:
    predicted = []
    for x in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
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


