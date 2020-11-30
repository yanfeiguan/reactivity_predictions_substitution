import argparse

from chemprop.parsing import add_predict_args

rdBase.DisableLog('rdApp.warning')


def parse_args():
    parser = argparse.ArgumentParser()
    add_predict_args(parser)
    parser.add_argument('-r', '--restart', action='store_true',
                        help='restart the training using the saved the checkpoint file')
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict reactivity for a given .csv file')
    parser.add_argument('-m', '--model', default='ml_QM_GNN', choices=['ml_QM_GNN', 'QM_GNN', 'GNN'],
                        help='model can be used')
    parser.add_argument('--model_dir', default='trained_model',
                        help='path to the checkpoint file of the trained model')
    parser.add_argument('--desc_path', default=None,
                        help='path to the file storing the descriptors (must be provided when using QM_GNN model)')
    parser.add_argument('-o', '--output_dir', default='output')
    parser.add_argument('-f', '--feature', default=50, type=int)
    parser.add_argument('-d', '--depth', default=4, type=int)
    parser.add_argument('-dp', '--data_path', default='data/regio_nonstereo_12k_QM', type=str)
    parser.add_argument('-rdp', '--ref_data_path', default=None, type=str)
    parser.add_argument('--ini_lr', default=0.001, type=float)
    parser.add_argument('--lr_ratio', default=0.95, type=float)
    args = parser.parse_args()

    if args.model == 'ml_QM_GNN':
        from ml_QM_GNN.WLN.data_loading import Graph_DataLoader
        from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
        from predict_desc.predict_desc import predict_desc
        from ml_QM_GNN.WLN.models import WLNPairwiseAtomClassifier
        df = predict_desc(args)
        initialize_qm_descriptors(df=df)
    else:
        if args.model == 'QM_GNN':
            from QM_GNN.WLN.data_loading import Graph_DataLoader
            from QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors
            from QM_GNN.WLN.models import WLNPairwiseAtomClassifier
            initialize_qm_descriptors(path=args.desc_path)
        elif args.model == 'GNN':
            from GNN.WLN.data_loading import Graph_DataLoader
            from GNN.WLN.models import WLNPairwiseAtomClassifier

    return args, Graph_DataLoader, WLNPairwiseAtomClassifier