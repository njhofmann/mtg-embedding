import argparse as ap


def init_common_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=o.ModelOptions.to_option, help='type of model to train')
    parser.add_argument('-d', '--data', required=True, type=o.DataOptions.to_option,
                        help='dataset to design an autoencoder around')
    parser.add_argument('-r', '--regime', required=True, type=o.TrainingRegimes.to_option,
                        help='training regime for model - evaluation and/or final model')
    parser.add_argument('-el', '--embedding_len', default=-1, type=int, help='size of input embedding length')
    parser.add_argument('-k', '--cross_val_folds', default=5, help='folds for cross validation')
    return parser