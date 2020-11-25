import argparse as ap
import sys
from typing import Dict, List, Tuple

import yaml as y
import numpy as np

from src import options as o, parser as p, util as u, grid_search as g
from src.model import embeddings as e, data as d, model_eval as m

"""Module for evaluating and / or training a final model from a large number of potential hyperparameters"""


def load_hyperparams_file(path: str) -> Dict[str, List[g.HyperparamVals]]:
    with open(path, 'r') as f:
        # TODO double check this
        return y.load(f.read())


def init_parser() -> ap.ArgumentParser:
    parser = p.init_common_parser()
    parser.add_argument('-p', '--hyperparams_file', required=True, help='path to file of hyperparameter values')
    return parser


def hyperparameter_search(test_data: List[List[int]], hyperparams: Dict[str, List[g.HyperparamVals]]) \
        -> Tuple[Dict[str, g.HyperparamVals], int]:
    best_params, best_score = None, 0
    for hyperparam_set in g.hyperparameter_grid_search(hyperparams):
        model = o.init_model(model_type, layers=hyperparam_set['layers'], vocab_size=unique_word_count,
                             sent_len=max_sent_len, embedding_len=embedding_len)
        test_loss, test_acc = m.eval_model(model, test_data, max_sent_len, cross_val_folds,
                                           batch_size=hyperparam_set['batch_size'],
                                           epochs=hyperparam_set['epochs'])

        if best_params is None or test_acc > best_score:
            best_params = hyperparam_set

    return best_params, best_score


if __name__ == '__main__':
    args = init_parser().parse_args(sys.argv[1:])
    cross_val_folds = args.cross_val_folds
    model_type = args.model
    hyperparams = load_hyperparams_file(args.hyperparams_path)
    data = o.load_data(args.data)
    max_sent_len = e.get_longest_sent_size(data)
    unique_word_count = e.vocab_size(data) + 1
    encoded_texts, tokenizer = e.get_tokenizer(data)

    embedding_len = u.get_embedding_len(args.embedding_len, unique_word_count)

    evaluate, final = u.eval_regime(args.regime)

    # TODO learning rate(s),
    #  TODO train test validation splits
    # TODO model training verobsity

    if evaluate:
        # TODO separate out inner and outer cross validation folds?
        # TODO remove data augmentation from outer loop
        outer_cross_valid = d.create_cross_valid_folds(encoded_texts, max_sent_len, args.cross_valid_folds)
        for i, (aug_train_data, reg_train_data, test_data) in enumerate(outer_cross_valid):
            reg_train_data = np.unique(reg_train_data)
            best_hyperparams, best_score = hyperparameter_search(reg_train_data, hyperparams)

            print(f'cross validation fold {i}, selected params: {best_hyperparams}')
            # TODO reinit final model and test on test data
            # TODO print out relevant info
            best_model = o.init_model(model_type, layers=best_hyperparams['layers'], vocab_size=unique_word_count,
                                      sent_len=max_sent_len, embedding_len=embedding_len)
            best_model.train(aug_train_data, reg_train_data, batch_size=best_hyperparams['batch_size'],
                             epochs=best_hyperparams['epochs'])
            fold_loss, fold_acc = best_model.eval(test_data)
            print(f'cross validation fold {i}: test loss {fold_loss}, test accuracy {fold_acc}')

    if final:
        # TODO hyperparam search
        # TODO train on full dataset
        pass
