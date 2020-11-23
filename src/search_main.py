import argparse as ap
import sys
from typing import Dict, List

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


if __name__ == '__main__':
    args = init_parser().parse_args(sys.argv[1:])
    hyperparams = load_hyperparams_file(args.hyperparams_path)
    data = o.load_data(args.data)
    max_sent_len = e.get_longest_sent_size(data)
    unique_word_count = e.vocab_size(data) + 1
    encoded_texts, tokenizer = e.get_tokenizer(data)

    # TODO separate out inner and outer cross validation folds?
    # TODO remove data augmentation from outer loop
    outer_cross_valid = d.create_cross_valid_folds(encoded_texts, max_sent_len, args.cross_valid_folds)
    for i, (aug_train_data, reg_train_data, test_data) in enumerate(outer_cross_valid):
        reg_train_data = np.unique(reg_train_data)
        best_params, best_score = None, 0
        for hyperparam_set in g.hyperparameter_grid_search(hyperparams):
            model = o.init_model()
            avg_loss, avg_acc = m.eval_model(model, )

            if best_params is None or avg_acc > best_score:
                best_params = hyperparam_set

        # TODO reinit final model and test on test data
        # TODO print out relevant info