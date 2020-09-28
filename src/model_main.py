import argparse as ap

import src.util as u

"""Program for training and evaluating different autoencoders based on user args"""


class ModelOptions(u.GenericOptions):
    Plain = 'plain'
    LSTM = 'lstm'
    LSTMAttention = 'lstm-attention'
    Transformer = 'transformer'


class DataOptions(u.GenericOptions):
    CardTexts = 'card-texts'
    CardTypes = 'card-type'
    ManaCosts = 'mana-costs'
    FlavorTexts = 'flavor-texts'


class TrainingRegimes(u.GenericOptions):
    Eval = 'eval'
    Final = 'final'
    EvalAndFinal = 'eval-and-final'


def get_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=ModelOptions.to_option, help='type of model to train')
    parser.add_argument('-d', '--data', required=True, type=DataOptions.to_option, help='dataset to use on autoencoder')
    parser.add_argument('-r', '--regime', type=TrainingRegimes.to_option,
                        help='training regime for model - evaluation and/or final model')
    parser.add_argument('-e', '--epochs', default=8, help='max epochs for training')
    parser.add_argument('-k', '--cross-val-folds', default=5, help='folds for cross validation')
    parser.add_argument('-l', '--layers', default=[32, 16], nargs='+',
                        help='num of nodes / LSTM units per layer, for plain and LSTM architectures respectively')

    return parser


if __name__ == '__main__':
    print(get_parser().parse_args(['-h']))